"""Validate finite-difference HVPs (and Lanczos/SLQ) against exact double-backward HVPs.

Small pure-PyTorch GPT (track-3 style: rotary, QK-norm, ReLU^2, softcap, untied head), trained
briefly with AdamW so we are not at init, then:
  1. gradient nondeterminism floor per precision mode
  2. FD-HVP relative error vs exact HVP (fp32 double backward) over eps, per precision mode, for
     random / top-eigvec / gradient directions
  3. Lanczos convergence of the top Ritz values (exact HVP), and FD-HVP Lanczos vs exact
  4. timings in units of one probe gradient

Precision modes for the gradient oracle (the thing FD differentiates):
  fp32   fp32 weights + fp32 matmuls (TF32 off)
  tf32   fp32 weights, TF32 matmuls
  bf16   fp32 master weights, bf16 autocast (weights rounded to bf16 at each matmul)
  fp8w   as bf16, plus per-tensor e4m3 weight quantization of the block matrices (WR-style FP8)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics import HVP, lanczos, slq_density, subspace_fraction, vdot, vnorm, vsub, vrandn_like, vscale, line_probe  # noqa

MODE = {"m": "fp32"}  # global precision mode read by the model


def load_tokens(path, n):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16, count=n, offset=1024).astype(np.int64))


def q_fp8(w: Tensor) -> Tensor:
    s = 448.0 / w.detach().abs().amax().clamp_min(1e-12)
    wq = (w.detach() * s).to(torch.float8_e4m3fn).float() / s
    return w + (wq - w).detach()


class Lin(nn.Module):
    def __init__(self, i, o, zero=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(o, i) if zero else torch.randn(o, i) / math.sqrt(i))

    def forward(self, x):
        w = self.weight
        if MODE["m"] == "wr":
            w = w + (w.detach().bfloat16().float() - w.detach())  # bf16 storage, as the WR banks
        if MODE["m"] in ("fp8w", "wr"):
            w = q_fp8(w)
        return F.linear(x, w.to(x.dtype) if x.dtype != torch.float32 else w)


def rotary(x, freq):
    T = x.size(1)
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    th = torch.outer(pos, freq)[None, :, None, :]
    c, s = th.cos(), th.sin()
    x1, x2 = x.float().chunk(2, -1)
    return torch.cat((x1 * c + x2 * s, -x1 * s + x2 * c), -1).type_as(x)


class Block(nn.Module):
    def __init__(self, d, hd=64):
        super().__init__()
        self.h, self.hd = d // hd, hd
        self.qkv = Lin(d, 3 * d)
        self.o = Lin(d, d, zero=True)
        self.fc = Lin(d, 4 * d)
        self.pr = Lin(4 * d, d, zero=True)
        f = (1 / 1024) ** torch.linspace(0, 1, steps=hd // 4)
        self.register_buffer("freq", torch.cat([f, f.new_zeros(hd // 4)]), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        h = F.rms_norm(x, (D,))
        q, k, v = self.qkv(h).view(B, T, 3, self.h, self.hd).unbind(2)
        q, k = F.rms_norm(q, (self.hd,)), F.rms_norm(k, (self.hd,))
        q, k = rotary(q, self.freq), rotary(k, self.freq)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2)) * 0.12
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (att.to(v.dtype) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(y)
        x = x + self.pr(F.relu(self.fc(F.rms_norm(x, (D,)))).square())
        return x


class GPT(nn.Module):
    def __init__(self, V=50304, L=6, d=512):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.emb.weight.data.normal_(0, 1)
        self.blocks = nn.ModuleList([Block(d) for _ in range(L)])
        self.head = nn.Parameter(torch.zeros(V, d))

    def forward(self, idx, tgt, reduction="mean"):
        ew = self.emb.weight.bfloat16() if MODE["m"] == "wr" else self.emb.weight
        x = F.rms_norm(F.embedding(idx, ew), (self.emb.weight.size(1),))
        for b in self.blocks:
            x = b(x)
        x = F.rms_norm(x, (x.size(-1),))
        hw = q_fp8(self.head + (self.head.detach().bfloat16().float() - self.head.detach())) if MODE["m"] == "wr" else self.head
        logits = F.linear(x, hw.to(x.dtype) if x.dtype != torch.float32 else hw).float()
        logits = 15 * logits * (logits.square() + 225).rsqrt()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), reduction=reduction)


def ctx():
    m = MODE["m"]
    torch.backends.cuda.matmul.allow_tf32 = (m == "tf32")
    torch.backends.cudnn.allow_tf32 = (m == "tf32")
    if m in ("bf16", "fp8w", "wr"):
        return torch.autocast(os.environ.get("DEV", "cuda"), dtype=torch.bfloat16)
    return torch.autocast(os.environ.get("DEV", "cuda"), enabled=False)


class Oracle:
    """Toy implementation of the Oracle interface in dynamics.py."""

    def __init__(self, model, probes: dict, mb: int):
        self.model, self.probes, self.mb = model, probes, mb
        self.params = [p for p in model.parameters()]
        self.names = [n for n, _ in model.named_parameters()]
        self.dist_group = None
        self.n_grad = 0

    def get_params(self):
        return [p.detach().float().clone() for p in self.params]

    @torch.no_grad()
    def set_params(self, vec):
        for p, v in zip(self.params, vec):
            p.copy_(v)

    def _chunks(self, probe):
        X, Y = self.probes[probe]
        for i in range(0, X.size(0), self.mb):
            yield X[i:i + self.mb], Y[i:i + self.mb]

    def loss_and_grad(self, probe):
        self.n_grad += 1
        for p in self.params:
            p.grad = None
        X, _ = self.probes[probe]
        n, tot = X.size(0), 0.0
        for x, y in self._chunks(probe):
            with ctx():
                l = self.model(x, y) * (x.size(0) / n)
            l.backward()
            tot += float(l)
        g = [p.grad.detach().float().clone() for p in self.params]
        for p in self.params:
            p.grad = None
        return tot, g

    @torch.no_grad()
    def loss(self, probe):
        X, _ = self.probes[probe]
        n, tot = X.size(0), 0.0
        for x, y in self._chunks(probe):
            with ctx():
                tot += float(self.model(x, y)) * x.size(0) / n
        return tot

    def hvp(self, probe, v):
        """Exact HVP: fp32 double backward (forces fp32 regardless of MODE)."""
        old = MODE["m"]
        MODE["m"] = "fp32"
        X, _ = self.probes[probe]
        n = X.size(0)
        out = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        try:
            for x, y in self._chunks(probe):
                with ctx():
                    l = self.model(x, y) * (x.size(0) / n)
                g = torch.autograd.grad(l, self.params, create_graph=True)
                gv = sum((gi * vi).sum() for gi, vi in zip(g, v))
                hv = torch.autograd.grad(gv, self.params)
                for o, h in zip(out, hv):
                    o += h
        finally:
            MODE["m"] = old
        return out


def rel_err(a, b):
    return vnorm(vsub(a, b)) / max(vnorm(b), 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/shawnghu/nanogpt-fb/data/fineweb10B/fineweb_train_000001.bin")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--train_steps", type=int, default=400)
    ap.add_argument("--probe_seqs", type=int, default=64)
    ap.add_argument("--mb", type=int, default=16)
    ap.add_argument("--out", default="out/validate_hvp.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = os.environ.get("DEV", "cuda")
    T = args.seq
    toks = load_tokens(args.data, 40_000_000) if os.path.exists(args.data) else torch.randint(0, 50304, (40_000_000,))
    model = GPT(L=args.layers, d=args.dim).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params {nparam/1e6:.1f}M", flush=True)

    # --- brief AdamW training (bf16 autocast) so we're not at init
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)
    B = 64
    MODE["m"] = "bf16"
    pos = 10_000_000
    t0 = time.time()
    for s in range(args.train_steps):
        buf = toks[pos:pos + B * T + 1].to(dev); pos += B * T
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        lr = 3e-3 * min(1, (s + 1) / 50) * (1 - s / args.train_steps * 0.9)
        for gp in opt.param_groups:
            gp["lr"] = lr
        with ctx():
            loss = model(x, y)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if s % 100 == 0 or s == args.train_steps - 1:
            print(f"train step {s} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    del opt

    # --- probe sets (fixed): probe A for everything; probe B (disjoint) for a cross-check
    def mk(start, n):
        buf = toks[start:start + n * T + 1].to(dev)
        return buf[:-1].view(n, T), buf[1:].view(n, T)
    probes = {"A": mk(0, args.probe_seqs), "B": mk(2_000_000, args.probe_seqs)}
    orc = Oracle(model, probes, args.mb)
    R = {"params_M": nparam / 1e6, "probe_tokens": args.probe_seqs * T, "config": vars(args)}

    # --- 1. nondeterminism floor + timing
    floor = {}
    for m in ["fp32", "tf32", "bf16", "fp8w"]:
        MODE["m"] = m
        orc.loss_and_grad("A")
        (torch.cuda.synchronize() if torch.cuda.is_available() else None); t = time.time()
        l1, g1 = orc.loss_and_grad("A")
        (torch.cuda.synchronize() if torch.cuda.is_available() else None); dt = time.time() - t
        l2, g2 = orc.loss_and_grad("A")
        floor[m] = {"loss": l1, "grad_rel_diff_repeat": rel_err(g1, g2), "grad_seconds": dt}
        if m == "fp32":
            g_ref = g1
        else:
            floor[m]["grad_rel_err_vs_fp32"] = rel_err(g1, g_ref)
        print(m, floor[m], flush=True)
    R["floor"] = floor
    MODE["m"] = "fp32"
    (torch.cuda.synchronize() if torch.cuda.is_available() else None); t = time.time()
    orc.hvp("A", vrandn_like(g_ref))
    (torch.cuda.synchronize() if torch.cuda.is_available() else None)
    R["exact_hvp_seconds"] = time.time() - t
    print("exact hvp s", R["exact_hvp_seconds"], flush=True)

    # --- 3a. exact Lanczos convergence (also gives the top eigvec for test 2)
    MODE["m"] = "fp32"
    ex = HVP(orc, "A", mode="exact")
    gen = torch.Generator(device=dev).manual_seed(1)
    v0 = vrandn_like(g_ref, gen)
    lz = {}
    ref = lanczos(ex, v0, 80, return_top=3)
    ref_top = ref.ritz_values[::-1][:10]
    lz["k80_top10"] = ref_top.tolist()
    lz["k80_bottom5"] = ref.ritz_values[:5].tolist()
    for k in [5, 10, 15, 20, 30, 40, 60]:
        r = lanczos(ex, v0, k)
        top = r.ritz_values[::-1][:5]
        lz[f"k{k}_top5"] = top.tolist()
        lz[f"k{k}_top5_relerr"] = (np.abs(top - ref_top[:5]) / np.abs(ref_top[:5])).tolist()
    # different start vector, same k, measures start-vector sensitivity
    r2 = lanczos(ex, vrandn_like(g_ref, torch.Generator(device=dev).manual_seed(7)), 30)
    lz["k30_seed7_top5"] = r2.ritz_values[::-1][:5].tolist()
    # probe B top eigenvalue (subset sensitivity)
    exB = HVP(orc, "B", mode="exact")
    rB = lanczos(exB, v0, 30)
    lz["probeB_k30_top5"] = rB.ritz_values[::-1][:5].tolist()
    R["lanczos_exact"] = lz
    print("lanczos", json.dumps(lz)[:800], flush=True)
    top_vec = ref.ritz_vectors[0]
    # gradient alignment with top eigenspace
    R["grad_frac_in_top3"] = subspace_fraction(g_ref, ref.ritz_vectors).tolist()

    # --- 2. FD accuracy vs eps per mode and direction
    dirs = {"random": vscale(v0, 1 / vnorm(v0)), "top_eig": top_vec, "grad": vscale(g_ref, 1 / vnorm(g_ref))}
    exact_hv = {k: orc.hvp("A", v) for k, v in dirs.items()}
    rq_exact = {k: vdot(exact_hv[k], dirs[k]) / vdot(dirs[k], dirs[k]) for k in dirs}
    R["rayleigh_exact"] = rq_exact
    fd = {}
    for m in ["fp32", "tf32", "bf16", "fp8w"]:
        MODE["m"] = m
        for eps in [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
            h = HVP(orc, "A", mode="fd", eps_rel=eps)
            for k, v in dirs.items():
                hv = h(v)
                rq = vdot(hv, v) / vdot(v, v)
                fd[f"{m}|{eps:g}|{k}"] = {"rel_err": rel_err(hv, exact_hv[k]),
                                          "rq_rel_err": abs(rq - rq_exact[k]) / abs(rq_exact[k])}
        print(m, {kk: (round(vv["rel_err"], 4), round(vv["rq_rel_err"], 4)) for kk, vv in fd.items() if kk.startswith(m)}, flush=True)
    R["fd"] = fd

    # --- 3b. FD Lanczos in the best cheap mode vs exact
    fdl = {}
    for m, eps in [("tf32", 1e-3), ("bf16", 1e-2), ("fp32", 1e-3)]:
        MODE["m"] = m
        h = HVP(orc, "A", mode="fd", eps_rel=eps)
        r = lanczos(h, v0, 30)
        fdl[f"{m}|{eps:g}"] = {"top5": r.ritz_values[::-1][:5].tolist(),
                               "top5_relerr_vs_exact_k80": (np.abs(r.ritz_values[::-1][:5] - ref_top[:5]) / np.abs(ref_top[:5])).tolist()}
        print("fd lanczos", m, fdl[f"{m}|{eps:g}"], flush=True)
    R["lanczos_fd"] = fdl

    # --- SLQ sanity (exact HVP, small)
    MODE["m"] = "fp32"
    t = time.time()
    s = slq_density(ex, g_ref, n_vectors=3, k=40)
    R["slq"] = {"seconds": time.time() - t, "nodes_min": float(s["slq_nodes"].min()), "nodes_max": float(s["slq_nodes"].max()),
                "neg_mass": float((s["slq_weights"] * (s["slq_nodes"] < 0)).sum() / 3)}
    np.savez(args.out.replace(".json", "_slq.npz"), **s)

    # --- line probe along a gradient step, sanity of the quadratic model
    MODE["m"] = "tf32"
    th0 = orc.get_params()
    u = vscale(g_ref, -1.0 / vnorm(g_ref) * vnorm(th0) * 0.005)  # step of 0.5% of ||theta|| along -g
    lpr = line_probe(orc, "A", th0, u, g0=g_ref)
    uHu = vdot(orc.hvp("A", u), u)
    R["line_probe"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in lpr.items()}
    R["line_probe"]["uHu_exact"] = uHu

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(R, f, indent=1)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
