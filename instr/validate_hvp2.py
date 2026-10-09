"""v2: FD-HVP step-size rules x precision modes, and FD-Lanczos vs exact Lanczos.

Reuses the toy GPT / Oracle of validate_hvp.py. Trains (or loads) a model, then:
  A. exact Lanczos (k=60) -> reference top eigenvalues and top-3 Ritz vectors
  B. FD-HVP error grid: mode x eps_rule x eps x richardson x direction (random / top_eig / grad /
     mix = (random + top_eig)/sqrt2)
  C. FD-Lanczos (k=30) per mode for the best configs, vs exact; plus probe-size sensitivity of
     lambda_max (exact, probe of 1/4 size) and lambda_max on a disjoint probe
Modes: fp32, bf16 (autocast; weights rounded to bf16 at use), fp8w (block matrices e4m3 per-tensor),
wr (WR-like: bf16 storage of matrices + e4m3 per-tensor of block matrices AND lm_head, bf16 embedding).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_hvp as V  # noqa
from dynamics import HVP, lanczos, vdot, vnorm, vrandn_like, vscale, vsub  # noqa


def rel_err(a, b):
    return vnorm(vsub(a, b)) / max(vnorm(b), 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/shawnghu/nanogpt-fb/data/fineweb10B/fineweb_train_000001.bin")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--train_steps", type=int, default=1500)
    ap.add_argument("--probe_seqs", type=int, default=64)
    ap.add_argument("--mb", type=int, default=16)
    ap.add_argument("--ckpt", default="out/toy_gpt.pt")
    ap.add_argument("--out", default="out/validate_hvp2.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev, T = "cuda", args.seq
    toks = V.load_tokens(args.data, 60_000_000)
    model = V.GPT(L=args.layers, d=args.dim).to(dev)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt))
        print("loaded", args.ckpt, flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)
        B, pos, t0 = 64, 10_000_000, time.time()
        V.MODE["m"] = "bf16"
        for s in range(args.train_steps):
            buf = toks[pos:pos + B * T + 1].to(dev); pos += B * T
            for gp in opt.param_groups:
                gp["lr"] = 3e-3 * min(1, (s + 1) / 100) * (1 - 0.9 * s / args.train_steps)
            with V.ctx():
                loss = model(buf[:-1].view(B, T), buf[1:].view(B, T))
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            if s % 250 == 0 or s == args.train_steps - 1:
                print(f"train {s} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        torch.save(model.state_dict(), args.ckpt)
        del opt

    def mk(start, n):
        buf = toks[start:start + n * T + 1].to(dev)
        return buf[:-1].view(n, T), buf[1:].view(n, T)
    probes = {"A": mk(0, args.probe_seqs), "B": mk(2_000_000, args.probe_seqs), "A4": mk(0, args.probe_seqs // 4)}
    orc = V.Oracle(model, probes, args.mb)
    R = {"config": vars(args)}
    V.MODE["m"] = "fp32"
    lossA, gA = orc.loss_and_grad("A")
    R["probe_loss"] = lossA
    R["tensor_norms"] = {n: float(p.norm()) for n, p in zip(orc.names, orc.params)}

    # A. exact reference
    ex = HVP(orc, "A", mode="exact")
    gen = torch.Generator(device=dev).manual_seed(1)
    v0 = vrandn_like(gA, gen)
    t = time.time()
    ref = lanczos(ex, v0, 60, return_top=3)
    ref_top = ref.ritz_values[::-1][:8]
    R["exact_k60_top8"] = ref_top.tolist()
    R["exact_k60_bottom3"] = ref.ritz_values[:3].tolist()
    R["exact_k60_seconds"] = time.time() - t
    top = ref.ritz_vectors[0]
    R["top_eig_tensor_mass"] = {n: float(x.pow(2).sum()) for n, x in zip(orc.names, top) if float(x.pow(2).sum()) > 0.01}
    print("exact top8", ref_top, "mass", R["top_eig_tensor_mass"], flush=True)
    for k in (10, 20, 30):
        r = lanczos(ex, v0, k)
        R[f"exact_k{k}_top5"] = r.ritz_values[::-1][:5].tolist()
    R["exact_probeA4_k30_top3"] = lanczos(HVP(orc, "A4", mode="exact"), v0, 30).ritz_values[::-1][:3].tolist()
    R["exact_probeB_k30_top3"] = lanczos(HVP(orc, "B", mode="exact"), v0, 30).ritz_values[::-1][:3].tolist()
    print("probe sens", R["exact_probeA4_k30_top3"], R["exact_probeB_k30_top3"], flush=True)

    rnd = vscale(v0, 1 / vnorm(v0))
    mix = vscale([a + b for a, b in zip(rnd, top)], 1 / vnorm([a + b for a, b in zip(rnd, top)]))
    dirs = {"random": rnd, "top_eig": top, "grad": vscale(gA, 1 / vnorm(gA)), "mix": mix}
    exact_hv = {k: orc.hvp("A", v) for k, v in dirs.items()}
    rq_exact = {k: vdot(exact_hv[k], dirs[k]) for k in dirs}
    R["rq_exact"] = rq_exact

    # B. grid
    grid = {}
    configs = [("maxrel", e, False) for e in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)] + \
              [("maxrel", e, True) for e in (1e-3, 3e-3, 1e-2)] + \
              [("global", e, False) for e in (1e-6, 1e-5)]
    for m in ["fp32", "bf16", "fp8w", "wr"]:
        V.MODE["m"] = m
        _, g1 = orc.loss_and_grad("A")
        _, g2 = orc.loss_and_grad("A")
        grid[f"{m}|repeat_grad_reldiff"] = rel_err(g1, g2)
        for rule, e, rich in configs:
            h = HVP(orc, "A", mode="fd", eps_rel=e, eps_rule=rule, richardson=rich)
            for k, v in dirs.items():
                hv = h(v)
                grid[f"{m}|{rule}|{e:g}|{'R' if rich else '-'}|{k}"] = [
                    round(rel_err(hv, exact_hv[k]), 5), round(abs(vdot(hv, v) - rq_exact[k]) / abs(rq_exact[k]), 5)]
        print(m, {k: v for k, v in grid.items() if k.startswith(m)}, flush=True)
    R["grid"] = grid

    # C. FD Lanczos per mode/config
    fdl = {}
    for m in ["fp32", "bf16", "fp8w", "wr"]:
        V.MODE["m"] = m
        for rule, e, rich in [("maxrel", 1e-3, False), ("maxrel", 3e-3, False), ("maxrel", 1e-2, False),
                              ("maxrel", 1e-2, True)]:
            h = HVP(orc, "A", mode="fd", eps_rel=e, eps_rule=rule, richardson=rich)
            t = time.time()
            r = lanczos(h, v0, 30)
            tp = r.ritz_values[::-1][:5]
            fdl[f"{m}|{rule}|{e:g}|{'R' if rich else '-'}"] = {
                "top5": tp.tolist(), "relerr": (np.abs(tp - ref_top[:5]) / np.abs(ref_top[:5])).round(4).tolist(),
                "bottom2": r.ritz_values[:2].tolist(), "seconds": time.time() - t}
            print("fdL", m, rule, e, rich, fdl[f"{m}|{rule}|{e:g}|{'R' if rich else '-'}"], flush=True)
    R["fd_lanczos"] = fdl
    with open(args.out, "w") as f:
        json.dump(R, f, indent=1)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
