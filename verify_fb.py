"""Gradient-correctness checks for the full-batch trainer.

  torchrun --nproc_per_node=N verify_fb.py key=value ...  (FBConfig keys; plus pre_steps via env PRE_STEPS)

1. determinism: two accumulations over the same dataset at the same parameters
2. order invariance: reversed microbatch order
3. concat: K microbatches accumulated vs ONE microbatch that concatenates them (same tokens; attention
   segments unchanged; differs only through the smear / n-gram look-back at K-1 boundary BOS tokens)
4. fp8 vs bf16: the training (fp8) gradient vs the eval-path (bf16, no fp8) gradient on the same tokens
5. world-size invariance: mean gradient saved to <out_dir>/grad_w{world}.pt; compared with any other
   world size's file already there (same seed, same dataset)
"""
import json
import os
import sys

import torch
# The CE kernel (perf/kernels/cross_entropy.py) is compiled into the CURRENT device at import time.
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fullbatch.config import parse_config
from fullbatch.trainer import FullBatchTrainer
from track_1_short.data import cu_seqlens_rows


def compare(a: dict, b: dict):
    out = {}
    tot_d = tot_n = 0.0
    for k in a:
        if k not in b:
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:  # banks are padded to a multiple of world size: compare the common rows
            n0 = min(x.shape[0], y.shape[0])
            x, y = x[:n0], y[:n0]
        x, y = x.double().flatten(), y.double().flatten()
        d, n = (x - y).norm().item(), y.norm().item()
        tot_d += d * d
        tot_n += n * n
        if n > 0:
            out[k] = dict(rel=d / n, cos=(torch.dot(x, y) / (x.norm() * y.norm() + 1e-30)).item())
    out["_total_rel"] = (tot_d / max(tot_n, 1e-30)) ** 0.5
    return out


def fd_check(t, cfg, gk, mbs, res):
    named = t.named
    flat_g = []
    for label, p in named:
        if label == "embed" and cfg.tied_embed:
            flat_g.append(torch.zeros(p.numel(), device=t.device))
        else:
            flat_g.append(gk[label].reshape(-1).float())
    g = torch.cat(flat_g)
    gnorm = g.norm().item()
    d = g / gnorm
    theta0 = t.get_flat()
    fd = dict(gnorm=gnorm, theta_norm=theta0.norm().item())
    torch.manual_seed(1234)
    dr = torch.randn_like(g)
    dr = dr * (g != 0)  # only coordinates that have gradient
    dr /= dr.norm()
    fd["g_dot_drand"] = torch.dot(g, dr).item()
    for eps in (0.25, 0.5, 1.0, 2.0):
        for name, dirv in (("g", d), ("rand", dr)):
            vals = {}
            for sgn in (1, -1):
                t.set_flat(theta0 + sgn * eps * dirv)
                vals[sgn] = (t.eval_loss_on(mbs), t.loss_on(mbs))
            t.set_flat(theta0)
            fd[f"eps{eps}_{name}_slope_bf16"] = (vals[1][0] - vals[-1][0]) / (2 * eps)
            fd[f"eps{eps}_{name}_slope_fp8"] = (vals[1][1] - vals[-1][1]) / (2 * eps)
    for eps in (0.25, 0.5, 1.0, 2.0):
        s_ = fd[f"eps{eps}_g_slope_bf16"]
        fd[f"eps{eps}_ratio_bf16"] = s_ / gnorm
        fd[f"eps{eps}_r_est_bf16"] = (max(gnorm / s_ - 1, 0)) ** 0.5 if s_ > 0 else None
    res["fd_check"] = fd


def main():
    cfg = parse_config(sys.argv[1:])
    pre_steps = int(os.environ.get("PRE_STEPS", "10"))
    K = int(os.environ.get("CONCAT_K", "4"))
    t = FullBatchTrainer(cfg)
    res = dict(world=t.world, dataset_tokens=t.dataset_tokens, checksum=t.dataset_checksum, pre_steps=pre_steps)
    t.accumulate()
    g_init = t.mean_grads()
    # move away from init (zero-initialized projections make several grads trivially zero)
    for s in range(pre_steps):
        r = t.train_step(s)
    res["loss_after_pre"] = t.accumulate()[0]

    # 1 + 2
    g1 = t.mean_grads()
    t.accumulate()
    g2 = t.mean_grads()
    res["determinism_total_rel"] = compare(g2, g1)["_total_rel"]
    mbs = t._step_microbatches(0)[::-1]
    t.accumulate(mbs=mbs)
    g3 = t.mean_grads()
    res["reverse_order_total_rel"] = compare(g3, g1)["_total_rel"]

    # 3a: the same K microbatches through ONE autograd graph (sum of their losses, single backward):
    # checks the fp32 accumulation, value_embeds buffer flush, scaling and embed tying against autograd.
    mbs = t.fixed_mbs(list(range(K)))
    t.accumulate(mbs=mbs)
    gk = t.mean_grads()
    if t.ngram is None:
        params = [p for _, p in t.named if p is not t.ve]
        bufs = [torch.zeros_like(t.ve_buf) for _ in mbs]  # one per graph (autograd version checks)
        total = 0
        for (key, inp, tgt, cu), buf in zip(mbs, bufs):
            total = total + t.model(inp, tgt, cu, t._slots_off(inp.numel()), t.fwd_cfg, value_embed_grad=buf).sum()
        grads = torch.autograd.grad(total, params, allow_unused=True)
        gmap = {id(p): g for p, g in zip(params, grads)}
        n = torch.tensor([float(K * cfg.micro_tokens * t.world)], device=t.device)
        gj = {}
        for (label, p) in t.named:
            g = sum(b.float() for b in bufs) if p is t.ve else gmap[id(p)]
            g = torch.zeros_like(p, dtype=torch.float32) if g is None else g.float().clone()
            dist.all_reduce(g)
            gj[label] = g / n.item()
        del bufs
        if cfg.tied_embed:
            gj["lm_head"] += gj.pop("embed").T
        res["joint_backward_vs_accum"] = compare(gk, gj)

    # 3: concat K microbatches (on every rank its first K) into one forward
    T = cfg.micro_tokens
    inputs = torch.cat([m[1] for m in mbs])
    targets = torch.cat([m[2] for m in mbs])
    cus = []
    for j, m in enumerate(mbs):
        cu = m[3]
        segs = torch.cat([cu[1:][cu[1:] < T], torch.tensor([T], device=cu.device, dtype=cu.dtype)])
        cus.append(segs + j * T)
    seg_all = torch.cat(cus)
    R = cu_seqlens_rows(K * T)
    cu_big = torch.full((R,), K * T, dtype=torch.int32, device=inputs.device)
    cu_big[0] = 0
    cu_big[1:len(seg_all) + 1] = seg_all
    t.accumulate(mbs=[(None, inputs, targets, cu_big)])
    gc = t.mean_grads()
    res["concat_vs_accum"] = compare(gc, gk)

    # 4: finite-difference check of the fp8 gradient g on these K microbatches, against the bf16 eval-path
    # loss and the fp8 training loss. Along d = g/|g|, the central-difference slope s = grad(L).d; if
    # g = g_true + n with n orthogonal to g_true, s/|g| = 1/(1 + |n|^2/|g_true|^2), so r = sqrt(|g|/s - 1)
    # estimates the relative error of the fp8 gradient. Random directions give s_rand vs g.d_rand too.
    if os.environ.get("FD", "0") == "1":
        fd_check(t, cfg, gk, mbs, res)


    # 5: world-size invariance (full dataset gradient g1)
    if t.master:
        os.makedirs(cfg.out_dir, exist_ok=True)
        for tag, gg in (("grad", g1), ("gradinit", g_init)):
            torch.save({k: v.cpu() for k, v in gg.items()}, os.path.join(cfg.out_dir, f"{tag}_w{t.world}.pt"))
            for f in os.listdir(cfg.out_dir):
                if f.startswith(tag + "_w") and f != f"{tag}_w{t.world}.pt":
                    other = torch.load(os.path.join(cfg.out_dir, f))
                    res[f"{tag}_vs_{f}"] = compare({k: v.cpu() for k, v in gg.items()}, other)["_total_rel"]
        print("VERIFY " + json.dumps(res, indent=1), flush=True)
        with open(os.path.join(cfg.out_dir, f"verify_w{t.world}.json"), "w") as f:
            json.dump(res, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
