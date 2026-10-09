"""Instrumentation tests on the REAL WR model through fb/'s FullBatchTrainer (n-gram off, any #GPUs).

Run from the fb/ directory (data globs are relative):
  python -m torch.distributed.run --standalone --nproc_per_node=N /path/instr/test_fb_oracle.py \
      [pre_steps=150] [suite_steps=4] [out=/path/instr/out/fb_oracle]  (+ any FBConfig key=value)

  1. brief minibatch training on a fixed set so we are off init (pre_steps)
  2. gradient nondeterminism floor of the WR kernels (same probe, same params, twice)
  3. FD-HVP self-consistency on the real model (no exact HVP exists through FA3/FP8):
     Rayleigh quotients over eps (plateau = trustworthy), symmetry v1.Hv2 vs v2.Hv1, Richardson
  4. FD-Lanczos top eigenvalues vs eps / Richardson / start vector
  5. DynamicsSuite end-to-end for a few full-batch steps with every probe enabled (timings, files)
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))  # before importing the model (fp8/CE kernels)

FB = "/workspace/shawnghu/nanogpt-fb/fb"
sys.path.insert(0, FB)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch._dynamo as dynamo  # noqa
dynamo.config.recompile_limit = 64
from fullbatch.config import parse_config  # noqa
from fullbatch.trainer import FullBatchTrainer  # noqa
from dynamics import HVP, lanczos, vdot, vnorm, vrandn_like, vscale, vsub  # noqa
from fb_hooks import Cadence, DynamicsSuite, FBOracle, pick_probe_ids  # noqa


def main():
    extra = {}
    rest = []
    for a in sys.argv[1:]:
        k, _, v = a.partition("=")
        if k in ("pre_steps", "suite_steps", "out", "probe_tokens"):
            extra[k] = v
        else:
            rest.append(a)
    pre = int(extra.get("pre_steps", 150))
    suite_steps = int(extra.get("suite_steps", 4))
    out = extra.get("out", "/workspace/shawnghu/nanogpt-fb/instr/out/fb_oracle")
    probe_tokens = int(float(extra.get("probe_tokens", 393216)))
    cfg = parse_config(["mode=minibatch", "minibatch_source=fixed", "dataset_tokens=25165824",
                        "batch_tokens=393216", f"steps={pre}", "val_every=0", "canon_mask=False"] + rest)
    t = FullBatchTrainer(cfg)
    R = {"world": t.world, "cfg": cfg.__dict__}
    say = (lambda *a: print(*a, flush=True)) if t.master else (lambda *a: None)
    os.makedirs(out, exist_ok=True)

    # 1. get off init
    t0 = time.time()
    for s in range(pre):
        rec = t.train_step(s)
        if s % 25 == 0 or s == pre - 1:
            say(f"pre {s} loss {rec['train_loss']:.4f} tok/s/gpu {rec['tok_per_s_per_gpu']:.0f} {time.time()-t0:.0f}s")
    R["pre_final_loss"] = rec["train_loss"]
    R["val_after_pre"] = t.evaluate()
    say("val", R["val_after_pre"])

    probe = pick_probe_ids(t, probe_tokens, seed=0)
    o = FBOracle(t, {"probe": probe, "probe2": pick_probe_ids(t, probe_tokens, seed=1, exclude=set(probe))})
    R["probe_tokens"] = len(probe) * cfg.micro_tokens * t.world
    R["names"] = o.names
    R["numel_M"] = sum(p.numel() for p in o.params) / 1e6
    theta = o.get_params()
    R["tensor_norms"] = {n: float(x.norm()) for n, x in zip(o.names, theta)}

    # 2. nondeterminism + timing
    torch.cuda.synchronize(); ta = time.time()
    l1, g1 = o.loss_and_grad("probe")
    torch.cuda.synchronize(); tb = time.time()
    l2, g2 = o.loss_and_grad("probe")
    R["probe_grad_seconds"] = tb - ta
    R["grad_repeat_reldiff"] = vnorm(vsub(g1, g2)) / vnorm(g1)
    R["grad_repeat_reldiff_per_tensor"] = {n: float((a - b).norm() / (a.norm() + 1e-30)) for n, a, b in zip(o.names, g1, g2)}
    R["loss_repeat_diff"] = l1 - l2
    torch.cuda.synchronize(); ta = time.time()
    lf = o.loss("probe")
    torch.cuda.synchronize()
    R["probe_fwd_seconds"] = time.time() - ta
    # set_params round trip is exact?
    o.set_params(theta)
    l3, g3 = o.loss_and_grad("probe")
    R["roundtrip_grad_reldiff"] = vnorm(vsub(g1, g3)) / vnorm(g1)
    _, gq = o.loss_and_grad("probe2")
    R["probe_vs_probe2_grad_cos"] = vdot(g1, gq) / (vnorm(g1) * vnorm(gq))
    say("floor", {k: R[k] for k in ("probe_grad_seconds", "grad_repeat_reldiff", "loss_repeat_diff",
                                    "probe_fwd_seconds", "roundtrip_grad_reldiff", "probe_vs_probe2_grad_cos")})

    # 3. FD self-consistency
    gen = torch.Generator(device=t.device).manual_seed(3)
    v1 = vscale(vrandn_like(g1, gen), 1.0)
    v1 = vscale(v1, 1 / vnorm(v1))
    v2 = vrandn_like(g1, gen); v2 = vscale(v2, 1 / vnorm(v2))
    gd = vscale(g1, 1 / vnorm(g1))
    cons = {}
    for rule, e, rich in [("maxrel", 3e-4, False), ("maxrel", 1e-3, False), ("maxrel", 3e-3, False),
                          ("maxrel", 1e-2, False), ("maxrel", 3e-2, False), ("maxrel", 1e-2, True),
                          ("maxrel", 3e-3, True)]:
        h = HVP(o, "probe", eps_rel=e, eps_rule=rule, richardson=rich)
        hv1, hv2, hg = h(v1), h(v2), h(gd)
        key = f"{rule}|{e:g}|{'R' if rich else '-'}"
        cons[key] = {"rq_rand": vdot(hv1, v1), "rq_rand2": vdot(hv2, v2), "rq_grad": vdot(hg, gd),
                     "sym_v2Hv1": vdot(v2, hv1), "sym_v1Hv2": vdot(v1, hv2),
                     "sym_gHv1": vdot(gd, hv1), "sym_v1Hg": vdot(v1, hg),
                     "eps": h._eps(v1)}
        say("cons", key, {k: round(v, 6) for k, v in cons[key].items()})
    R["fd_consistency"] = cons

    # 4. FD Lanczos
    fdl = {}
    v0 = vrandn_like(g1, torch.Generator(device=t.device).manual_seed(11))
    v0b = vrandn_like(g1, torch.Generator(device=t.device).manual_seed(12))
    for rule, e, rich, start, k in [("maxrel", 1e-3, False, "a", 20), ("maxrel", 3e-3, False, "a", 20),
                                    ("maxrel", 1e-2, False, "a", 20), ("maxrel", 3e-3, False, "b", 20),
                                    ("maxrel", 3e-3, True, "a", 20), ("maxrel", 3e-3, False, "a", 40)]:
        h = HVP(o, "probe", eps_rel=e, eps_rule=rule, richardson=rich)
        torch.cuda.synchronize(); ta = time.time()
        r = lanczos(h, v0 if start == "a" else v0b, k, return_top=1)
        key = f"{rule}|{e:g}|{'R' if rich else '-'}|{start}|k{k}"
        top = r.ritz_values[::-1][:5]
        # re-evaluate the top Ritz vector's Rayleigh quotient with an independent eps (validation)
        hc = HVP(o, "probe", eps_rel=1e-2 if e != 1e-2 else 3e-3, eps_rule="maxrel")
        rq = vdot(hc(r.ritz_vectors[0]), r.ritz_vectors[0])
        fdl[key] = {"top5": top.tolist(), "bottom2": r.ritz_values[:2].tolist(), "seconds": time.time() - ta,
                    "rq_top_vec_crosscheck": rq,
                    "top_vec_mass": {n: float(x.pow(2).sum()) for n, x in zip(o.names, r.ritz_vectors[0]) if float(x.pow(2).sum()) > 0.02}}
        say("fdL", key, fdl[key])
    R["fd_lanczos"] = fdl
    o.set_params(theta)

    # 5. suite end-to-end in full-batch mode on a small set (every probe on, small k)
    t.cfg.mode = "full"
    cad = Cadence(val=1, norms=1, microbatch_stats=1, spectra=1, line_probe=1, curvature=1, lanczos=2,
                  lanczos_k=10, slq=3, slq_vectors=2, slq_k=10, raw_grad=1, raw_update=1, checkpoint=2,
                  hvp_eps_rel=3e-3)
    suite = DynamicsSuite(t, out + "/suite", cad, probe_tokens=probe_tokens, hvp_probe_tokens=probe_tokens // 2)
    recs = []
    for s in range(suite_steps):
        rec = suite.train_step(pre + s)
        recs.append({k: v for k, v in rec.items() if isinstance(v, (int, float))})
        say("suite", json.dumps(recs[-1])[:3000])
    R["suite"] = recs
    if t.master:
        with open(out + "/results.json", "w") as f:
            json.dump(R, f, indent=1, default=str)
        print("WROTE", out + "/results.json", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
