"""v2 tests on the real WR model via FullBatchTrainer (n-gram off), with frozen fp8 state during probes.

python -m torch.distributed.run --standalone --nproc_per_node=N test_fb_oracle2.py [pre_steps=150] [out=...]
"""
import hashlib
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
from dynamics import HVP, ScaledHVP, lanczos, vdot, vnorm, vrandn_like, vscale, vsub  # noqa
from fb_hooks import Cadence, DynamicsSuite, FBOracle, pick_probe_ids  # noqa


def state_hash(t):
    h = hashlib.sha256()
    for n, p in t.raw_model.named_parameters():
        h.update(p.detach().float().cpu().numpy().tobytes())
    for n, b in t.raw_model.named_buffers():
        if n.split(".")[-1] in ("ngram_cache", "canon_mask"):
            continue
        h.update(b.detach().view(torch.uint8).cpu().numpy().tobytes() if b.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                 else b.detach().float().cpu().numpy().tobytes())
    for p, st in t.optimizer.param_states.items():
        for k, v in st.items():
            if torch.is_tensor(v):
                h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    extra, rest = {}, []
    for a in sys.argv[1:]:
        k, _, v = a.partition("=")
        (extra.__setitem__(k, v) if k in ("pre_steps", "out", "suite_steps", "only") else rest.append(a))
    pre = int(extra.get("pre_steps", 150))
    out = extra.get("out", "/workspace/shawnghu/nanogpt-fb/instr/out/fb_oracle2")
    cfg = parse_config(["mode=minibatch", "minibatch_source=fixed", "dataset_tokens=100663296",
                        "batch_tokens=393216", f"steps={pre}", "canon_mask=False"] + rest)
    t = FullBatchTrainer(cfg)
    say = (lambda *a: print(*a, flush=True)) if t.master else (lambda *a: None)
    R = {"world": t.world}
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for s in range(pre):
        rec = t.train_step(s)
        if s % 50 == 0 or s == pre - 1:
            say(f"pre {s} loss {rec['train_loss']:.4f} {time.time()-t0:.0f}s")
    R["val_after_pre"] = t.evaluate()
    say("val", R["val_after_pre"])

    MB = cfg.micro_tokens * t.world
    probes = {f"p{n}": pick_probe_ids(t, n * MB, seed=0) for n in (4, 8, 16, 32)}
    probes["q8"] = pick_probe_ids(t, 8 * MB, seed=5, exclude=set(probes["p32"]))
    o = FBOracle(t, probes)
    theta = o.get_params()

    # A. roundtrip / frozen-state determinism
    l0, g0 = o.loss_and_grad("p8")              # live state, no freeze
    h_before = state_hash(t)
    o.begin_probe()
    o.set_params(theta)
    l1, g1 = o.loss_and_grad("p8")
    o.set_params(theta)
    l2, g2 = o.loss_and_grad("p8")
    o.end_probe(theta)
    h_after = state_hash(t)
    R["A"] = {"frozen_repeat_grad_reldiff": vnorm(vsub(g1, g2)) / vnorm(g1),
              "frozen_vs_live_grad_reldiff": vnorm(vsub(g0, g1)) / vnorm(g0),
              "state_hash_before": h_before, "state_hash_after": h_after, "state_restored": h_before == h_after}
    say("A", R["A"])

    only = extra.get("only", "ACDEF")
    o.begin_probe()
    o.set_params(theta)
    gen = torch.Generator(device=t.device).manual_seed(3)
    v1 = vrandn_like(g1, gen); v1 = vscale(v1, 1 / vnorm(v1))
    gd = vscale(g1, 1 / vnorm(g1))
    C = {}
    for e in ((1e-4, 3e-4, 1e-3, 3e-3, 1e-2) if "C" in only else ()):
        h = HVP(o, "p8", eps_rel=e)
        a, b = h(gd), h(gd)
        hv1 = h(v1)
        C[f"{e:g}"] = {"rq_grad": vdot(a, gd), "rq_grad_repeat_reldiff": vnorm(vsub(a, b)) / vnorm(a),
                       "rq_rand": vdot(hv1, v1), "sym_gHv1": vdot(gd, hv1), "sym_v1Hg": vdot(v1, a)}
        say("C", e, C[f"{e:g}"])
    R["C"] = C

    # D. Lanczos: eps, start vector, probe size
    D = {}
    v0 = vrandn_like(g1, torch.Generator(device=t.device).manual_seed(11))
    v0b = vrandn_like(g1, torch.Generator(device=t.device).manual_seed(12))
    runs = [("p8", 3e-4, "a"), ("p8", 1e-3, "a"), ("p8", 3e-3, "a"), ("p8", 1e-3, "b"),
            ("p4", 1e-3, "a"), ("p16", 1e-3, "a"), ("p32", 1e-3, "a"), ("q8", 1e-3, "a")]
    for probe, e, st in (runs if "D" in only else []):
        h = HVP(o, probe, eps_rel=e)
        ta = time.time()
        r = lanczos(h, v0 if st == "a" else v0b, 20, return_top=1)
        key = f"{probe}|{e:g}|{st}"
        D[key] = {"top5": r.ritz_values[::-1][:5].tolist(), "bottom2": r.ritz_values[:2].tolist(),
                  "seconds": time.time() - ta, "probe_tokens": len(probes[probe]) * MB,
                  "top_vec_mass": {n: round(float(x.pow(2).sum()), 3) for n, x in zip(o.names, r.ritz_vectors[0])
                                   if float(x.pow(2).sum()) > 0.02}}
        say("D", key, D[key])
    R["D"] = D

    # E. Adam-preconditioned sharpness
    s, beta1 = o.adam_precond_sqrt()
    h = HVP(o, "p8", eps_rel=1e-3)
    rp = lanczos(ScaledHVP(h, s), [x * (si > 0) for x, si in zip(v0, s)], 15 if "E" in only else 2, return_top=1)
    R["E"] = {"precond_top5": rp.ritz_values[::-1][:5].tolist(),
              "beta1": dict(zip(o.names, beta1)),
              "top_vec_mass": {n: round(float(x.pow(2).sum()), 3) for n, x in zip(o.names, rp.ritz_vectors[0])
                               if float(x.pow(2).sum()) > 0.02}}
    say("E", R["E"])
    o.end_probe(theta)
    R["state_restored_after_BCDE"] = state_hash(t) == h_before
    say("restored after BCDE:", R["state_restored_after_BCDE"])

    # F. suite with 64-microbatch steps, all probes on
    t.K = 64
    t.cfg.steps = pre + 10
    cad = Cadence(val=1, norms=1, microbatch_stats=1, spectra=1, line_probe=1, curvature=1, lanczos=1,
                  lanczos_k=20, precond_lanczos=1, slq=1, slq_vectors=2, slq_k=30, raw_grad=1, raw_update=1,
                  checkpoint=1)
    suite = DynamicsSuite(t, out + "/suite", cad, probe_tokens=16 * MB, hvp_probe_tokens=8 * MB)
    recs = []
    for i in range(int(extra.get("suite_steps", 3))):
        rec = suite.train_step(pre + i)
        recs.append({k: v for k, v in rec.items() if isinstance(v, (int, float))})
        say("F", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in recs[-1].items()
                             if not k.startswith("spec/")}))
    suite.close()
    R["F"] = recs
    if t.master:
        with open(out + "/results.json", "w") as f:
            json.dump(R, f, indent=1, default=str)
        print("WROTE", out + "/results.json", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
