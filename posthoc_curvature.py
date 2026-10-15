"""Post-hoc curvature analysis of the full-batch run at its saved checkpoints.

For each checkpoint step s (weights theta_s, optimizer state before step s), this:
  1. loads the checkpoint from the Hugging Face export format (model.safetensors + resume_state/),
  2. recomputes the exact full-batch gradient and takes optimizer step s (the run's own step), and checks
     the full-dataset training loss against the run's log,
  3. measures, at theta_s along the actual update u_s:
     - raw Hessian extreme eigenvalues (Lanczos) and per-block top eigenvalues (ANVIL matrices / embeddings /
       everything else), the Adam-preconditioned top eigenvalue (Cohen et al. 2022), the line probe along u
       (all from instr/fb_hooks.py, i.e. the same code as the run's own instrumentation, every probe due);
     - a line probe along each block's part of the update separately (which block's step sits at the edge);
     - an ANVIL/Muon-preconditioned sharpness: lambda_max of S H S over the ANVIL matrices, where S^2 is the
       optimizer's local linear response to the gradient (defined below), and the stability threshold of
       ANVIL's momentum recurrence for that quantity.

Muon-preconditioned sharpness (our definition; there is no established one we know of). ANVIL's update for a
matrix is  u = -eff_lr * E(F(X)),  X = lerp(g, blend(fast rail, slow rail), mu)  (Nesterov input), F the
whitening cascade (a fixed odd polynomial map applied to the singular values of X / (1.05 ||X||_F)), E the
lane equalizer. Linearizing around the current X, a gradient perturbation along X's singular structure is
scaled by the secant gain f(sigma)/sigma of the cascade, times eff_lr and the equalizer. We take
P^-1 = c_j * Q diag(f(sigma)/sigma) Q^T on the short side of each matrix j, with the scalar c_j fitted so that
P^-1 X best reproduces the actual update u (reported: cosine between them), and S = P^-1/2. The momentum
recurrence (two EMA rails, blend weight, Nesterov lookahead) is then linear in a scalar quadratic mode with
preconditioned curvature a, and we report the largest a for which it is stable. This ignores the off-axis
(rotation) part of the polar map's derivative and weight decay, so treat it as a first-order estimate.

Usage (4 GPUs, the run's world size): torchrun --nproc_per_node=4 posthoc_curvature.py <export_dir> <out_dir> <step> [...]
"""
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
import numpy as np
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("NFB_INSTR_DIR", os.path.join(HERE, "instr")))
from safetensors.torch import load_file  # noqa: E402

from fullbatch.config import parse_config  # noqa: E402
from fullbatch.trainer import FullBatchTrainer  # noqa: E402
from track_1_short.optim.anvil import ANVIL_MAPS, FROBENIUS_MARGIN, RAIL_FAST_BETA, RAIL_FAST_WEIGHT, RAIL_SLOW_BETA  # noqa: E402
from fb_hooks import Cadence, DynamicsSuite  # noqa: E402
from dynamics import HVP, lanczos, line_probe, vrandn_like  # noqa: E402


def load_hf_checkpoint(t, ckdir):
    """Restore a checkpoint exported to the Hugging Face layout into a FullBatchTrainer (same world size)."""
    rs = os.path.join(ckdir, "resume_state")
    meta = json.load(open(os.path.join(rs, f"meta_rank{t.rank}.json")))
    assert meta["world"] == t.world, "resume_state is sharded for the run's world size (4)"
    opt = {}
    for k, v in load_file(os.path.join(rs, f"optimizer_rank{t.rank}.safetensors")).items():
        label, key = k.split("/", 1)
        opt.setdefault(label, {})[key] = v
    for k, v in meta["non_tensor_optimizer_state"].items():
        label, key = k.split("/", 1)
        opt.setdefault(label, {})[key] = v
    rng = load_file(os.path.join(rs, f"rng_rank{t.rank}.safetensors"))
    t.restore_state(dict(
        params=load_file(os.path.join(ckdir, "model.safetensors")),
        buffers=load_file(os.path.join(rs, f"buffers_rank{t.rank}.safetensors")),
        mlp_quantize_calls=meta["mlp_quantize_calls"], optimizer=opt, split_embed=meta["split_embed"],
        step_count=meta["step_count"], data={}, rng_cpu=rng["rng_cpu"], rng_cuda=rng["rng_cuda"],
        world=meta["world"], rank=meta["rank"]))
    return meta["step_count"]


def cascade_map(s):
    for a, b, c in ANVIL_MAPS:
        s = a * s + b * s ** 3 + c * s ** 5
    return s


def anvil_threshold(beta_f, beta_s, w, mu):
    """Largest preconditioned curvature a for which ANVIL's linear momentum recurrence is stable."""
    def rho(a):
        # state (x, vf, vs); h = x is the (preconditioned) gradient of the quadratic mode
        M = np.zeros((3, 3))
        # vf' = bf vf + (1-bf) x ; vs' = bs vs + (1-bs) x
        vf = np.array([1 - beta_f, beta_f, 0.0])
        vs = np.array([1 - beta_s, 0.0, beta_s])
        blend = w * vf + (1 - w) * vs
        X = (1 - mu) * np.array([1.0, 0, 0]) + mu * blend
        M[0] = np.array([1.0, 0, 0]) - a * X
        M[1], M[2] = vf, vs
        return max(abs(np.linalg.eigvals(M)))
    grid = np.linspace(1e-3, 200, 40001)
    stable = [a for a in grid if rho(a) <= 1 + 1e-9]
    return float(max(stable)) if stable else float("nan")


@torch.no_grad()
def anvil_preconditioner(t, o, G, u, step, loss_scale):
    """Per-ANVIL-tensor list of (S matrices [n_mat, k, k], tall flag) or None; plus calibration stats."""
    opt = t.optimizer
    rb = t.rail_beta(step)
    engaged = step >= round(t.cfg.rail_engage_frac * t.cfg.steps)
    beta_f, w = (RAIL_FAST_BETA, RAIL_FAST_WEIGHT) if engaged else (rb, 1.0)
    S, cal = [], {}
    for i, p in enumerate(o.params):
        cfg = opt.param_cfgs[p]
        if cfg.optim != "anvil":
            S.append(None)
            continue
        vel = opt.param_states[p]["velocity"]                      # [2, chunk, r, c], post-step rails
        parts = [torch.empty_like(vel) for _ in range(t.world)]
        dist.all_gather(parts, vel.contiguous())
        full = torch.cat(parts, dim=1)                             # rank r owns matrices [r*chunk, (r+1)*chunk)
        g = G[i].float().reshape(cfg.reshape) * loss_scale
        blend = w * full[0] + (1 - w) * full[1]
        X = g.lerp(blend, rb)                                       # Nesterov input to the cascade
        uu = u[i].float().reshape(cfg.reshape)
        tall = X.shape[-2] > X.shape[-1]
        A = X.transpose(-1, -2) @ X if tall else X @ X.transpose(-1, -2)
        e, Q = torch.linalg.eigh(A.double())
        e = e.clamp_min(0)
        sig = e.sqrt()
        d = e.sum(-1, keepdim=True).sqrt() * FROBENIUS_MARGIN + 1e-6
        gain = cascade_map(sig / d) / sig.clamp_min(1e-30)
        gain = torch.where(sig > 1e-12 * sig.amax(-1, keepdim=True), gain, torch.zeros_like(gain))
        Pg = (Q * gain.unsqueeze(-2)) @ Q.transpose(-1, -2)          # Q diag(gain) Q^T
        MX = (X.double() @ Pg) if tall else (Pg @ X.double())
        num = -(uu.double() * MX).sum((-2, -1))
        den = MX.square().sum((-2, -1)).clamp_min(1e-300)
        c = (num / den).clamp_min(0)                                # per-matrix scale, fitted to the actual update
        cos = num / (den.sqrt() * uu.double().square().sum((-2, -1)).sqrt().clamp_min(1e-300))
        Sm = (Q * (c.unsqueeze(-1) * gain * loss_scale).sqrt().unsqueeze(-2)) @ Q.transpose(-1, -2)
        S.append((Sm.float(), tall, tuple(cfg.reshape)))
        live = torch.isfinite(cos) & (c > 0)   # frozen / gradient-free matrices have no update (0/0)
        cal[o.names[i]] = dict(n_matrices=int(cos.numel()), n_updated=int(live.sum()),
                               cos_median=float(cos[live].median()) if live.any() else float("nan"),
                               cos_min=float(cos[live].min()) if live.any() else float("nan"),
                               c_median=float(c[live].median()) if live.any() else float("nan"))
    thr = anvil_threshold(beta_f, RAIL_SLOW_BETA, w, rb)
    return S, cal, dict(beta_fast=beta_f, beta_slow=RAIL_SLOW_BETA, fast_weight=w, nesterov=rb, threshold=thr)


def apply_S(S, v, params):
    out = []
    for Sm, x in zip(S, v):
        if Sm is None:
            out.append(torch.zeros_like(x))
            continue
        m, tall, shape = Sm
        X = x.float().reshape(shape)
        Y = X @ m if tall else m @ X
        out.append(Y.reshape(x.shape).to(x.dtype))
    return out


def main():
    export, out_dir, steps = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
    run_cfg = json.load(open(os.path.join(export, "full_batch/config.json")))
    args = [f"{k}={v}" for k, v in run_cfg.items() if v is not None and not isinstance(v, (list, dict))]
    args += ["resume=False", "dyn=False", "token_loss_every=0", f"out_dir={out_dir}"]
    cfg = parse_config(args)
    orig = {}
    for line in open(os.path.join(export, "full_batch/metrics.jsonl")):
        r = json.loads(line)
        if "train_loss" in r and "event" not in r:
            orig[r["step"]] = r
    t = FullBatchTrainer(cfg)
    os.makedirs(out_dir, exist_ok=True)
    for step in steps:
        T0 = time.time()
        assert load_hf_checkpoint(t, os.path.join(export, f"full_batch/checkpoints/step_{step:06d}")) == step
        cad = Cadence(val=1, norms=1, microbatch_stats=0, spectra=0, line_probe=1, curvature=1, lanczos=1,
                      lanczos_k=30, group_lanczos_every=1, precond_lanczos=1, slq=0, raw_grad=0, raw_update=0,
                      checkpoint=0)
        suite = DynamicsSuite(t, os.path.join(out_dir, f"suite_step_{step:06d}"), cad)
        rec = suite.train_step(step)
        o, G, u = suite.o, suite.prev_G, suite.prev_u
        theta1 = o.get_params()
        theta0 = [a - b for a, b in zip(theta1, u)]
        res = dict(step=step, train_loss_recomputed=rec["train_loss"], train_loss_logged=orig.get(step, {}).get("train_loss"))
        res.update({k: v for k, v in rec.items() if k.startswith(("sharp/", "line/", "curv/", "val_loss", "norm/"))})
        groups = suite.groups
        o.begin_probe()
        try:
            o.set_params(theta0)
            for gname, idx in groups.items():
                ug = [x if i in idx else torch.zeros_like(x) for i, x in enumerate(u)]
                lp = line_probe(o, "probe", theta0, ug, alphas=cad.line_alphas)
                res[f"block_line/{gname}/a_star"] = lp["line/a_star"]
                res[f"block_line/{gname}/stability_ratio"] = (lp["line/uHu_fit"] / (-2 * lp["line/g_dot_u"])
                                                             if lp["line/g_dot_u"] != 0 else float("nan"))
                res[f"block_line/{gname}/dL_actual"] = lp["line/dL_actual"]
                res[f"block_line/{gname}/update_norm"] = float(sum(x.float().pow(2).sum() for x in ug) ** 0.5)
            S, cal, rails = anvil_preconditioner(t, o, G, u, step, cfg.loss_scale)
            res["anvil_precond/calibration"] = cal
            res["anvil_precond/rails"] = rails
            hvp = HVP(o, "hvp", mode="fd", eps_rel=cad.hvp_eps_rel)

            def shs(v):
                return apply_S(S, hvp(apply_S(S, v, o.params)), o.params)

            gen = torch.Generator(device=t.device).manual_seed(3000 + step)
            v0 = apply_S(S, vrandn_like(G, gen), o.params)
            r = lanczos(shs, v0, 20)
            res["anvil_precond/lambda_max"] = float(r.ritz_values[-1])
            res["anvil_precond/ritz_top5"] = [float(x) for x in r.ritz_values[-5:]]
            res["anvil_precond/over_threshold"] = res["anvil_precond/lambda_max"] / rails["threshold"]
        finally:
            o.end_probe(theta1)
        suite.close()
        res["seconds"] = time.time() - T0
        if t.master:
            with open(os.path.join(out_dir, f"step_{step:06d}.json"), "w") as f:
                json.dump(res, f, indent=1)
            print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
