"""Glue between fb/fullbatch/trainer.py (FullBatchTrainer) and instr/dynamics.py.

Usage in the full-batch training loop (replaces t.train_step):

    from fb_hooks import DynamicsSuite, Cadence
    suite = DynamicsSuite(t, run_dir, Cadence(...), probe_tokens=..., seed=0)
    for step in range(cfg.steps):
        rec = suite.train_step(step)        # accumulate (+free stats) -> optimizer step -> probes
    suite.close()

Everything is a collective (all ranks call it); params and mean gradients are replicated, so the
dynamics math runs redundantly on every rank and only rank 0 writes files.

Ordering inside train_step (probes call t.accumulate, which zeroes the accumulators, so every probe
runs AFTER the optimizer step has consumed the full gradient):
  1. t.accumulate(on_microbatch=...)    full gradient G_t at theta_t + per-microbatch stats
  2. G = t.mean_grads(); theta0 = params; finalize microbatch stats; norms
  3. t.optimizer_step(step);  u = theta1 - theta0
  4. cadenced probes at theta0 (line probe along u, HVPs, Lanczos, SLQ, Adam-preconditioned Lanczos)
  5. params AND every model buffer (fp8 caches, delayed fp8 scales, amax histories) restored to
     exactly their post-step state, so probing cannot perturb the training trajectory
  6. cadenced val loss, spectra, raw artifact dumps (async)

FP8 state: the training forward reads fp8 weight caches whose (delayed) scales depend on the history
of refreshes and on activation amaxes recorded by every forward/backward. FBOracle.begin_probe()
snapshots all of it; during probing every set_params() first restores that snapshot and then
requantizes, so the quantization is a deterministic function of (snapshot, params) -- this matters for
finite differences (both +eps and -eps see the same scales) -- and end_probe() puts back the exact
post-step state (params bit-exact, buffers bit-exact, no requantize).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from dynamics import (HVP, DynamicsLogger, MaskedHVP, MicrobatchStats, ScaledHVP, lanczos, line_probe,
                      matrix_spectra_distributed, slq_density, spectral_summary, step_norms, subspace_fraction, vdot,
                      vnorm, vrandn_like, vsub)

# Buffers never snapshotted: large and either static or rewritten before every use.
_SKIP_BUFFERS = ("ngram_cache", "canon_mask", "prefix_table")


@dataclass
class Cadence:
    """Every-k-steps cadences (0 = never). See NOTES.md [instr] for the cost table."""
    val: int = 1
    norms: int = 1
    microbatch_stats: int = 1
    spectra: int = 1             # weight / gradient / update singular values (Gram method)
    line_probe: int = 1          # len(line_alphas)+1 probe forwards
    curvature: int = 5           # u.H.u and g.H.g (2 FD HVPs)
    lanczos: int = 25            # top-k sharpness + Ritz vectors (k HVPs) + ANVIL-block (k/2)
    lanczos_k: int = 20
    group_lanczos: tuple = ("anvil", "embed", "other")   # per-block sharpness, k/2 HVPs each
    group_lanczos_every: int = 100
    precond_lanczos: int = 25    # Adam-preconditioned sharpness of the Adam params (k/2 HVPs)
    slq: int = 200               # spectral density (n_vectors * k HVPs)
    slq_vectors: int = 4
    slq_k: int = 40
    raw_grad: int = 1            # bf16 full gradient of dense params
    raw_update: int = 1          # bf16 update of dense params
    checkpoint: int = 100        # dense params fp32 (n-gram table and optimizer state excluded)
    hvp_eps_rel: float = 1e-3    # maxrel rule; validated in validate_hvp2 / test_fb_oracle2
    curv_u_frac: float = 0.25    # u.H.u perturbation = this fraction of the step
    line_alphas: tuple = (-0.5, 0.25, 0.5, 1.0, 1.5, 2.0)


def _due(step, k):
    return k > 0 and step % k == 0


class FBOracle:
    """dynamics.Oracle over a FullBatchTrainer. Probes are fixed sets of LOCAL microbatch ids per rank."""

    def __init__(self, t, probes: dict[str, list[int]]):
        self.t = t
        self.mbs = {k: t.fixed_mbs(v) for k, v in probes.items()}
        tied = t.cfg.tied_embed
        self.named = [(l, p) for l, p in t.params() if not (tied and l == "embed")]
        self.names = [l for l, _ in self.named]
        self.params = [p for _, p in self.named]
        self.dist_group = None  # replicated
        self._snap = None

    # ---- fp8/buffer state
    def _buffers(self):
        return {n: b for n, b in self.t.raw_model.named_buffers() if n.split(".")[-1] not in _SKIP_BUFFERS}

    def begin_probe(self):
        m = self.t.raw_model
        self._snap = ({n: b.detach().clone() for n, b in self._buffers().items()},
                      getattr(m, "mlp_quantize_calls", None))

    def _restore_buffers(self):
        bufs, calls = self._snap
        live = self._buffers()
        with torch.no_grad():
            for n, b in bufs.items():
                live[n].copy_(b)
        if calls is not None:
            self.t.raw_model.mlp_quantize_calls = calls

    def end_probe(self, theta_live):
        """Put back the live params and the exact post-step buffer state (no requantize)."""
        with torch.no_grad():
            for p, v in zip(self.params, theta_live):
                p.copy_(v)
            if self.t.cfg.tied_embed:
                self.t.raw_model.embed.weight.copy_(self.t.raw_model.lm_head.weight.T)
        self._restore_buffers()
        self._snap = None

    # ---- Oracle interface
    def get_params(self):
        return [p.detach().float().clone() for p in self.params]

    @torch.no_grad()
    def set_params(self, vec):
        for p, v in zip(self.params, vec):
            p.copy_(v)
        if self.t.cfg.tied_embed:
            self.t.raw_model.embed.weight.copy_(self.t.raw_model.lm_head.weight.T)
        if self._snap is not None:
            self._restore_buffers()
        self.t.refresh_fp8()

    def loss_and_grad(self, probe):
        loss, _ = self.t.accumulate(mbs=self.mbs[probe])
        g = self.t.mean_grads()
        return loss, [g[n] for n in self.names]

    def loss(self, probe):
        return self.t.loss_on(self.mbs[probe])

    # ---- Adam preconditioner (for preconditioned sharpness)
    @torch.no_grad()
    def adam_precond_sqrt(self):
        """Vec s with s^2 = Adam's per-element step multiplier step_size / (sqrt(v) + eps) for Adam params
        (bias-corrected, current lr), 0 for ANVIL params; plus the Adam beta1 per tensor."""
        opt = self.t.optimizer
        s, beta1 = [], []
        for name, p in self.named:
            cfg = opt.param_cfgs[p]
            if cfg.optim != "adam" or p not in opt.param_states:
                s.append(torch.zeros_like(p, dtype=torch.float32)); beta1.append(None); continue
            st = opt.param_states[p]
            v = st["exp_avg_sq"]
            if cfg.comms == "sharded" and self.t.world > 1:
                full = torch.empty((v.shape[0] * self.t.world, *v.shape[1:]), dtype=v.dtype, device=v.device)
                dist.all_gather_into_tensor(full, v.contiguous())
                v = full[:p.shape[0]]
            v = v.reshape(p.shape)
            b1, b2 = cfg.adam_betas
            tstep = max(st["step"], 1)
            step_size = cfg.lr * cfg.lr_mul * (1 - b2 ** tstep) ** 0.5 / (1 - b1 ** tstep)
            s.append((step_size / (v.sqrt() + cfg.eps)).sqrt())
            beta1.append(b1)
        return s, beta1


def pick_probe_ids(t, tokens: int, seed: int, exclude: set | None = None) -> list[int]:
    """Same number of local microbatches on every rank, chosen deterministically."""
    per_rank = max(1, round(tokens / (t.cfg.micro_tokens * t.world)))
    g = torch.Generator().manual_seed(seed * 7919 + t.rank)
    ids = [i for i in torch.randperm(len(t.data), generator=g).tolist() if not exclude or i not in exclude]
    return ids[:per_rank]


class DynamicsSuite:
    def __init__(self, t, run_dir: str, cadence: Cadence = Cadence(), probe_tokens: int = 1_572_864,
                 hvp_probe_tokens: int = 1_572_864, seed: int = 0):
        self.t, self.c = t, cadence
        self.log = DynamicsLogger(run_dir, enabled=(t.rank == 0))
        probe = pick_probe_ids(t, probe_tokens, seed)
        hvp_probe = probe[:max(1, round(hvp_probe_tokens / (t.cfg.micro_tokens * t.world)))]
        self.o = FBOracle(t, {"probe": probe, "hvp": hvp_probe})
        self.names = self.o.names
        self.label_param = dict(t.params())
        self.mbstats = MicrobatchStats(self.names)
        self.prev_G = self.prev_u = None
        self.matrix_names = [n for n in self.names if n in ("qk_bank", "vo_bank", "mlp_bank", "lm_head", "embed")]
        self.groups = {"anvil": {i for i, n in enumerate(self.names)
                                 if t.optimizer.param_cfgs[self.o.params[i]].optim == "anvil"},
                       "embed": {i for i, n in enumerate(self.names) if n in ("lm_head", "embed", "value_embeds")}}
        self.groups["other"] = set(range(len(self.names))) - self.groups["anvil"] - self.groups["embed"]
        if t.rank == 0:
            self.log.log(-1, {"probe_microbatches_per_rank": len(probe), "hvp_microbatches_per_rank": len(hvp_probe),
                              "names": self.names, "numels": [p.numel() for p in self.o.params]})

    def close(self):
        self.log.close()

    # -------------------------------------------------------------- per-microbatch callback
    def _on_mb(self, info):
        gd, n = info["grads"], info["tokens"]
        tied = self.t.cfg.tied_embed
        vec = []
        for name in self.names:
            p = self.label_param[name]
            g = gd.get(p)
            if g is None:
                g = torch.zeros((), device=self.t.device)
            elif tied and name == "lm_head" and gd.get(self.label_param["embed"]) is not None:
                g = g.float() + gd[self.label_param["embed"]].float().T
            vec.append(g)
        self.mbstats.on_microbatch(vec, n, loss=info["loss_sum"], scale=1.0 / n)

    # -------------------------------------------------------------- the instrumented step
    def train_step(self, step: int) -> dict:
        t, c = self.t, self.c
        rec, arr = {}, {}
        T0 = time.perf_counter()
        want_mb = _due(step, c.microbatch_stats)
        self.mbstats.reset()
        loss, tokens = t.accumulate(on_microbatch=self._on_mb if want_mb else None, step=step)
        torch.cuda.synchronize()
        T1 = time.perf_counter()
        Gd = t.mean_grads()
        G = [Gd[n] for n in self.names]
        theta0 = self.o.get_params()
        rec.update(train_loss=loss, tokens=tokens, t_grad=T1 - T0, lr_mul=t.lr_multiplier(step))
        if want_mb:
            s, a = self.mbstats.finalize(G)
            rec.update(s); arr.update(a)
        t.optimizer_step(step)
        torch.cuda.synchronize()
        theta1 = self.o.get_params()
        u = vsub(theta1, theta0)
        rec["t_opt"] = time.perf_counter() - T1
        if _due(step, c.norms):
            s, a = step_norms(self.names, theta0, G, u, self.prev_G, self.prev_u)
            rec.update(s); arr.update(a)
            rec["full/g_dot_u"] = vdot(G, u)
        if _due(step, c.spectra):
            Ts = time.perf_counter()
            mats = {f"{kind}/{n}": vec[self.names.index(n)] for kind, vec in (("w", theta0), ("g", G), ("u", u))
                    for n in self.matrix_names}
            sv = matrix_spectra_distributed(mats, t.rank, t.world)
            arr.update(sv)
            for k, v in sv.items():
                for kk, vv in spectral_summary(v).items():
                    rec[f"spec/{k}/{kk}"] = vv
            rec["t_spectra"] = time.perf_counter() - Ts

        # ---- probes at theta0; the exact post-step state (theta1 + buffers) is restored afterwards
        probing = any(_due(step, k) for k in (c.line_probe, c.curvature, c.lanczos, c.precond_lanczos, c.slq))
        T2 = time.perf_counter()
        if probing:
            self.o.begin_probe()
            try:
                self.o.set_params(theta0)
                if _due(step, c.line_probe):
                    lp = line_probe(self.o, "probe", theta0, u, alphas=c.line_alphas)
                    for k, v in lp.items():
                        (arr if isinstance(v, np.ndarray) else rec)[k] = v
                    rec["line/stability_ratio"] = (rec["line/uHu_fit"] / (-2 * rec["line/g_dot_u"])
                                                   if rec["line/g_dot_u"] != 0 else float("nan"))
                if any(_due(step, k) for k in (c.curvature, c.lanczos, c.precond_lanczos, c.slq)):
                    hvp = HVP(self.o, "hvp", mode="fd", eps_rel=c.hvp_eps_rel)
                    if _due(step, c.curvature):
                        # along u: perturb by a fraction of the actual step (a maxrel eps would be far
                        # below the fp8 weight resolution for a ~1%-relative update); along g: maxrel.
                        hvp_u = HVP(self.o, "hvp", mode="fd", eps_rel=c.curv_u_frac * vnorm(u), eps_rule="abs")
                        for nm, d in (("u", u), ("g", G)):
                            hd = (hvp_u if nm == "u" else hvp)(d)
                            dd = vdot(d, d)
                            rec[f"curv/{nm}Hv"] = vdot(hd, d)
                            rec[f"curv/{nm}Hv_over_vv"] = rec[f"curv/{nm}Hv"] / dd
                            if nm == "u":
                                rec["curv/uHu_over_neg2gu"] = rec["curv/uHv"] / (-2 * vdot(G, u))
                    if _due(step, c.lanczos):
                        gen = torch.Generator(device=t.device).manual_seed(1000 + step)
                        r = lanczos(hvp, vrandn_like(G, gen), c.lanczos_k, return_top=5)
                        arr["lanczos_ritz"] = r.ritz_values
                        arr["lanczos_alphas"], arr["lanczos_betas"] = r.alphas, r.betas
                        rec["sharp/lambda_max"] = float(r.ritz_values[-1])
                        rec["sharp/lambda_min"] = float(r.ritz_values[0])
                        arr["frac_G_in_top5"] = subspace_fraction(G, r.ritz_vectors)
                        arr["frac_u_in_top5"] = subspace_fraction(u, r.ritz_vectors)
                        arr["top_vec_tensor_mass"] = np.array([float(x.pow(2).sum()) for x in r.ritz_vectors[0]])
                        rec["sharp/lanczos_seconds"] = r.seconds
                        del r
                        for gname in (c.group_lanczos if _due(step, c.group_lanczos_every) else ()):
                            idx = self.groups[gname]
                            v0 = [x if i in idx else torch.zeros_like(x) for i, x in enumerate(vrandn_like(G, gen))]
                            rg = lanczos(MaskedHVP(hvp, idx), v0, max(8, c.lanczos_k // 2))
                            rec[f"sharp/{gname}_lambda_max"] = float(rg.ritz_values[-1])
                            arr[f"{gname}_lanczos_ritz"] = rg.ritz_values
                    if _due(step, c.precond_lanczos):
                        s, beta1 = self.o.adam_precond_sqrt()
                        gen = torch.Generator(device=t.device).manual_seed(2000 + step)
                        rp = lanczos(ScaledHVP(hvp, s), [x * (si > 0) for x, si in zip(vrandn_like(G, gen), s)],
                                     max(8, c.lanczos_k // 2))
                        rec["sharp/adam_precond_lambda_max"] = float(rp.ritz_values[-1])
                        arr["adam_precond_ritz"] = rp.ritz_values
                        b1 = [b for b in beta1 if b is not None]
                        rec["sharp/adam_eos_threshold_min_beta1"] = (2 + 2 * min(b1)) / (1 - min(b1))
                    if _due(step, c.slq):
                        sq = slq_density(hvp, G, c.slq_vectors, c.slq_k, seed=step, basis_dtype=torch.bfloat16)
                        arr.update(sq)
                    rec["hvp_calls"] = hvp.n_calls
            finally:
                self.o.end_probe(theta1)
        rec["t_probe"] = time.perf_counter() - T2

        if _due(step, c.val):
            T3 = time.perf_counter()
            rec.update(t.evaluate())
            rec["t_val"] = time.perf_counter() - T3
        Tw = time.perf_counter()
        if t.rank == 0:
            if _due(step, c.raw_grad):
                self.log.save_raw(step, "grad", self.names, G)
            if _due(step, c.raw_update):
                self.log.save_raw(step, "update", self.names, u)
            if _due(step, c.checkpoint):
                self.log.save_raw(step, "params", self.names, theta1, dtype=torch.float32)
        rec["t_save"] = time.perf_counter() - Tw
        rec["t_total"] = time.perf_counter() - T0
        self.prev_G, self.prev_u = G, u
        self.log.log(step, rec, arr)
        return rec
