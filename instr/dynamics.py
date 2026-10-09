"""Learning-dynamics instrumentation for the full-batch run.

Everything here works on *lists of tensors* ("vectors" = one tensor per tracked parameter), so it
is agnostic to how the trainer stores parameters. The trainer supplies an Oracle (see below); this
module never touches the model directly.

Main pieces
-----------
- vector helpers (dot/norm/axpy on tensor lists, with an optional all-reduce for sharded params)
- MicrobatchStats: free per-microbatch gradient statistics collected during the full-batch
  accumulation (gradient noise scale, per-microbatch alignment with the previous full gradient)
- HVP: Hessian-vector products on a fixed probe subset. Two modes:
    "fd"    central finite difference of probe gradients, (g(t+eps v) - g(t-eps v)) / 2 eps.
            Works with any kernels, but needs the perturbation to survive the forward's weight
            precision (see validate_hvp.py: fails under fp8 weights, marginal under bf16 weights,
            good with fp32 master weights fed to bf16/fp32 compute).
    "exact" oracle.hvp(probe, v) (double backward on a reference forward), if the oracle has one.
- lanczos(): top-k Ritz values/vectors with full reorthogonalization; slq_density(): stochastic
  Lanczos quadrature spectral density.
- line_probe(): probe loss along the actual update, theta + a*u for several a, plus the
  quadratic-model quantities (g.u, u.H.u from the fit, optimal step multiple a*).
- matrix_spectra(): singular values of weight / gradient / update matrices.
- DynamicsLogger: per-step jsonl scalars + per-step npz arrays + optional raw bf16 artifacts.

Oracle interface (implemented by the full-batch trainer; a toy one lives in validate_hvp.py)
------------------------------------------------------------------------------------------
    oracle.names: list[str]                       one name per tracked tensor
    oracle.get_params() -> list[Tensor]           fp32 copies of the tracked params
    oracle.set_params(list[Tensor])               write them back (in whatever storage dtype)
    oracle.loss_and_grad(probe) -> (float, list)  mean loss and fp32 grad on a FIXED probe set
    oracle.loss(probe) -> float                   forward only
    oracle.per_token_loss(probe) -> Tensor        optional, [T] fp32
    oracle.hvp(probe, v) -> list                  optional exact HVP
    oracle.dist_group                             None, or a process group if vectors are sharded
                                                  (dots are then all-reduced over it)
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

Vec = list[Tensor]


# ----------------------------------------------------------------------------------------------
# vector helpers

def _allreduce(x: Tensor, group) -> Tensor:
    if group is not None:
        dist.all_reduce(x, group=group)
    return x


def vdot(a: Vec, b: Vec, group=None) -> float:
    s = torch.zeros((), dtype=torch.float64, device=a[0].device)
    for x, y in zip(a, b):
        s += torch.dot(x.reshape(-1).double(), y.reshape(-1).double()) if x.numel() < 2**24 else \
            (x.float() * y.float()).sum(dtype=torch.float64)
    return float(_allreduce(s, group))


def vdot_groups(a: Vec, b: Vec) -> Tensor:
    """Per-tensor dots (local, float64)."""
    return torch.stack([(x.float() * y.float()).sum(dtype=torch.float64) for x, y in zip(a, b)])


def vnorm(a: Vec, group=None) -> float:
    return math.sqrt(max(vdot(a, a, group), 0.0))


def vscale(a: Vec, s: float) -> Vec:
    return [x * s for x in a]


def vaxpy_(y: Vec, a: float, x: Vec) -> Vec:
    """y += a*x in place."""
    torch._foreach_add_(y, [xi.to(yi.dtype) for xi, yi in zip(x, y)], alpha=a)
    return y


def vsub(a: Vec, b: Vec) -> Vec:
    return [x.float() - y.float() for x, y in zip(a, b)]


def vrandn_like(a: Vec, gen: torch.Generator | None = None) -> Vec:
    return [torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=gen) for x in a]


def vrademacher_like(a: Vec, gen: torch.Generator | None = None) -> Vec:
    return [torch.randint(0, 2, x.shape, device=x.device, generator=gen).float() * 2 - 1 for x in a]


# ----------------------------------------------------------------------------------------------
# free per-microbatch statistics (collected inside the full-batch accumulation loop)

class MicrobatchStats:
    """Call reset() before a step's accumulation, on_microbatch(grads, ntokens) after each
    microbatch backward with that microbatch's OWN (non-accumulated) mean-over-tokens gradient,
    and finalize(full_grad) after the step's gradient is complete.

    With token weights w_i = n_i / N and full gradient G = sum_i w_i g_i, the token-weighted
    microbatch variance  V = sum_i w_i ||g_i - G||^2 = sum_i w_i ||g_i||^2 - ||G||^2  is exact.
    The "simple" gradient noise scale (McCandlish et al. 2018) in tokens is then
    B_simple ~= n_bar * V / ||G||^2  (n_bar = mean microbatch tokens): the batch size (in tokens) at
    which minibatch noise equals the signal. Per-group versions are also kept.
    Cost: one squared norm (and optionally one dot with the previous full gradient) per microbatch
    over the dense params -- reading ~0.6 GB bf16, negligible next to a microbatch fwd/bwd.
    """

    def __init__(self, names: list[str], keep_per_microbatch: bool = True, track_prev_dot: bool = True):
        self.names = names
        self.keep = keep_per_microbatch
        self.track_prev_dot = track_prev_dot
        self.prev_full: Vec | None = None
        self.reset()

    def reset(self):
        self.n_tot = 0
        self.sq_w = None          # per-group sum_i n_i ||g_i||^2
        self.mb_sq = []           # per microbatch total ||g_i||^2
        self.mb_dot_prev = []     # per microbatch <g_i, G_{t-1}>
        self.mb_n = []
        self.mb_loss = []

    @torch.no_grad()
    def on_microbatch(self, grads: Vec, ntokens: int, loss=None, scale: float = 1.0):
        """grads: this microbatch's gradient tensors (any dtype); scale converts them to the mean-token
        gradient (e.g. 1/ntokens for token-summed grads). No host syncs and no fp32 copies: norms use
        fp32-accumulating reductions, so this costs ~1 ms per microbatch."""
        sq = torch.stack([torch.linalg.vector_norm(g, dtype=torch.float32).double().square() for g in grads]) * scale ** 2
        self.sq_w = sq * ntokens if self.sq_w is None else self.sq_w + sq * ntokens
        self.n_tot += ntokens
        if self.keep:
            self.mb_sq.append(sq.sum())
            self.mb_n.append(ntokens)
            if loss is not None:
                self.mb_loss.append(torch.as_tensor(loss, device=sq.device, dtype=torch.float64) / ntokens)
            if self.prev_full is not None and self.track_prev_dot:
                d = sum((g * p.to(g.dtype)).sum(dtype=torch.float32).double() for g, p in zip(grads, self.prev_full)
                        if g.ndim > 0)
                self.mb_dot_prev.append(d * scale)

    @torch.no_grad()
    def finalize(self, full_grad: Vec) -> tuple[dict, dict]:
        G2 = torch.stack([g.float().pow(2).sum(dtype=torch.float64) for g in full_grad])
        Ew = self.sq_w / self.n_tot                       # sum_i w_i ||g_i||^2, per group
        V = (Ew - G2).clamp_min(0)
        n_bar = self.n_tot / max(len(self.mb_n), 1) if self.mb_n else float("nan")
        scal = {
            "gns/var_total": float(V.sum()), "gns/G2_total": float(G2.sum()),
            "gns/B_simple_tokens": float(n_bar * V.sum() / G2.sum()) if self.mb_n else float("nan"),
            "gns/microbatch_tokens": n_bar,
        }
        arrays = {"gns_var_per_group": V.cpu().numpy(), "gns_G2_per_group": G2.cpu().numpy()}
        if self.keep and self.mb_sq:
            arrays["mb_sqnorm"] = torch.stack(self.mb_sq).cpu().numpy().astype(np.float32)
            arrays["mb_ntokens"] = np.asarray(self.mb_n, dtype=np.int32)
            if self.mb_loss:
                arrays["mb_loss"] = torch.stack(self.mb_loss).cpu().numpy().astype(np.float32)
            if self.mb_dot_prev:
                d = torch.stack(self.mb_dot_prev).cpu().numpy()
                arrays["mb_dot_prevG"] = d.astype(np.float32)
                scal["gns/frac_mb_neg_align_prevG"] = float((d < 0).mean())
        self.prev_full = [g.detach().float().clone() for g in full_grad]
        return scal, arrays


# ----------------------------------------------------------------------------------------------
# Hessian-vector products on a fixed probe subset

class HVP:
    """H v on probe `probe`. mode="fd": central difference of probe gradients.

    Step size rules (eps_rule), r = eps_rel:
      "maxrel" (default): the largest per-tensor relative change is r, i.e.
                eps = r / max_i(||v_i|| / ||theta_i||). Directions concentrated in a few small tensors
                (top eigenvectors, the gradient) then get a small step (low truncation error), while
                spread-out directions get a step large enough to survive low-precision weights.
      "global": ||eps v|| = r * ||theta||   (bad when ||theta|| is dominated by big embeddings)
      "abs":    ||eps v|| = r
    richardson=True uses +-eps and +-2eps (4 gradients) to cancel the O(eps^2) truncation term.
    Each plain fd HVP costs 2 probe gradients; the base point is restored exactly from a saved copy."""

    def __init__(self, oracle, probe, mode: str = "fd", eps_rel: float = 1e-3, eps_rule: str = "maxrel",
                 richardson: bool = False):
        self.o, self.probe, self.mode = oracle, probe, mode
        self.eps_rel, self.eps_rule, self.richardson = eps_rel, eps_rule, richardson
        self.group = getattr(oracle, "dist_group", None)
        self.n_calls = 0
        self.n_grads = 0
        self.refresh_base()

    def refresh_base(self):
        self.theta0 = self.o.get_params()
        self.theta_norm = vnorm(self.theta0, self.group)
        self.tensor_norms = [float(_allreduce(x.float().pow(2).sum(dtype=torch.float64), self.group).sqrt())
                             for x in self.theta0]

    def _eps(self, v: Vec) -> float:
        vn = vnorm(v, self.group)
        if self.eps_rule == "global":
            return self.eps_rel * self.theta_norm / vn
        if self.eps_rule == "abs":
            return self.eps_rel / vn
        ratios = [float(_allreduce(x.float().pow(2).sum(dtype=torch.float64), self.group).sqrt()) / max(tn, 1e-12)
                  for x, tn in zip(v, self.tensor_norms)]
        return self.eps_rel / max(max(ratios), 1e-30)

    def _g(self, v: Vec, s: float) -> Vec:
        self.o.set_params([t + s * x for t, x in zip(self.theta0, v)])
        self.n_grads += 1
        return self.o.loss_and_grad(self.probe)[1]

    def __call__(self, v: Vec) -> Vec:
        self.n_calls += 1
        if self.mode == "exact":
            return [h.float() for h in self.o.hvp(self.probe, v)]
        if vnorm(v, self.group) == 0:
            return [torch.zeros_like(x) for x in v]
        eps = self._eps(v)
        try:
            gp, gm = self._g(v, eps), self._g(v, -eps)
            d1 = [(a.float() - b.float()) / (2 * eps) for a, b in zip(gp, gm)]
            del gp, gm
            if self.richardson:
                gp2, gm2 = self._g(v, 2 * eps), self._g(v, -2 * eps)
                d2 = [(a.float() - b.float()) / (4 * eps) for a, b in zip(gp2, gm2)]
                d1 = [(4 * a - b) / 3 for a, b in zip(d1, d2)]
        finally:
            self.o.set_params(self.theta0)
        return d1


class MaskedHVP:
    """Hessian block restricted to a subset of the tracked tensors (others treated as fixed):
    v -> M H M v, where M zeroes the tensors whose index is not in `keep`. Use it for per-group
    sharpness (e.g. only the ANVIL matrix banks, only embeddings)."""

    def __init__(self, hvp, keep: set[int]):
        self.hvp, self.keep = hvp, keep

    def __call__(self, v: Vec) -> Vec:
        vm = [x if i in self.keep else torch.zeros_like(x) for i, x in enumerate(v)]
        hv = self.hvp(vm)
        return [x if i in self.keep else torch.zeros_like(x) for i, x in enumerate(hv)]


class ScaledHVP:
    """Symmetrically preconditioned Hessian S H S with S = diag(s) (s a Vec of per-element scales).
    For an Adam-type group with step  -lr * m / (sqrt(v) + eps), s = sqrt(lr / (sqrt(v) + eps)) gives
    the 'preconditioned sharpness' of Cohen et al. 2022 (adaptive edge of stability: Adam with
    momentum beta1 is stable while lambda_max(P^-1 H) < (2 + 2 beta1) / ((1 - beta1) lr), here with lr
    folded into s so the threshold is (2 + 2 beta1) / (1 - beta1))."""

    def __init__(self, hvp, s: Vec):
        self.hvp, self.s = hvp, s

    def __call__(self, v: Vec) -> Vec:
        hv = self.hvp([x * si for x, si in zip(v, self.s)])
        return [x * si for x, si in zip(hv, self.s)]


# ----------------------------------------------------------------------------------------------
# Lanczos / SLQ

@dataclass
class LanczosResult:
    alphas: np.ndarray
    betas: np.ndarray
    ritz_values: np.ndarray          # ascending
    ritz_weights: np.ndarray         # (first component of eigvecs)^2, for SLQ
    ritz_vectors: list | None = None  # top-k Ritz vectors as Vecs (if requested)
    n_hvp: int = 0
    seconds: float = 0.0


def lanczos(hvp, v0: Vec, k: int, group=None, reorth: bool = True, basis_dtype=torch.float32,
            basis_device=None, return_top: int = 0) -> LanczosResult:
    """k-step Lanczos with full reorthogonalization against the stored basis.
    Memory: k vectors of the tracked-param size (basis_dtype/basis_device configurable:
    e.g. bf16 on GPU halves it; 'cpu' moves it off the GPU)."""
    t0 = time.time()
    dev = basis_device
    q = vscale(v0, 1.0 / vnorm(v0, group))
    Q = []
    alphas, betas = [], []
    beta, q_prev = 0.0, None
    for j in range(k):
        Q.append([x.to(dtype=basis_dtype, device=dev or x.device) for x in q])
        w = hvp(q)
        a = vdot(w, q, group)
        alphas.append(a)
        vaxpy_(w, -a, q)
        if q_prev is not None:
            vaxpy_(w, -beta, q_prev)
        if reorth:
            for _ in range(2):  # twice is enough (Kahan-Parlett)
                for qi in Q:
                    qi_ = [x.to(device=w[0].device, dtype=torch.float32) for x in qi]
                    c = vdot(w, qi_, group)
                    vaxpy_(w, -c, qi_)
        beta = vnorm(w, group)
        if j < k - 1:
            if beta < 1e-10:
                break
            betas.append(beta)
            q_prev, q = q, vscale(w, 1.0 / beta)
    m = len(alphas)
    T = np.diag(alphas) + np.diag(betas[: m - 1], 1) + np.diag(betas[: m - 1], -1)
    evals, evecs = np.linalg.eigh(T)
    res = LanczosResult(np.array(alphas), np.array(betas), evals, evecs[0, :] ** 2,
                        n_hvp=m, seconds=time.time() - t0)
    if return_top:
        vecs = []
        for idx in range(m - 1, max(m - 1 - return_top, -1), -1):
            y = evecs[:, idx]
            r = [torch.zeros_like(x, dtype=torch.float32, device=v0[0].device) for x in v0]
            for c, qi in zip(y, Q):
                vaxpy_(r, float(c), [x.to(device=r[0].device, dtype=torch.float32) for x in qi])
            vecs.append(r)
        res.ritz_vectors = vecs
    return res


def slq_density(hvp, template: Vec, n_vectors: int, k: int, group=None, seed: int = 0,
                **lanczos_kw) -> dict:
    """Stochastic Lanczos quadrature (Ghorbani et al. 2019): returns per-probe Ritz nodes/weights;
    the density is sum_v sum_i w_i delta(x - node_i) / n_vectors (smooth with a Gaussian to plot)."""
    gen = torch.Generator(device=template[0].device).manual_seed(seed)
    nodes, weights = [], []
    for _ in range(n_vectors):
        v = vrademacher_like(template, gen)
        r = lanczos(hvp, v, k, group=group, **lanczos_kw)
        nodes.append(r.ritz_values)
        weights.append(r.ritz_weights)
    return {"slq_nodes": np.stack(nodes), "slq_weights": np.stack(weights)}


def subspace_fraction(g: Vec, vecs: list[Vec], group=None) -> np.ndarray:
    """Cumulative fraction of ||g||^2 inside span of the (orthonormal) Ritz vectors, top first."""
    g2 = vdot(g, g, group)
    c = np.array([vdot(g, v, group) ** 2 for v in vecs])
    return np.cumsum(c) / g2


# ----------------------------------------------------------------------------------------------
# line probe along the actual update

def line_probe(oracle, probe, theta0: Vec, u: Vec, alphas=(-0.5, 0.25, 0.5, 1.0, 1.5, 2.0),
               g0: Vec | None = None, group=None) -> dict:
    """Probe loss at theta0 + a*u (forward only, len(alphas)+1 forwards). Leaves the params at
    theta0 on return; if theta0 is not the live point, the caller must restore the live params.
    Returns raw losses plus quadratic-model quantities:
      d1 = g.u (from g0 if given, else from the fit), curv = u.H.u (fit), a_star = -d1/curv
    (a_star < 0.5 means the step overshoots along its own direction by >2x)."""
    alphas = sorted(set(alphas) | {0.0})
    L = []
    try:
        for a in alphas:
            oracle.set_params([t + a * x for t, x in zip(theta0, u)])
            L.append(oracle.loss(probe))
    finally:
        oracle.set_params(theta0)
    A, Lv = np.array(alphas), np.array(L)
    c2, c1, c0 = np.polyfit(A, Lv, 2)  # L ~ c0 + c1 a + c2 a^2 near 0
    near = np.abs(A) <= 0.5
    if near.sum() >= 3:
        c2n, c1n, _ = np.polyfit(A[near], Lv[near], 2)
    else:
        c2n, c1n = c2, c1
    d1 = vdot(g0, u, group) if g0 is not None else c1n
    curv = 2 * c2n
    return {"line_alphas": A.astype(np.float32), "line_losses": Lv.astype(np.float64),
            "line/g_dot_u": d1, "line/uHu_fit": curv,
            "line/a_star": (-d1 / curv) if curv != 0 else float("nan"),
            "line/dL_actual": float(Lv[A == 1.0][0] - Lv[A == 0.0][0]) if 1.0 in alphas else float("nan"),
            "line/dL_quadratic": d1 + 0.5 * curv}


# ----------------------------------------------------------------------------------------------
# matrix spectra

@torch.no_grad()
def matrix_spectra_distributed(named: dict[str, Tensor], rank: int, world: int, method: str = "gram32") -> dict:
    """matrix_spectra with the matrices split round-robin over ranks and the (small) results
    all-gathered as objects, so the cost is ~1/world per rank. Every rank gets the full dict."""
    keys = sorted(named)
    mine = matrix_spectra({k: named[k] for i, k in enumerate(keys) if i % world == rank}, method=method)
    if world == 1:
        return mine
    allp = [None] * world
    dist.all_gather_object(allp, mine)
    out = {}
    for d in allp:
        out.update(d)
    return out


@torch.no_grad()
def matrix_spectra(named: dict[str, Tensor], method: str = "gram") -> dict[str, np.ndarray]:
    """Singular values of each 2D matrix, or of each slice of a stacked bank (leading dims flattened).
    method="gram": sqrt(eigvalsh(M^T M)) over the smaller dimension, Gram formed in fp64 -- ~100x faster
    than svdvals for the 50k x 768 lm_head, accurate for all but singular values < ~1e-7 * top.
    Rows of all-zero slices (unused bank slots) come out as zeros."""
    out = {}
    for name, M in named.items():
        if M.ndim < 2:
            continue
        Mf = M.detach()
        Mf = Mf.reshape(-1, *Mf.shape[-2:])
        if method in ("gram", "gram32"):
            A = Mf.double() if method == "gram" else Mf.float()
            G = A.transpose(-1, -2) @ A if A.shape[-2] >= A.shape[-1] else A @ A.transpose(-1, -2)
            ev = torch.linalg.eigvalsh(G).clamp_min(0).sqrt().flip(-1)
            out[f"sv/{name}"] = ev.float().cpu().numpy()
        else:
            out[f"sv/{name}"] = torch.linalg.svdvals(Mf.float()).cpu().numpy().astype(np.float32)
    return out


def spectral_summary(sv: np.ndarray) -> dict:
    """Scalar summaries per matrix (averaged over non-zero slices): top sv, stable rank
    ||W||_F^2/||W||_2^2, effective rank exp(entropy of normalized sv^2)."""
    s = sv.reshape(-1, sv.shape[-1]).astype(np.float64)
    s = s[s[:, 0] > 0]
    if len(s) == 0:
        return {"top": 0.0, "stable_rank": 0.0, "eff_rank": 0.0}
    p = (s ** 2) / (s ** 2).sum(-1, keepdims=True)
    erank = np.exp(-(p * np.log(p + 1e-300)).sum(-1))
    return {"top": float(s[:, 0].mean()), "stable_rank": float(((s ** 2).sum(-1) / s[:, 0] ** 2).mean()),
            "eff_rank": float(erank.mean())}


# ----------------------------------------------------------------------------------------------
# basic per-step norms / cosines (cheap, every step)

@torch.no_grad()
def step_norms(names, theta: Vec, grad: Vec, update: Vec | None, prev_grad: Vec | None,
               prev_update: Vec | None, group=None) -> tuple[dict, dict]:
    def per(a, b):
        return vdot_groups(a, b)
    out, arr = {}, {}
    gg = per(grad, grad)
    tt = per(theta, theta)
    arr["w_norm"] = tt.sqrt().cpu().numpy()
    arr["g_norm"] = gg.sqrt().cpu().numpy()
    out["norm/grad"] = float(gg.sum().sqrt())
    out["norm/weights"] = float(tt.sum().sqrt())
    if update is not None:
        uu = per(update, update)
        gu = per(grad, update)
        arr["u_norm"] = uu.sqrt().cpu().numpy()
        arr["cos_g_u"] = (gu / (gg.sqrt() * uu.sqrt() + 1e-30)).cpu().numpy()
        out["norm/update"] = float(uu.sum().sqrt())
        out["cos/grad_update"] = float(gu.sum() / (gg.sum().sqrt() * uu.sum().sqrt() + 1e-30))
        out["ratio/update_over_weight"] = float(uu.sum().sqrt() / tt.sum().sqrt())
        if prev_update is not None:
            pu = per(update, prev_update)
            ppu = per(prev_update, prev_update)
            arr["cos_u_uprev"] = (pu / (uu.sqrt() * ppu.sqrt() + 1e-30)).cpu().numpy()
            out["cos/update_prevupdate"] = float(pu.sum() / (uu.sum().sqrt() * ppu.sum().sqrt() + 1e-30))
    if prev_grad is not None:
        pg = per(grad, prev_grad)
        ppg = per(prev_grad, prev_grad)
        arr["cos_g_gprev"] = (pg / (gg.sqrt() * ppg.sqrt() + 1e-30)).cpu().numpy()
        out["cos/grad_prevgrad"] = float(pg.sum() / (gg.sum().sqrt() * ppg.sum().sqrt() + 1e-30))
    return out, arr


# ----------------------------------------------------------------------------------------------
# logger

class DynamicsLogger:
    """run_dir/dyn/scalars.jsonl  one json object per step (only scalars)
    run_dir/dyn/step_XXXXXX.npz    arrays for that step (spectra, per-microbatch stats, ...)
    run_dir/raw/{kind}_XXXXXX.pt   optional raw tensors (bf16), e.g. full gradient / update"""

    def __init__(self, run_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.dir = os.path.join(run_dir, "dyn")
        self.raw = os.path.join(run_dir, "raw")
        if enabled:
            os.makedirs(self.dir, exist_ok=True)
            os.makedirs(self.raw, exist_ok=True)
            self.f = open(os.path.join(self.dir, "scalars.jsonl"), "a")

    def log(self, step: int, scalars: dict, arrays: dict | None = None):
        if not self.enabled:
            return
        rec = {"step": step, "time": time.time()}
        rec.update({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                    for k, v in scalars.items()})
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        if arrays:
            np.savez(os.path.join(self.dir, f"step_{step:06d}.npz"),
                     **{k.replace("/", "__"): v for k, v in arrays.items()})

    def save_raw(self, step: int, kind: str, names: list[str], vec: Vec, dtype=torch.bfloat16):
        """Device->host copy now (so the caller may overwrite vec), file write on a background thread."""
        if not self.enabled:
            return
        import threading
        cpu = {n: v.detach().to(dtype).to("cpu", non_blocking=True) for n, v in zip(names, vec)}
        torch.cuda.synchronize()
        path = os.path.join(self.raw, f"{kind}_{step:06d}.pt")
        if not hasattr(self, "_threads"):
            self._threads = []
        self._threads = [th for th in self._threads if th.is_alive()]
        while len(self._threads) >= 4:  # bound host memory
            self._threads.pop(0).join()
        th = threading.Thread(target=torch.save, args=(cpu, path), daemon=False)
        th.start()
        self._threads.append(th)

    def close(self):
        for th in getattr(self, "_threads", []):
            th.join()
