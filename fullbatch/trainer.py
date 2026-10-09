"""FullBatchTrainer: the WR (record #360) model and ANVIL+Adam optimizer, driven by exact gradients
accumulated over many microbatches per optimizer step.

Public API (what instrumentation builds on); all methods are collectives (every rank calls them):
  t = FullBatchTrainer(cfg)                       build model, optimizer, data
  t.accumulate(mb_ids=None, on_microbatch=None)   zero + accumulate the summed gradient over this rank's
                                                  microbatches (all of them by default) into fp32
                                                  accumulators t.acc[param] (and the n-gram owners'
                                                  accumulator); returns (global mean loss, global tokens).
                                                  on_microbatch(info) is called after every microbatch
                                                  with info = dict(index, loss_sum, tokens, grads) where
                                                  grads maps param -> this microbatch's (bf16/fp32) grad.
  t.mean_grads()                                  fp32 all-reduced MEAN-loss gradient per parameter
                                                  (dict label -> tensor), from the current accumulators.
  t.optimizer_step(step)                          one ANVIL/Adam (+ n-gram) step from the accumulators
  t.train_step(step)                              accumulate + optimizer_step (+ logging fields)
  t.loss_on(mb_ids)                               mean training loss (fp8 training kernels) on microbatches
  t.evaluate()                                    validation losses (bf16 eval path)
  t.params()                                      list of (label, param) of the trainable parameters
  t.get_flat() / t.set_flat(vec)                  flat fp32 copy of / overwrite the bf16/fp32 params
                                                  (set_flat refreshes the fp8 weight caches)
  t.refresh_fp8()                                 after any manual parameter change
  t.state_dict() / t.load_state_dict(sd)          model + optimizer (+ n-gram shard per rank)
Microbatch ids are indices into this rank's list (t.data.global_ids gives their global ids).
"""
import copy
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch import nn

from track_1_short.model.gpt import GPT, ForwardScheduleConfig, NUM_LAYERS
from track_1_short.ngram_table import NGRAM_DIM
from track_1_short.optim.anvil import (
    RAIL_FAST_BETA,
    RAIL_FAST_WEIGHT,
    AnvilAndAdam,
    anvil_bank_update,
)
from track_1_short.perf.kernels.mlp import prime_stage_cache
from track_1_short.training import VALUE_EMBED_BETAS, VALUE_EMBED_LR_MUL, VALUE_EMBED_WD_MUL

from fullbatch.data import FixedDataset, StreamBatches, val_chunks
from fullbatch.ngram_dense import DenseNgram, ngram_row_ids_device

BLOCK = 128


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
    dist.barrier()
    return rank, world, device


def param_table():
    """record #360's TrainingManager.param_table and orders (value_embeds keeps its own betas/mults)."""
    t = {
        "qk_bank":        {"optim": "anvil", "comms": "sharded", "adam_betas": None},
        "vo_bank":        {"optim": "anvil", "comms": "sharded", "adam_betas": None},
        "mlp_bank":       {"optim": "anvil", "comms": "sharded", "adam_betas": None},
        "scalars":        {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 5.0, "wd_mul": 0.0},
        "smear_gate":     {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.01, "wd_mul": 0.0},
        "ve_gate_bank":   {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99]},
        "lm_head":        {"optim": "adam", "comms": "sharded", "adam_betas": [0.5, 0.95], "wd_mul": 150.},
        "post_lambdas":   {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 1.0, "wd_mul": 0.0},
        "resid_lambdas":  {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.95], "lr_mul": 5.0, "wd_mul": 0.0},
        "value_embeds":   {"optim": "adam", "comms": "sharded", "adam_betas": list(VALUE_EMBED_BETAS), "lr_mul": VALUE_EMBED_LR_MUL, "wd_mul": VALUE_EMBED_WD_MUL},
        "embed":          {"optim": "adam", "comms": "sharded", "adam_betas": [0.5, 0.95], "wd_mul": 150.},
        "mudd_w1":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
        "mudd_w2":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
        "mudd_w2g":   {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
        "mudd_b2":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25, "wd_mul": 0.0},
        "mudd_gate_w1": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1},
        "mudd_gate_w2": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1},
        "mudd_gate_b2": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1, "wd_mul": 0.0},
        "_mudd_gate_scale": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1, "wd_mul": 0.0},
    }
    scatter_order = ["qk_bank", "vo_bank", "mlp_bank", "value_embeds", "lm_head", "embed",
                     "scalars", "smear_gate", "ve_gate_bank", "post_lambdas", "resid_lambdas",
                     "mudd_w1", "mudd_w2", "mudd_w2g", "mudd_b2", "mudd_gate_w1", "mudd_gate_w2", "mudd_gate_b2",
                     "_mudd_gate_scale"]
    work_order = ["qk_bank", "vo_bank", "lm_head", "embed", "value_embeds",
                  "scalars", "smear_gate", "ve_gate_bank", "mudd_b2", "mudd_gate_b2", "_mudd_gate_scale",
                  "post_lambdas", "resid_lambdas", "mudd_w2", "mudd_w2g", "mudd_gate_w2", "mudd_w1", "mudd_gate_w1",
                  "mlp_bank"]
    return t, scatter_order, work_order


class FullBatchTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        _t0 = time.time(); self.init_times = {}
        self.partial_resume = None   # set by train_fb.py to continue a step from a partial checkpoint
        self.partial_hook = None     # called after every microbatch of a training step: hook(n_done, loss_sum, tokens)
        self.rank, self.world, self.device = setup_distributed()
        self.init_times['dist'] = round(time.time() - _t0, 1)
        self.master = self.rank == 0
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

        # ---------------- data ----------------
        T = cfg.micro_tokens
        # The fused CE kernel indexes [rows, vocab] with 32-bit unsigned arithmetic (blockIdx.x * VOCAB).
        assert T * ((cfg.vocab_size + 127) // 128 * 128) < 2 ** 32, "micro_tokens too large for the CE kernel"
        if cfg.mode == "full" or cfg.minibatch_source == "fixed":
            self.data = FixedDataset(cfg, self.rank, self.world)
            cs = torch.tensor([self.data.local_checksum], dtype=torch.int64, device=self.device)
            dist.all_reduce(cs)
            self.dataset_checksum = int(cs.item())
            self.dataset_tokens = self.data.num_tokens
            self.init_times['data'] = round(time.time() - _t0, 1)
        else:
            self.data = None
            self.dataset_checksum = None
            self.dataset_tokens = None
        if cfg.mode == "minibatch":
            assert cfg.batch_tokens % (T * self.world) == 0, "batch_tokens must be a multiple of micro_tokens*world"
            self.K = cfg.batch_tokens // T
            if cfg.minibatch_source == "stream":
                self.stream = StreamBatches(cfg, self.rank, self.world, self.K)
            else:
                assert self.K <= self.data.M
                self.epoch_perm, self.perm_pos = None, 0
        self.val, self.num_val_chunks = val_chunks(cfg, self.rank, self.world)
        self.init_times['val'] = round(time.time() - _t0, 1)

        # ---------------- model ----------------
        model = GPT(vocab_size=cfg.vocab_size, num_layers=NUM_LAYERS, num_heads=6, head_dim=128, model_dim=768,
                    max_seq_len=cfg.val_chunk_tokens, ngram_dim=NGRAM_DIM, world_size=self.world,
                    device=self.device).cuda()
        model.cast_matrix_weights_bf16()
        for p in model.parameters():
            dist.broadcast(p.detach(), 0)
        dist.broadcast(model.ngram_sign_pool, 0)
        self.init_times['model'] = round(time.time() - _t0, 1)
        self.raw_model = model
        # Fixed rotary state: replay the WR's YaRN window chain once.
        sc, lc = cfg.chain(cfg.yarn_short_chain), cfg.chain(cfg.yarn_long_chain)
        assert sc[-1] == cfg.ws_short and lc[-1] == cfg.ws_long
        for a, b in zip(sc, sc[1:]):
            for y in (model.yarn, model.yarn_paired_head):
                y.apply(a * BLOCK, b * BLOCK)
        for a, b in zip(lc, lc[1:]):
            model.yarn_wide.apply(a * BLOCK, b * BLOCK)
        model.quantize_attn_fp8()
        model.quantize_mlp_fp8(refresh_lm=True)
        self.init_times['quant'] = round(time.time() - _t0, 1)
        prime_stage_cache()
        self.init_times['prime'] = round(time.time() - _t0, 1)
        if cfg.canon_mask and cfg.tokenizer == "gpt2":
            self._build_canon_mask()
            self.init_times['canon'] = round(time.time() - _t0, 1)
        self.model = torch.compile(model, dynamic=False, fullgraph=True) if cfg.compile else model

        self.fwd_cfg = ForwardScheduleConfig(
            mtp_weights=torch.tensor(cfg.mtp, device=self.device),
            prefix_weight=torch.tensor([cfg.prefix_weight], device=self.device),
            ws_short=cfg.ws_short * BLOCK, ws_long=cfg.ws_long * BLOCK,
            train_max_seq_len=min(cfg.max_seq_len, cfg.attn_cap), sampled_loss=None)

        # ---------------- optimizer ----------------
        table, so, wo = param_table()
        self.optimizer = AnvilAndAdam(
            model.named_parameters(), param_table=table, scatter_order=so, work_order=wo,
            adam_defaults=dict(lr=cfg.adam_lr, eps=1e-10, weight_decay=cfg.adam_wd),
            anvil_defaults=dict(lr=cfg.anvil_lr, momentum=0.95, beta2=0.9, weight_decay=cfg.anvil_wd),
            bank_update=anvil_bank_update)
        self.optimizer.reset()
        self.init_times['optim'] = round(time.time() - _t0, 1)
        if not cfg.tied_embed:
            self.optimizer.split_embed = True
        # Parameters whose gradient flows through autograd (value_embeds' goes to its own buffer).
        self.named = [(p.label, p) for p in model.parameters()]
        self.ve = model.value_embeds
        self.ve_buf = torch.zeros(self.ve.shape, dtype=torch.float16, device=self.device)
        self.acc = {p: torch.zeros(p.shape, dtype=torch.float32, device=self.device) for _, p in self.named}

        # ---------------- n-gram ----------------
        self.ngram = DenseNgram(model.ngram_cache, self.rank, self.world, self.device, cfg.ngram_rows) if cfg.ngram else None
        self.zero_slots = {}
        self.step_count = 0
        self.last_tokens = 0
        self.mb_hook = None        # extra per-microbatch callback (same info as on_microbatch), e.g. token losses
        self._last_tok_loss = None

    # ------------------------------------------------------------------ helpers
    def _build_canon_mask(self):
        from track_1_short.canonical_mask import build_canonical_mask
        import tiktoken
        V = self.raw_model.vocab_size
        if self.master:
            # Building the mask takes minutes on the CPU; it depends only on the tokenizer, so cache it.
            path = os.path.join(os.environ.get("NFB_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")), f"canon_mask_gpt2_V{V}.pt")
            if os.path.exists(path):
                mask = torch.load(path)
            else:
                mask = torch.from_numpy(build_canonical_mask(V, tiktoken.get_encoding("gpt2")._mergeable_ranks))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(mask, path + f".tmp{os.getpid()}")
                os.replace(path + f".tmp{os.getpid()}", path)
            self.raw_model.canon_mask.copy_(mask)
        dist.broadcast(self.raw_model.canon_mask, 0)
        self.canon = self.raw_model.canon_mask.clone()

    def params(self):
        return list(self.named)

    def _slots_off(self, n):
        if n not in self.zero_slots:
            self.zero_slots[n] = torch.zeros(2 * n, dtype=torch.int32, device=self.device)
        return self.zero_slots[n]

    def _step_microbatches(self, step):
        """This rank's microbatches for an optimizer step: list of (key, inputs, targets, cu) on device."""
        cfg = self.cfg
        if cfg.mode == "full":
            return ((("d", i), *self.data.get(i, self.device)) for i in range(len(self.data)))
        if cfg.minibatch_source == "stream":
            inputs, targets, cu = self.stream.next_step()
            return [(None, inputs[i].to(self.device, non_blocking=True), targets[i].to(self.device, non_blocking=True).long(),
                     cu[i].to(self.device, non_blocking=True)) for i in range(inputs.shape[0])]
        # Fixed-set minibatches: shuffled epochs over global microbatch ids, identical on every rank.
        # Rank r owns the global ids g with g % world == r, and each step takes K/world from each residue.
        gids = self._next_fixed_ids()
        mine = {g: i for i, g in enumerate(self.data.global_ids)}
        local = [mine[g] for g in gids if g % self.world == self.rank]
        return [(("d", i), *self.data.get(i, self.device)) for i in local]

    def _next_fixed_ids(self):
        """K global ids, balanced over owners: K/world from each residue class, in shuffled epoch order."""
        M, W, per = self.data.M, self.world, self.K // self.world
        if self.epoch_perm is None or self.perm_pos + per > M // W:
            g = torch.Generator().manual_seed(self.cfg.seed * 1000003 + self.step_count)
            # one permutation per residue class (same on every rank)
            self.epoch_perm = [torch.randperm(M // W, generator=g) * W + r for r in range(W)]
            self.perm_pos = 0
        ids = []
        for r in range(W):
            ids.extend(self.epoch_perm[r][self.perm_pos:self.perm_pos + per].tolist())
        self.perm_pos += per
        return ids

    # ------------------------------------------------------------------ gradient
    def zero_acc(self):
        torch._foreach_zero_(list(self.acc.values()))
        if self.ngram is not None:
            self.ngram.zero_grad()

    def _fwd_bwd(self, key, inputs, targets, cu, want_grads=True):
        """One microbatch: returns (loss_sum, grads dict param->grad) and accumulates n-gram/value_embeds grads."""
        T = inputs.numel()
        sink = None
        if self.ngram is not None:
            rid = ngram_row_ids_device(inputs, self.ngram.num_rows)
            pull, slots = self.ngram.route(key, rid)
            self.ngram.land(pull)
            if want_grads:
                sink = self._sink(T)
        else:
            slots = self._slots_off(T)
        if not want_grads:
            with torch.no_grad():
                loss = self.model(inputs, targets, cu, slots, self.fwd_cfg).sum()
            return loss.detach(), None
        tok_loss = self.model(inputs, targets, cu, slots, self.fwd_cfg, ngram_sink=sink, value_embed_grad=self.ve_buf)
        self._last_tok_loss = tok_loss.detach()
        loss = tok_loss.sum()
        leaves = [p for _, p in self.named if p is not self.ve] + ([sink] if sink is not None else [])
        grads = torch.autograd.grad(loss, leaves, allow_unused=True)
        if sink is not None:
            self.ngram.accumulate(pull, slots, grads[-1], mark_touched=(self.step_count == 0))
            grads = grads[:-1]
        gd = {p: g for p, g in zip([p for _, p in self.named if p is not self.ve], grads) if g is not None}
        # value_embeds: flush its fp16 per-microbatch buffer into the fp32 accumulator
        gd[self.ve] = self.ve_buf
        return loss.detach(), gd

    def _sink(self, T):
        if not hasattr(self, "_sinks"):
            self._sinks = {}
        if T not in self._sinks:
            self._sinks[T] = torch.zeros(2 * T, NGRAM_DIM, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        return self._sinks[T]

    def accumulate(self, mbs=None, on_microbatch=None, step=None):
        """Zero and accumulate the token-summed gradient over microbatches (default: this step's)."""
        self.model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        tokens, start_idx = 0, 0
        # Mid-step resume (full mode, this step's own microbatches only): continue a partially accumulated
        # gradient from a partial checkpoint (see train_fb.py save_partial / load_partial).
        pr = self.partial_resume if mbs is None else None
        self.partial_resume = None
        if pr is not None:
            assert self.cfg.mode == "full" and pr["step"] == (step if step is not None else self.step_count)
            for p_, v in zip(self.acc.keys(), pr["acc"]):
                self.acc[p_].copy_(v)
            loss_sum.fill_(pr["loss_sum"])
            tokens, start_idx = pr["tokens"], pr["idx"]
            if pr.get("on_restore") is not None:
                pr["on_restore"]()   # buffers / rng / per-microbatch stats, after any per-step resets
        else:
            self.zero_acc()
        hook = self.partial_hook if mbs is None else None
        # mb_hook (e.g. per-token loss saving) only sees the training step's own microbatches, not probe passes
        mb_hook = self.mb_hook if mbs is None else None
        if mbs is None:
            if self.cfg.mode == "full":
                mbs = ((("d", i), *self.data.get(i, self.device)) for i in range(start_idx, len(self.data)))
            else:
                mbs = self._step_microbatches(step if step is not None else self.step_count)
        for idx, (key, inputs, targets, cu) in enumerate(mbs, start=start_idx):
            loss, gd = self._fwd_bwd(key, inputs, targets, cu)
            if on_microbatch is not None or mb_hook is not None:
                info = dict(index=idx, key=key, loss_sum=loss, tokens=inputs.numel(), grads=gd,
                            token_losses=self._last_tok_loss)
                if on_microbatch is not None:
                    on_microbatch(info)
                if mb_hook is not None:
                    mb_hook(info)
            keys = list(gd.keys())
            torch._foreach_add_([self.acc[p] for p in keys], [gd[p] for p in keys])
            self.ve_buf.zero_()
            loss_sum += loss.double()
            tokens += inputs.numel()
            if hook is not None:
                hook(idx + 1, loss_sum, tokens)
        stats = torch.stack([loss_sum, torch.tensor(float(tokens), dtype=torch.float64, device=self.device)])
        dist.all_reduce(stats)
        self.last_tokens = int(stats[1].item())
        return (stats[0] / stats[1]).item(), self.last_tokens

    def fixed_mbs(self, local_ids):
        """Microbatch list for loss_on/accumulate from local dataset indices (fixed dataset only)."""
        return [(("d", i), *self.data.get(i, self.device)) for i in local_ids]

    @torch.no_grad()
    def loss_on(self, mbs):
        self.model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        tokens = 0
        for key, inputs, targets, cu in mbs:
            loss, _ = self._fwd_bwd(key, inputs, targets, cu, want_grads=False)
            loss_sum += loss.double()
            tokens += inputs.numel()
        stats = torch.stack([loss_sum, torch.tensor(float(tokens), dtype=torch.float64, device=self.device)])
        dist.all_reduce(stats)
        return (stats[0] / stats[1]).item()

    @torch.no_grad()
    def eval_loss_on(self, mbs):
        """Mean loss of the bf16 eval path (no fp8, no canonical mask) on training microbatches."""
        self.model.eval()
        self.raw_model.canon_mask.zero_()
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        tokens = 0
        for key, inputs, targets, cu in mbs:
            if self.ngram is not None:
                pull, slots = self.ngram.route(key, ngram_row_ids_device(inputs, self.ngram.num_rows))
                self.ngram.land(pull)
            else:
                slots = self._slots_off(inputs.numel())
            loss_sum += self.model(inputs, targets, cu, slots, self.fwd_cfg).double().sum()
            tokens += inputs.numel()
        stats = torch.stack([loss_sum, torch.tensor(float(tokens), dtype=torch.float64, device=self.device)])
        dist.all_reduce(stats)
        self.model.train()
        return (stats[0] / stats[1]).item()

    @torch.no_grad()
    def mean_grads(self):
        """fp32 mean-token-loss gradient per label, all-reduced (copies; accumulators untouched).
        Tied embeddings: 'lm_head' includes embed's gradient (transposed) and 'embed' is dropped."""
        out = {}
        for label, p in self.named:
            g = self.acc[p].clone()
            dist.all_reduce(g)
            out[label] = g.div_(self.last_tokens)
        if self.cfg.tied_embed and "embed" in out:
            out["lm_head"] += out.pop("embed").T
        return out

    # ------------------------------------------------------------------ optimizer
    def lr_multiplier(self, step):
        c, n = self.cfg, self.cfg.steps
        lr = c.lr_mul
        w = int(c.warmup_frac * n)
        if w > 0 and step < w:
            lr = lr * (step + 1) / w
        cd_start = int(n * (1 - c.cooldown_frac))
        if step >= cd_start and c.cooldown_frac > 0:
            t = min(1.0, (step - cd_start) / max(1, n - cd_start))
            lr = lr * (1 - t) + c.lr_floor * t
        return lr

    def rail_beta(self, step, beta_min=0.85, beta_max=0.93):
        n = self.cfg.steps
        wu = max(1, round(self.cfg.rail_warmup_frac * n))
        cd = max(1, round(self.cfg.rail_cooldown_frac * n))
        if step < wu:
            return beta_min + step / wu * (beta_max - beta_min)
        if step > n - cd:
            return beta_max - (step - (n - cd)) / cd * (beta_max - beta_min)
        return beta_max

    @torch.no_grad()
    def optimizer_step(self, step):
        cfg = self.cfg
        scale = cfg.loss_scale * self.world / self.last_tokens  # optimizer AVG-reduces over ranks
        for label, p in self.named:
            p.grad = self.acc[p].mul(scale).to(p.dtype)
        s = self.lr_multiplier(step)
        rb = self.rail_beta(step)
        if step >= round(cfg.rail_engage_frac * cfg.steps):
            self.optimizer.set_rails(fast_beta=RAIL_FAST_BETA, fast_weight=RAIL_FAST_WEIGHT)
        else:
            self.optimizer.set_rails(fast_beta=rb, fast_weight=1.0)
        for p, pc in self.optimizer.param_cfgs.items():
            pc.lr = pc.initial_lr * s
            if pc.optim == "anvil":
                pc.momentum = rb
        self.optimizer.step(do_adam=True, sparse_update=None, deferred_labels=frozenset())
        if self.ngram is not None:
            self.ngram.adam_step(grad_scale=cfg.loss_scale / self.last_tokens, lr=cfg.adam_lr * s * cfg.ngram_lr_mul,
                                 weight_decay=cfg.adam_wd, eps=1e-10)
        self.refresh_fp8()
        self.step_count = step + 1

    def refresh_fp8(self):
        self.raw_model.quantize_attn_fp8()
        self.raw_model.quantize_mlp_fp8(refresh_lm=True)

    def train_step(self, step, on_microbatch=None):
        t0 = time.perf_counter()
        loss, tokens = self.accumulate(on_microbatch=on_microbatch, step=step)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.optimizer_step(step)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return dict(step=step, train_loss=loss, tokens=tokens, t_grad=t1 - t0, t_opt=t2 - t1,
                    tok_per_s_per_gpu=tokens / (t1 - t0) / self.world, lr_mul=self.lr_multiplier(step))

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self):
        """Val loss on the WR val set (10.5M tokens), bf16 eval path; plain and (if built) canonical-masked."""
        m = self.raw_model
        self.model.eval()
        out = {}
        variants = [("val_loss", False)] + ([("val_loss_canon", True)] if hasattr(self, "canon") else [])
        for name, canon in variants:
            m.canon_mask.copy_(self.canon) if canon else m.canon_mask.zero_()
            total = torch.zeros((), dtype=torch.float64, device=self.device)
            for k, (inputs, targets, cu) in enumerate(self.val):
                inputs, targets, cu = inputs.to(self.device), targets.to(self.device), cu.to(self.device)
                if self.ngram is not None:
                    pull, slots = self.ngram.route(("v", k), ngram_row_ids_device(inputs, self.ngram.num_rows))
                    self.ngram.land(pull)
                else:
                    slots = self._slots_off(inputs.numel())
                total += self.model(inputs, targets, cu, slots, self.fwd_cfg).double().mean()
            dist.all_reduce(total)
            out[name] = (total / self.num_val_chunks).item()
        m.canon_mask.zero_()
        self.model.train()
        return out

    # ------------------------------------------------------------------ params
    @torch.no_grad()
    def get_flat(self):
        return torch.cat([p.detach().float().reshape(-1) for _, p in self.named])

    @torch.no_grad()
    def set_flat(self, vec):
        o = 0
        for _, p in self.named:
            n = p.numel()
            p.copy_(vec[o:o + n].view_as(p))
            o += n
        if self.cfg.tied_embed:
            self.raw_model.embed.weight.copy_(self.raw_model.lm_head.weight.T)
        self.refresh_fp8()

    # ------------------------------------------------------------------ checkpoint / resume (per rank)
    _SKIP_BUFFERS = ("ngram_cache", "canon_mask", "prefix_table")

    def capture_state(self):
        """Everything this rank needs to continue bit-exactly from the current step boundary: params,
        every model buffer (fp8 caches and delayed scales/amaxes, rotary tables), the fp8 call counter,
        this rank's optimizer state (sharded ANVIL rails/lanes/mantissas, Adam moments and step counts),
        data position and RNG. CPU copies."""
        assert self.ngram is None, "checkpointing with the n-gram table is not supported"
        m = self.raw_model
        cpu = lambda x: x.detach().to("cpu", copy=True)
        params = {n: cpu(p) for n, p in m.named_parameters()}
        buffers = {n: cpu(b) for n, b in m.named_buffers() if n.split(".")[-1] not in self._SKIP_BUFFERS}
        opt = {}
        for p, st in self.optimizer.param_states.items():
            label = self.optimizer.param_cfgs[p].label
            opt[label] = {k: (cpu(v) if torch.is_tensor(v) else v) for k, v in st.items()}
        data = {}
        if self.cfg.mode == "minibatch" and self.cfg.minibatch_source == "fixed":
            data = dict(epoch_perm=self.epoch_perm, perm_pos=self.perm_pos)
        return dict(params=params, buffers=buffers, mlp_quantize_calls=m.mlp_quantize_calls, optimizer=opt,
                    split_embed=self.optimizer.split_embed, step_count=self.step_count, data=data,
                    rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(self.device),
                    world=self.world, rank=self.rank)

    @torch.no_grad()
    def restore_state(self, st):
        assert st["world"] == self.world and st["rank"] == self.rank, "resume needs the same world size"
        m = self.raw_model
        named_p = dict(m.named_parameters())
        for n, v in st["params"].items():
            named_p[n].copy_(v)
        named_b = dict(m.named_buffers())
        for n, v in st["buffers"].items():
            named_b[n].copy_(v)
        m.mlp_quantize_calls = st["mlp_quantize_calls"]
        by_label = {cfg.label: p for p, cfg in self.optimizer.param_cfgs.items()}
        for label, sd in st["optimizer"].items():
            live = self.optimizer.param_states[by_label[label]]
            for k, v in sd.items():
                if torch.is_tensor(v):
                    live[k].copy_(v)   # in place: banks and flat replicated buffers alias these
                else:
                    live[k] = v
        self.optimizer.split_embed = st["split_embed"]
        self.step_count = st["step_count"]
        if st["data"]:
            self.epoch_perm, self.perm_pos = st["data"]["epoch_perm"], st["data"]["perm_pos"]
        if self.cfg.mode == "minibatch" and self.cfg.minibatch_source == "stream":
            self.stream.skip_steps(self.step_count)
        torch.set_rng_state(st["rng_cpu"])
        torch.cuda.set_rng_state(st["rng_cuda"], self.device)

    def state_dict(self):
        sd = dict(model={k: v.detach().cpu() for k, v in self.raw_model.state_dict().items()},
                  optimizer=self.optimizer.state_dict(), step=self.step_count, cfg=self.cfg.__dict__)
        return sd
