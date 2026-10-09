"""The hashed n-gram table for full-batch training: sharded rows, per-microbatch row pulls, fp32
gradient accumulation at the owners, one dense row-Adam step per optimizer step.

Same table, hashes, sign pool and update rule as track_1_short/ngram_table.py (Adam with beta1 = 0 and
one second moment per row, cautious decay gated on the update), but:
  - every optimizer step is an update event (no 2/4-step cadence);
  - the gradient of a row is the exact sum over every token of the step's batch, accumulated in fp32
    at the owner (the WR sums a cycle in fp16 and merges in bf16);
  - routes (which rows a microbatch needs from which owner) are computed per microbatch and may be cached
    across steps when the dataset is fixed.
Memory per rank: shard (bf16) + fp32 grad accumulator = 6 bytes x 84.6M x 768 / world  (48.7 GB at 8 ranks).
"""
import torch
import torch.distributed as dist
from torch import Tensor

from track_1_short.ngram_table import (
    NGRAM_ADAM_BETA2,
    NGRAM_DIM,
    NGRAM_VOCAB_SIZE,
    NGRAM_WD_MUL,
)
from track_1_short.perf.kernels.row_scatter import scatter_add_rows
from track_1_short.sharded_rows import RowPull, all_to_all_rows

BIGRAM_ROW_MULS = (36313, 27191)
TRIGRAM_ROW_MULS = (17351, 60961, 45259)


def ngram_row_ids_device(x: Tensor, num_rows: int = NGRAM_VOCAB_SIZE) -> Tensor:
    """track_1_short.ngram_table.ngram_row_ids on the device ([T] int32 -> [2T] int32); same int32 wrap.
    num_rows < the WR's table size is for testing the plumbing on fewer GPUs (different hashes)."""
    x = x.to(torch.int32)
    n = x.numel()
    HALF = num_rows // 2
    bigram_mod, trigram_mod = HALF - 1, num_rows - HALF - 1
    out = torch.empty(2 * n, dtype=torch.int32, device=x.device)
    bigram, trigram = out[:n], out[n:]
    bigram[0] = bigram_mod
    # The WR computes bigram[1:] in place from a copy of x whose element 0 was overwritten, so position 1
    # reads the reserved id as its previous token.
    prev = x.clone()
    prev[0] = bigram_mod
    bigram[1:] = torch.bitwise_xor(BIGRAM_ROW_MULS[0] * x[1:], BIGRAM_ROW_MULS[1] * prev[:-1]) % bigram_mod
    trigram[0] = trigram[1] = HALF + trigram_mod
    trigram[2:] = torch.bitwise_xor(
        torch.bitwise_xor(TRIGRAM_ROW_MULS[0] * x[2:], TRIGRAM_ROW_MULS[1] * x[1:-1]),
        TRIGRAM_ROW_MULS[2] * x[:-2],
    ) % trigram_mod + HALF
    return out


class DenseNgram:
    def __init__(self, cache: Tensor, rank: int, world_size: int, device, num_rows: int = NGRAM_VOCAB_SIZE):
        assert num_rows % world_size == 0 and num_rows % 2 == 0
        self.num_rows = num_rows
        self.cache, self.rank, self.world, self.device = cache, rank, world_size, device
        self.rows_per_rank = num_rows // world_size
        self.first_row = rank * self.rows_per_rank
        self.bounds = torch.arange(world_size + 1, dtype=torch.int64, device=device) * self.rows_per_rank
        self.shard = torch.zeros(self.rows_per_rank, NGRAM_DIM, dtype=torch.bfloat16, device=device)
        self.grad_acc = torch.zeros(self.rows_per_rank, NGRAM_DIM, dtype=torch.float32, device=device)
        self.exp_avg_sq = torch.zeros(self.rows_per_rank, dtype=torch.float32, device=device)
        self.touched = torch.zeros(self.rows_per_rank, dtype=torch.bool, device=device)
        self.events = 0
        self.route_cache: dict = {}

    # ---- routes ----
    def make_route(self, row_ids: Tensor) -> tuple[RowPull, Tensor]:
        """Collective. Returns (pull, slots) for one microbatch's [2T] row ids."""
        want = torch.unique(row_ids)  # sorted
        slots = torch.searchsorted(want, row_ids, out_int32=True)
        want = want.to(torch.int32)
        cuts = torch.searchsorted(want.to(torch.int64), self.bounds).cpu()
        send = (cuts[1:] - cuts[:-1]).clone()
        lo, hi = int(cuts[self.rank]), int(cuts[self.rank + 1])
        send[self.rank] = 0
        if self.world > 1:
            send_d = send.to(self.device)
            recv_d = torch.empty_like(send_d)
            dist.all_to_all_single(recv_d, send_d)
            recv = recv_d.cpu()
        else:
            recv = torch.zeros_like(send)
        pull = RowPull(want=want, lo=lo, hi=hi, send_counts=send.tolist(), recv_counts=recv.tolist())
        pull.peer_want = pull.peer_rows()
        pull.asked = (all_to_all_rows(pull.peer_want, pull.send_counts, pull.recv_counts) if self.world > 1
                      else want[:0])
        return pull, slots

    def route(self, key, row_ids: Tensor):
        if key is not None and key in self.route_cache:
            return self.route_cache[key]
        r = self.make_route(row_ids)
        if key is not None:
            self.route_cache[key] = r
        return r

    # ---- per microbatch ----
    @torch.no_grad()
    def land(self, pull: RowPull):
        """Fill the model's cache with the pull's rows, in want order."""
        lo, hi, n = pull.lo, pull.hi, pull.num_rows
        if self.world > 1:
            served = self.shard.index_select(0, (pull.asked - self.first_row).long())
            received = all_to_all_rows(served, pull.recv_counts, pull.send_counts)
            self.cache[:lo].copy_(received[:lo])
            self.cache[hi:n].copy_(received[lo:])
        torch.index_select(self.shard, 0, (pull.want[lo:hi] - self.first_row).long(), out=self.cache[lo:hi])

    @torch.no_grad()
    def accumulate(self, pull: RowPull, slots: Tensor, sink_grad: Tensor, mark_touched: bool = False):
        """Collective. Sum the microbatch's sink gradient per row and add it into the owners' fp32 accumulators."""
        per_slot = torch.zeros(pull.num_rows, NGRAM_DIM, dtype=torch.float32, device=self.device)
        scatter_add_rows(sink_grad, slots, per_slot)
        lo, hi = pull.lo, pull.hi
        own = (pull.want[lo:hi] - self.first_row).long()
        if own.numel():
            self.grad_acc.index_add_(0, own, per_slot[lo:hi])
        if mark_touched and own.numel():
            self.touched[own] = True
        if self.world > 1:
            wire = torch.cat([per_slot[:lo], per_slot[hi:]])
            received = all_to_all_rows(wire, pull.send_counts, pull.recv_counts)
            asked = (pull.asked - self.first_row).long()
            if asked.numel():
                self.grad_acc.index_add_(0, asked, received)
                if mark_touched:
                    self.touched[asked] = True

    # ---- optimizer ----
    @torch.no_grad()
    def adam_step(self, grad_scale: float, lr: float, weight_decay: float, eps: float,
                  beta2: float = NGRAM_ADAM_BETA2, wd_mul: float = NGRAM_WD_MUL, chunk_rows: int = 1 << 20):
        """One dense event over the shard: rows with no gradient have update 0 (beta1 = 0, decay gated on
        the update) and only their second moment decays, exactly as the WR's lazy replay."""
        self.events += 1
        step_size = lr * (1 - beta2 ** self.events) ** 0.5
        decay = lr * lr * weight_decay * wd_mul
        for lo in range(0, self.rows_per_rank, chunk_rows):
            hi = min(lo + chunk_rows, self.rows_per_rank)
            g = self.grad_acc[lo:hi].mul_(grad_scale)
            v = self.exp_avg_sq[lo:hi]
            v.mul_(beta2).add_(g.square().mean(1), alpha=1 - beta2)
            upd = g.div_((v.sqrt() + eps)[:, None]).mul_(step_size)
            p = self.shard[lo:hi]
            pf = p.float()
            upd.add_(torch.where(upd * pf > 0, pf * decay, 0.0))
            p.copy_(pf - upd)
        self.grad_acc.zero_()

    def zero_grad(self):
        self.grad_acc.zero_()
