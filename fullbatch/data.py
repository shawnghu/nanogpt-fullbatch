"""Deterministic microbatch packing, world-size invariant.

The token stream (FineWeb .bin shards) is packed into a single global sequence of microbatches of
exactly micro_tokens tokens, with the WR's rule (track_1_short/data.py Shard.next_batch at
world_size=1): every microbatch starts at a BOS, documents are cut to max_seq_len, the last document
is cut to fill the microbatch, and the rest of a cut document is skipped. Attention segments are
further cut at attn_cap. Microbatch k of the global sequence goes to rank k % world_size, so the set
of microbatches (the "dataset") does not depend on the number of GPUs.
"""
import glob
import os
import itertools
import threading
import queue
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from track_1_short.data import cu_seqlens_rows

HEADER_INTS = 256


def load_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    assert header[0] == 20240520 and header[1] == 1, f"bad header in {path}"
    n = int(header[2])
    tokens = np.fromfile(path, dtype=np.uint16, count=n, offset=HEADER_INTS * 4)
    assert tokens.size == n
    return tokens


def cu_table_rows(T: int) -> int:
    """Rows of the padded cu_seqlens table. The WR's table (128 rows at 49152 tokens) overflows on stretches
    of very short documents (seen in fineweb100B: 139 segments); padding rows are T, so more rows change
    nothing numerically."""
    return max(cu_seqlens_rows(T), T // 128)


@dataclass
class Microbatch:
    inputs: np.ndarray    # [T] int32
    targets: np.ndarray   # [T] int32
    cu: np.ndarray        # [R] int32 cumulative attention-segment ends, padded with T


def _segment_ends(lengths: np.ndarray, cap: int) -> list[int]:
    cum = np.cumsum(lengths).tolist()
    ends = []
    for start, stop in pairwise([0, *cum]):
        ends.extend(range(start + cap, stop, cap))
        ends.append(stop)
    return ends


def pack_stream(files: list[Path], T: int, max_seq_len: int, cap: int, bos_id: int, cycle: bool, keep=None):
    """Yield Microbatch objects forever (cycle) or until the files run out. keep(k) -> bool decides, for
    the k-th microbatch of the stream, whether to materialize it (others are yielded as None)."""
    R = cu_table_rows(T)
    file_iter = itertools.cycle(files) if cycle else iter(files)
    counter = 0
    for path in file_iter:
        tokens = load_shard(path)
        size = tokens.size
        bos = np.flatnonzero(tokens == bos_id)
        n = bos.size
        i = 0
        while True:
            starts, ends = [], []
            cur_len, idx, ok = 0, i, True
            while cur_len <= T:
                if idx >= n:
                    ok = False
                    break
                cur = int(bos[idx])
                starts.append(cur)
                idx += 1
                end = min(int(bos[idx]) if idx < n else size, cur + max_seq_len, cur + T - cur_len + 1)
                ends.append(end)
                cur_len += end - cur
            if not ok:
                break  # shard exhausted: the partial microbatch is dropped, as in the WR loader
            assert cur_len == T + 1
            i = idx
            k_stream = counter
            counter += 1
            if keep is not None and not keep(k_stream):
                yield None
                continue
            buf = np.concatenate([tokens[s:e] for s, e in zip(starts, ends)]).astype(np.int32)
            lengths = np.array(ends) - np.array(starts)
            lengths[-1] -= 1
            seg = _segment_ends(lengths, cap) if max_seq_len > cap else np.cumsum(lengths).tolist()
            assert len(seg) < R, f"{len(seg)} segments overflow the {R}-row cu_seqlens table"
            cu = np.full(R, T, dtype=np.int32)
            cu[0] = 0
            cu[1:len(seg) + 1] = seg
            yield Microbatch(inputs=buf[:-1].copy(), targets=buf[1:].copy(), cu=cu)


CACHE_ROOT = Path(os.environ.get("NFB_CACHE_DIR", Path(__file__).resolve().parent.parent / "cache")) / "packed"
LOCAL_CACHE_ROOT = Path(os.environ.get("NFB_LOCAL_CACHE_DIR", "/tmp/nfb-cache")) / "packed"


def _packed_cache(cfg, M: int, skip: int, files: list[Path]):
    """The first M microbatches (after `skip`) of the packed stream, as memory-mapped arrays
    buf [M, T+1] uint16 (inputs = buf[:, :-1], targets = buf[:, 1:]) and cu [M, R] int32. Built once by
    whichever process gets the lock (packing in Python takes ~10 min at 800M tokens) and reused by every
    later process, at any world size. World-size independent, so a cache hit is bit-identical to packing."""
    import fcntl, hashlib, json, os
    T = cfg.micro_tokens
    key = json.dumps(dict(files=[str(f) for f in files], T=T, max_seq_len=cfg.max_seq_len, cap=cfg.attn_cap,
                          bos=cfg.bos_id, M=M, skip=skip, v=1), sort_keys=True)
    d = CACHE_ROOT / hashlib.sha1(key.encode()).hexdigest()[:16]
    d.mkdir(parents=True, exist_ok=True)
    done = d / "DONE"
    if not done.exists():
        with open(d / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not done.exists():
                R = cu_table_rows(T)
                buf = np.lib.format.open_memmap(d / "buf.tmp.npy", mode="w+", dtype=np.uint16, shape=(M, T + 1))
                cu = np.lib.format.open_memmap(d / "cu.tmp.npy", mode="w+", dtype=np.int32, shape=(M, R))
                gen = pack_stream(files, T, cfg.max_seq_len, cfg.attn_cap, cfg.bos_id, cycle=False,
                                  keep=lambda k: k >= skip)
                for _ in range(skip):
                    next(gen)
                for k in range(M):
                    mb = next(gen)
                    buf[k, :T] = mb.inputs
                    buf[k, T] = mb.targets[-1]
                    cu[k] = mb.cu
                buf.flush(); cu.flush(); del buf, cu
                os.replace(d / "buf.tmp.npy", d / "buf.npy")
                os.replace(d / "cu.tmp.npy", d / "cu.npy")
                (d / "key.json").write_text(key)
                done.write_text("ok")
    # Strided per-rank reads of a memmap on network storage are slow on a cold node: stage one sequential
    # copy per node on local disk (flock-guarded) and memory-map that.
    local = LOCAL_CACHE_ROOT / d.name
    local.mkdir(parents=True, exist_ok=True)
    if not (local / "DONE").exists():
        import shutil
        with open(local / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (local / "DONE").exists():
                for name in ("buf.npy", "cu.npy"):
                    shutil.copyfile(d / name, local / (name + ".tmp"))
                    os.replace(local / (name + ".tmp"), local / name)
                (local / "DONE").write_text("ok")
    return np.load(local / "buf.npy", mmap_mode="r"), np.load(local / "cu.npy", mmap_mode="r")


class FixedDataset:
    """The first M microbatches (after skipping some) of the packed stream; this rank keeps its own."""

    def __init__(self, cfg, rank: int, world_size: int):
        files = sorted(Path(f) for f in glob.glob(cfg.train_glob))
        assert files, cfg.train_glob
        T = cfg.micro_tokens
        # A multiple of 8 microbatches, so the dataset is the same set at world size 1, 2, 4 or 8.
        M = cfg.dataset_tokens // (T * 8) * 8
        assert M > 0 and M % world_size == 0
        self.M, self.T = M, T
        self.num_tokens = M * T
        skip = cfg.dataset_skip_microbatches
        import sys, time as _time
        _t = _time.time(); tl = {}
        buf, cu = _packed_cache(cfg, M, skip, files)
        tl["cache"] = _time.time() - _t
        # Convert straight into one pinned buffer: on busy nodes every extra 800 MB host copy cost tens of
        # seconds (fresh-page faults under memory-cgroup pressure), and the old path made five per rank.
        n = M // world_size
        pinned = torch.empty((n, T + 1), dtype=torch.int32, pin_memory=True)
        pinned.numpy()[:] = buf[rank::world_size]
        tl["slice"] = _time.time() - _t
        self.global_ids = list(range(rank, M, world_size))
        self.inputs = pinned[:, :-1]     # row views are contiguous: rows are copied to the GPU one at a time
        self.targets = pinned[:, 1:]
        self.cu = torch.from_numpy(np.ascontiguousarray(cu[rank::world_size])).pin_memory()
        tl["pin"] = _time.time() - _t
        # A fingerprint of the whole dataset (identical on every rank and world size once all-reduced).
        # sum_i (gid_i + 1) * sum(inputs_i), in exact integer arithmetic, as one vectorized op (the per-row
        # loop it replaces took ~8 min on a loaded node).
        row_sums = self.inputs.sum(dim=1, dtype=torch.int64)
        self.local_checksum = int((row_sums * (torch.tensor(self.global_ids, dtype=torch.int64) + 1)).sum().item())
        tl["checksum"] = _time.time() - _t
        print(f"[data rank {rank}] " + " ".join(f"{k}={v:.1f}s" for k, v in tl.items()), file=sys.stderr, flush=True)

    def __len__(self):
        return len(self.global_ids)

    def get(self, i: int, device):
        return (self.inputs[i].to(device, non_blocking=True),
                self.targets[i].to(device, non_blocking=True).long(),
                self.cu[i].to(device, non_blocking=True))


class StreamBatches:
    """Minibatch mode from the stream: each step takes the next K global microbatches; rank r keeps
    k % world == r. A background thread packs ahead."""

    def __init__(self, cfg, rank: int, world_size: int, K: int):
        files = sorted(Path(f) for f in glob.glob(cfg.train_glob))
        self.rank, self.world, self.K = rank, world_size, K
        self.skip = 0  # microbatches to discard first (resume); set before the first next_step()
        # step s takes stream microbatches [s*K, (s+1)*K); within a step, index k goes to rank k % world
        self.gen = pack_stream(files, cfg.micro_tokens, cfg.max_seq_len, cfg.attn_cap, cfg.bos_id, cycle=True,
                               keep=lambda k: k >= self.skip and (k % K) % world_size == rank)
        self.q = queue.Queue(maxsize=4)
        self._started = False

    def skip_steps(self, steps: int):
        assert not self._started
        self.skip = steps * self.K

    def _run(self):
        try:
            self._loop()
        except BaseException as e:  # surface loader failures in next_step() instead of hanging
            self.q.put(e)

    def _loop(self):
        for _ in range(self.skip):
            next(self.gen)
        while True:
            step_mbs = []
            for k in range(self.K):
                mb = next(self.gen)
                if k % self.world == self.rank:
                    step_mbs.append(mb)
            inputs = torch.from_numpy(np.stack([m.inputs for m in step_mbs])).pin_memory()
            targets = torch.from_numpy(np.stack([m.targets for m in step_mbs])).pin_memory()
            cu = torch.from_numpy(np.stack([m.cu for m in step_mbs])).pin_memory()
            self.q.put((inputs, targets, cu))

    def next_step(self):
        if not self._started:
            self._started = True
            threading.Thread(target=self._run, daemon=True).start()
        item = self.q.get()
        if isinstance(item, BaseException):
            raise RuntimeError("stream loader thread failed") from item
        return item


def build_train_sample(cfg, rank: int, world_size: int):
    """A fixed sample of the fixed dataset for measuring train loss: n microbatches (n = train_eval_tokens /
    micro_tokens, rounded to a multiple of world) spread evenly over the first M microbatches of the packed
    stream (M as in FixedDataset with world 8, i.e. the same set at any world size up to 8); sample i goes to
    rank i % world. Works in any mode (re-packs the stream). Returns (list of CPU tensors, global ids)."""
    T = cfg.micro_tokens
    M = cfg.dataset_tokens // (T * 8) * 8
    n = max(world_size, cfg.train_eval_tokens // T // world_size * world_size)
    gids = sorted({int(i * M / n) for i in range(n)})
    assert len(gids) == n
    mine = {g for i, g in enumerate(gids) if i % world_size == rank}
    files = sorted(Path(f) for f in glob.glob(cfg.train_glob))
    buf, cu = _packed_cache(cfg, M, 0, files)
    out = []
    for k in sorted(mine):
        b = np.asarray(buf[k]).astype(np.int32)
        out.append((torch.from_numpy(b[:-1].copy()), torch.from_numpy(b[1:].astype(np.int64)), torch.from_numpy(np.asarray(cu[k]).copy())))
    return out, [g for g in gids if g in mine]


def val_chunks(cfg, rank: int, world_size: int):
    """The WR validation set: val_tokens tokens of the val shard in chunks of val_chunk_tokens, not
    BOS-aligned; chunk k goes to rank k % world (the WR's chunks exactly when world = 8)."""
    files = sorted(Path(f) for f in glob.glob(cfg.val_glob))
    tokens = load_shard(files[0])
    C = cfg.val_chunk_tokens
    num = cfg.val_tokens // C
    assert num % world_size == 0, "val chunks must split evenly over ranks"
    R = cu_seqlens_rows(C)
    out = []
    for k in range(rank, num, world_size):
        buf = tokens[k * C: (k + 1) * C + 1].astype(np.int32)
        inputs, targets = buf[:-1], buf[1:]
        bos_pos = np.flatnonzero(inputs == cfg.bos_id)
        cu = np.full(R, C, dtype=np.int32)
        cu[0] = 0
        cu[1:len(bos_pos) + 1] = bos_pos
        out.append((torch.from_numpy(inputs.copy()), torch.from_numpy(targets.astype(np.int64)), torch.from_numpy(cu)))
    return out, num
