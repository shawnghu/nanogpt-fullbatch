"""Where do per-token training losses differ between K separate microbatches and their concatenation?"""
import os
import sys

import torch
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fullbatch.config import parse_config
from fullbatch.trainer import FullBatchTrainer
from track_1_short.data import cu_seqlens_rows

cfg = parse_config(sys.argv[1:])
t = FullBatchTrainer(cfg)
for s in range(int(os.environ.get("PRE_STEPS", "10"))):
    t.train_step(s)
K, T = 4, cfg.micro_tokens
mbs = t.fixed_mbs(list(range(K)))
t.model.train()
with torch.no_grad():
    sep = [t.model(i, tg, cu, t._slots_off(T), t.fwd_cfg).float() for _, i, tg, cu in mbs]
    sep2 = [t.model(i, tg, cu, t._slots_off(T), t.fwd_cfg).float() for _, i, tg, cu in mbs]
    inputs = torch.cat([m[1] for m in mbs]); targets = torch.cat([m[2] for m in mbs])
    segs = torch.cat([torch.cat([m[3][1:][m[3][1:] < T], torch.tensor([T], device=t.device, dtype=torch.int32)]) + j * T
                      for j, m in enumerate(mbs)])
    R = cu_seqlens_rows(K * T)
    cu_big = torch.full((R,), K * T, dtype=torch.int32, device=t.device); cu_big[0] = 0; cu_big[1:len(segs) + 1] = segs
    big = t.model(inputs, targets, cu_big, t._slots_off(K * T), t.fwd_cfg).float()
sep = torch.cat(sep); sep2 = torch.cat(sep2)
d = (big - sep).abs()
print("repeat max diff", (sep - sep2).abs().max().item())
print("mean loss sep", sep.mean().item(), "big", big.mean().item())
print("abs diff: mean", d.mean().item(), "max", d.max().item(), "frac>1e-3", (d > 1e-3).float().mean().item())
for j in range(K):
    dj = d[j * T:(j + 1) * T]
    print(f"mb {j}: mean {dj.mean().item():.3e} max {dj.max().item():.3e} first8 {[round(x, 4) for x in dj[:8].tolist()]}")
# positions of big diffs
idx = torch.nonzero(d > 1e-2).flatten()
print("n>1e-2:", idx.numel(), "first positions", idx[:30].tolist())
starts = cu_big[:len(segs) + 1].tolist()
print("segment starts (first 30)", starts[:30])
dist.destroy_process_group()
