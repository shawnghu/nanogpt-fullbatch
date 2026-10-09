"""Summarize sweep runs: python analyze_sweep.py runs/sweep [target=3.28] [metric=val_loss_canon]"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

root = sys.argv[1]
target = float(sys.argv[2]) if len(sys.argv) > 2 else 3.28
metric = sys.argv[3] if len(sys.argv) > 3 else "val_loss_canon"
rows = []
for d in sorted(glob.glob(os.path.join(root, "*"))):
    f = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(f):
        continue
    cfg = json.load(open(os.path.join(d, "config.json")))
    recs = [json.loads(l) for l in open(f) if l.strip()]
    vals = [r for r in recs if metric in r]
    done = any(r.get("event") == "end" for r in recs)
    last = vals[-1] if vals else {}
    tps = [r["tok_per_s_per_gpu"] for r in recs if "tok_per_s_per_gpu" in r][2:]
    rows.append(dict(name=os.path.basename(d), B=cfg["batch_tokens"] or cfg["dataset_tokens"], S=cfg["steps"],
                     lr=cfg["lr_mul"], final=last.get(metric), plain=last.get("val_loss"),
                     at=last.get("step"), done=done, tps=(sum(tps) / len(tps) if tps else None)))
print(f"{'name':40s} {'B':>9s} {'S':>5s} {'lr':>5s} {'step':>5s} {metric:>14s} {'val_loss':>9s} done tok/s/gpu")
for r in rows:
    f = f"{r['final']:.4f}" if r["final"] else "-"
    p = f"{r['plain']:.4f}" if r["plain"] else "-"
    t = f"{r['tps']:.0f}" if r["tps"] else "-"
    print(f"{r['name']:40s} {r['B']:9d} {r['S']:5d} {r['lr']:5.2f} {str(r['at']):>5s} {f:>14s} {p:>9s} {r['done']!s:5s} {t}")
# steps-to-target per (B, lr): interpolate final loss vs log(S) between bracketing finished runs
groups = defaultdict(list)
for r in rows:
    if r["done"] and r["final"]:
        groups[(r["B"], r["lr"])].append((r["S"], r["final"]))
print(f"\nsteps to {metric} <= {target} (log-linear interpolation of final loss vs steps)")
for (B, lr), pts in sorted(groups.items()):
    pts.sort()
    est = None
    for (s0, l0), (s1, l1) in zip(pts, pts[1:]):
        if l0 > target >= l1:
            w = (l0 - target) / (l0 - l1)
            est = math.exp(math.log(s0) + w * (math.log(s1) - math.log(s0)))
    if est is None and len(pts) >= 2 and pts[-1][1] > target:
        (s0, l0), (s1, l1) = pts[-2], pts[-1]
        if l0 > l1:
            est = -math.exp(math.log(s1) + (l1 - target) / (l0 - l1) * (math.log(s1) - math.log(s0)))  # extrapolated (negative = flag)
    print(f"B={B:9d} lr={lr:5.2f} pts={pts} S*={'%.0f' % abs(est) if est else '-'}{' (extrapolated)' if est and est < 0 else ''}"
          f"{'  tokens=%.2fB' % (abs(est) * B / 1e9) if est else ''}")
