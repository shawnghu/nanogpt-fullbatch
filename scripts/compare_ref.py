"""Matched-step comparison of a run against a reference (the halting rule, offline).
python compare_ref.py <run_dir> <ref_dir> [margin=0.03] [span=10]
Prints x (= updates done), run train_eval/val, ref (interpolated), deltas, and whether the halting rule fires."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(path):
    pts = {"train_eval": [], "val_loss": [], "train_loss": []}
    for line in open(os.path.join(path, "metrics.jsonl")):
        r = json.loads(line)
        if "step" not in r or "event" in r:
            continue
        for k in pts:
            if k in r and r[k] == r[k] and (k != "train_loss" or "val_loss" in r):
                pts[k].append((r["step"] + 1, r[k]))
    return {k: sorted(dict(v).items()) for k, v in pts.items()}


def interp(pts, x):
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


run, ref = load(sys.argv[1]), load(sys.argv[2])
margin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
span = int(sys.argv[4]) if len(sys.argv) > 4 else 10
bad_since, fired = None, None
print(f"{'x':>5} {'train':>7} {'ref':>7} {'d':>7} | {'val':>7} {'ref':>7} {'d':>7}")
for x, v in run["val_loss"]:
    tr = dict(run["train_eval"]).get(x)
    rt, rv = interp(ref["train_eval"], x), interp(ref["val_loss"], x)
    dt = None if tr is None or rt is None else tr - rt
    dv = None if rv is None else v - rv
    worse = (dt is not None and dt > margin) or (dv is not None and dv > margin)
    if worse:
        bad_since = bad_since or x
        if x - bad_since >= span and fired is None:
            fired = x
    elif dt is not None or dv is not None:
        bad_since = None
    f = lambda a: "   -   " if a is None else f"{a:7.3f}"
    print(f"{x:5d} {f(tr)} {f(rt)} {f(dt)} | {f(v)} {f(rv)} {f(dv)} {'*' if worse else ''}")
print("halting rule fires at x =", fired)
