import json, glob, os, sys
for d in sorted(glob.glob(sys.argv[1])):
    f = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(f): continue
    out = []
    for l in open(f):
        r = json.loads(l)
        if "val_loss" in r:
            out.append(f"{r['step']} {r['train_loss']:.3f}/{r['val_loss_canon']:.3f}")
    print(os.path.basename(d) + ": " + "; ".join(out))
