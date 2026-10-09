"""Summarize abl runs: python summarize.py runs/*.out  (reads env from the runlists)."""
import glob, os, re, subprocess, sys, json
lists = {}
for f in glob.glob("runs/list*.txt") + glob.glob("runs/queue.txt"):
    for line in open(f):
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            lists[parts[0]] = dict(p.split("=", 1) for p in parts[1:])
def tokens_for(envs):
    code = ("from track_1_short.config import *\nfrom track_1_short.schedule import TrainingSchedule\n"
            "import torch\nh=Hyperparameters()\ns=TrainingSchedule(TRAINING_STAGES,h.num_scheduled_iterations,h.num_extension_iterations,"
            "torch.device('cpu'),LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT)\n"
            "print(s.total_steps, sum(s.lookup(i)[0].batch_size for i in range(s.total_steps)))")
    out = subprocess.run([sys.executable, "-c", code], env={**os.environ, **envs}, capture_output=True, text=True)
    return out.stdout.split() if out.returncode == 0 else ("?", "?")
rows = []
for f in sorted(sys.argv[1:]):
    name = os.path.basename(f)[:-4]
    txt = open(f, errors="replace").read()
    vals = re.findall(r"step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms", txt)
    tr = re.findall(r"step:(\d+)/\d+ train_eval_loss:([\d.]+)", txt)
    steps, toks = tokens_for(lists.get(name, {}))
    status = "done" if "peak memory" in txt else ("ERR" if "Traceback" in txt or "Error" in txt else "running")
    last = vals[-1] if vals else None
    rows.append(dict(name=name, status=status, steps=steps, tokens_M=round(int(toks)/1e6,1) if toks!="?" else "?",
                     final_val=float(last[2]) if last and last[0]==last[1] else None,
                     train_time_s=round(int(last[3])/1000,1) if last else None,
                     curve=[(int(a), float(c)) for a,b,c,d in vals],
                     train_eval=[(int(a), float(b)) for a,b in tr]))
for r in rows:
    print(f"{r["name"]:28s} {r["status"]:7s} steps={r["steps"]:>5} tok={r["tokens_M"]:>7}M val={r["final_val"]} t={r["train_time_s"]}s")
json.dump(rows, open("runs/summary.json","w"))
