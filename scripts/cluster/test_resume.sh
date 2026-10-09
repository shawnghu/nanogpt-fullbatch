#!/bin/bash
#SBATCH -p general
#SBATCH --qos=high-eur
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 01:30:00
#SBATCH -o /workspace/shawnghu/nanogpt-fb/logs/fb_%x_%j.out
export HF_HOME=/workspace/shawnghu/nanogpt-fb/hf HF_HUB_OFFLINE=1
V=$(/workspace/shawnghu/nanogpt-fb/stage_venv.sh) || V=/workspace/shawnghu/nanogpt-fb/venv
export PATH=$V/bin:$PATH CUDA_HOME=/usr/local/cuda TORCHINDUCTOR_CACHE_DIR=/tmp/shawnghu-inductor-fb NCCL_DEBUG=WARN
cd /workspace/shawnghu/nanogpt-fb/fb
R=runs/test_resume; rm -rf $R; mkdir -p $R
run() { python -m torch.distributed.run --standalone --nproc_per_node=2 train_fb.py "$@"; echo "=== EXIT $? $*"; }
# (a) minibatch from the fixed set
A="mode=minibatch minibatch_source=fixed dataset_tokens=3145728 batch_tokens=393216 steps=14 val_tokens=524288 train_eval_tokens=393216 train_eval_every=2 ckpt_minutes=0"
run $A out_dir=$R/mbA
run $A out_dir=$R/mbB stop_after_steps=5
run $A out_dir=$R/mbB stop_after_steps=4
run $A out_dir=$R/mbB
# (b) full batch + dynamics suite + token losses
F="mode=full dataset_tokens=1572864 steps=12 val_tokens=524288 train_eval_tokens=393216 train_eval_every=2 dyn=true token_loss_every=3 ckpt_minutes=0 ckpt_milestone_every=6"
run $F out_dir=$R/fbA
run $F out_dir=$R/fbB stop_after_steps=7
run $F out_dir=$R/fbB
# (c) halting rule against a fake reference 0.5 nats better than run A
python - <<'PY'
import json
out = open("runs/test_resume/fake_ref.jsonl", "w")
for l in open("runs/test_resume/mbA/metrics.jsonl"):
    r = json.loads(l)
    if "val_loss" in r and "step" in r and r["step"] >= 0:
        out.write(json.dumps({"step": r["step"], "val_loss": r["val_loss"] - 0.5, "train_eval": r["train_eval"]}) + "\n")
PY
run $A out_dir=$R/halt halt_ref=$R/fake_ref.jsonl halt_span=4 halt_min_step=2
run $A out_dir=$R/halt halt_ref=$R/fake_ref.jsonl halt_span=4 halt_min_step=2
ls -R $R | head -80
