#!/bin/bash
# usage: ladder_submit.sh <name> <ngpu> <qos> <hours> key=value ...   (common 800M-set args added)
N=$1; NG=$2; Q=$3; H=$4; shift 4
cd /workspace/shawnghu/nanogpt-fb/fb
COMMON="dataset_tokens=800000000 train_eval_tokens=10027008 val_tokens=10485760 ckpt_minutes=15 log_every=1"
sbatch -J L_$N --gres=gpu:$NG --qos=$Q --cpus-per-task=$((NG*4)) --mem=$((NG*90))G -t $H:00:00 \
  run_fb.sh train_fb.py $COMMON out_dir=runs/ladder800/$N "$@"
