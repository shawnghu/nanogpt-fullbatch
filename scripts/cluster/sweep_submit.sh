#!/bin/bash
# usage: sweep_submit.sh <ngpu> <B> <steps> <lr_mul> [extra key=value ...]
NG=$1; B=$2; S=$3; LR=$4; shift 4
name=B${B}_S${S}_L${LR}${SWEEP_TAG}
cd /workspace/shawnghu/nanogpt-fb/fb
# time limit: tokens / (0.85M tok/s/GPU) + 15 min slack
mins=$(python3 -c "print(int($B*$S/(0.85e6*$NG)/60)+15)")
sbatch -J sw_$name --gres=gpu:$NG --cpus-per-task=$((NG*4)) --mem=$((NG*75))G -t $mins run_fb.sh train_fb.py mode=minibatch minibatch_source=stream \
  batch_tokens=$B steps=$S lr_mul=$LR val_every=$((S/12)) log_every=10 out_dir=runs/sweep/$name "$@"
