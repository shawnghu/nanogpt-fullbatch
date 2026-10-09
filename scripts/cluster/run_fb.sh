#!/bin/bash
# usage: sbatch --gres=gpu:N [-t ..] run_fb.sh <script.py> key=value ...
#SBATCH -p general
#SBATCH --qos=high-eur
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH -t 04:00:00
#SBATCH -o /workspace/shawnghu/nanogpt-fb/logs/fb_%x_%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
export HF_HOME=/workspace/shawnghu/nanogpt-fb/hf HF_HUB_OFFLINE=1
V=$(/workspace/shawnghu/nanogpt-fb/stage_venv.sh) || V=/workspace/shawnghu/nanogpt-fb/venv
export PATH=$V/bin:$PATH CUDA_HOME=/usr/local/cuda
export TORCHINDUCTOR_CACHE_DIR=/tmp/shawnghu-inductor-fb
export NCCL_DEBUG=WARN
cd /workspace/shawnghu/nanogpt-fb/fb
NGPU=$(nvidia-smi -L >/dev/null; echo ${SLURM_GPUS_ON_NODE:-1})
echo "host $(hostname) gpus $NGPU restart ${SLURM_RESTART_COUNT:-0} $(date) args $@"
python -m torch.distributed.run --standalone --nproc_per_node=$NGPU "$@"
echo "EXIT $?"
