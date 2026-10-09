#!/bin/bash
#SBATCH -p general
#SBATCH --qos=high-eur
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH -t 01:00:00
#SBATCH -o /workspace/shawnghu/nanogpt-fb/logs/fb_%x_%j.out
# runs each line of the file $1 as "train_fb.py/verify_fb.py args" on 8 GPUs, sequentially
export HF_HOME=/workspace/shawnghu/nanogpt-fb/hf HF_HUB_OFFLINE=1
V=$(/workspace/shawnghu/nanogpt-fb/stage_venv.sh) || V=/workspace/shawnghu/nanogpt-fb/venv
export PATH=$V/bin:$PATH CUDA_HOME=/usr/local/cuda TORCHINDUCTOR_CACHE_DIR=/tmp/shawnghu-inductor-fb NCCL_DEBUG=WARN
cd /workspace/shawnghu/nanogpt-fb/fb
echo "host $(hostname)"
while read -r line; do
  [ -z "$line" ] && continue
  echo "=== RUN $line"
  python -m torch.distributed.run --standalone --nproc_per_node=8 $line
  echo "=== EXIT $? $line"
done < "$1"
