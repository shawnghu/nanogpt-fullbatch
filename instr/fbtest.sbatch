#!/bin/bash
#SBATCH -J nfb-instr-fbtest
#SBATCH -p general
#SBATCH --qos=high-eur
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH -t 01:30:00
#SBATCH -o /workspace/shawnghu/nanogpt-fb/instr/logs/%x_%j.out
export HF_HOME=/workspace/shawnghu/nanogpt-fb/hf HF_HUB_OFFLINE=1
V=$(/workspace/shawnghu/nanogpt-fb/stage_venv.sh) || V=/workspace/shawnghu/nanogpt-fb/venv
export PATH=$V/bin:$PATH CUDA_HOME=/usr/local/cuda
export TORCHINDUCTOR_CACHE_DIR=/tmp/shawnghu-inductor-instr
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /workspace/shawnghu/nanogpt-fb/fb
NGPU=${SLURM_GPUS_ON_NODE:-1}
echo "host $(hostname) gpus $NGPU"
python -m torch.distributed.run --standalone --nproc_per_node=$NGPU /workspace/shawnghu/nanogpt-fb/instr/test_fb_oracle.py "$@"
echo "EXIT $?"
