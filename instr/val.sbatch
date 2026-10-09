#!/bin/bash
#SBATCH -J nfb-instr-val
#SBATCH -p general
#SBATCH --qos=high-eur
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH -t 03:00:00
#SBATCH -o /workspace/shawnghu/nanogpt-fb/instr/logs/%x_%j.out
export PATH=/workspace/shawnghu/nanogpt-fb/venv/bin:$PATH
cd /workspace/shawnghu/nanogpt-fb/instr
hostname; nvidia-smi -L
python -u validate_hvp.py "$@"
echo EXIT $?
