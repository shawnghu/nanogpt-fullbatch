#!/bin/bash
#SBATCH -p general
#SBATCH --qos=high-eur
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH -t 06:00:00
#SBATCH -o /workspace/shawnghu/nanogpt-fb/logs/abl/%x_%j.out
# usage: sbatch -J <name> abl_job.sbatch <runlist>   (runlist lines: "<run_name> VAR=val VAR=val ...")
V=$(/workspace/shawnghu/nanogpt-fb/stage_venv.sh)
export PATH=$V/bin:$PATH HF_HOME=/workspace/shawnghu/nanogpt-fb/hf HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda
export TORCHINDUCTOR_CACHE_DIR=/tmp/shawnghu-nfb/inductor-abl
cd /workspace/shawnghu/nanogpt-fb/abl
echo "node $(hostname) start $(date)"
while read -r name envs; do
  [ -z "$name" ] && continue; [[ "$name" == \#* ]] && continue
  out=runs/$name.out
  if grep -q "peak memory" $out 2>/dev/null; then echo "skip $name (done)"; continue; fi
  echo "=== $name: $envs  $(date)"
  env ABL_ALLOW_WORLD_SIZE=1 $envs python -m torch.distributed.run --standalone --nproc_per_node=4 train_gpt.py > $out 2>&1
  echo "=== $name exit $? $(date): $(grep -a val_loss $out | tail -1)"
done < "$1"
echo "all done $(date)"
