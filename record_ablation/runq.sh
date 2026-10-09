#!/bin/bash
# Pod queue runner: processes /root/abl/runs/queue.txt line by line ("<name> VAR=val ..."), forever.
# Append lines to queue.txt to enqueue; finished runs have runs/<name>.out containing "peak memory".
. /root/venv/bin/activate
export LD_LIBRARY_PATH=$(python -c "import nvidia, os; print(':'.join(os.path.join(p,'cu13','lib') for p in nvidia.__path__))" 2>/dev/null):$(dirname $(find /root/venv -name 'libcudart.so.13' | head -1)):$LD_LIBRARY_PATH
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCHINDUCTOR_CACHE_DIR=/root/inductor HF_HUB_OFFLINE=1
cd /root/abl; mkdir -p runs
touch runs/queue.txt
while true; do
  line=$(grep -v '^#' runs/queue.txt | while read -r n e; do [ -n "$n" ] && ! grep -q "peak memory" runs/$n.out 2>/dev/null && ! [ -f runs/$n.failed ] && { echo "$n $e"; break; }; done)
  if [ -z "$line" ]; then sleep 20; continue; fi
  read -r name envs <<< "$line"
  echo "=== $name: $envs $(date)" >> runs/runq.log
  env $envs python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py > runs/$name.out 2>&1
  rc=$?
  [ $rc -ne 0 ] && touch runs/$name.failed
  echo "=== $name exit $rc $(date): $(grep -a val_loss runs/$name.out | tail -1)" >> runs/runq.log
done
