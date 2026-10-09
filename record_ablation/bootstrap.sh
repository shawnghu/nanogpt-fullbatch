#!/bin/bash
# Run on a fresh RunPod 8xH200 pod (root). Expects /root/abl to hold the abl code.
set -e
cd /root
nvidia-smi --query-gpu=name,driver_version --format=csv | head -3
pip install -q uv 2>/dev/null || true
uv venv -q -p 3.12 /root/venv || python3 -m venv /root/venv
. /root/venv/bin/activate
uv pip install -q numpy tqdm huggingface-hub==1.29.0 kernels==0.16.1 setuptools typing-extensions==4.15.0 datasets tiktoken
uv pip install -q torch==2.10 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q nvidia-cuda-runtime-cu13 2>/dev/null || uv pip install -q "nvidia-cuda-runtime>=13" || true
python -c "import torch;print(torch.__version__, torch.cuda.device_count())"
cd /root/abl && rm -f data/fineweb10B && python data/cached_fineweb10B.py 12
hf download devenpzak/flash-attn3-12864 --revision 64c1e6d1f2780e7931839f41426ddcdb564a7cb9 >/dev/null
HUB=${HF_HOME:-/root/.cache/huggingface}/hub; cp -rl $HUB/models--devenpzak--flash-attn3-12864 $HUB/kernels--devenpzak--flash-attn3-12864 || true
echo BOOTSTRAP_DONE
