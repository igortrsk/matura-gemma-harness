#!/usr/bin/env bash
# One-time fine-tuning environment (persistent, on /workspace). Driver supports CUDA 13.x -> cu130 wheels.
# torch + torchvision must come from the same index in one command, otherwise a later install
# upgrades torch alone and torchvision breaks ("operator torchvision::nms does not exist").
set -ex
export UV_LINK_MODE=copy
[ -d /workspace/ft-venv ] || uv venv /workspace/ft-venv --python 3.11
source /workspace/ft-venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers accelerate peft trl bitsandbytes datasets sentencepiece protobuf gguf
python -c "import torch,transformers,peft,trl,bitsandbytes;print(\"OK\",torch.__version__,transformers.__version__,peft.__version__,trl.__version__,bitsandbytes.__version__,torch.cuda.is_available())"
echo SETUP_DONE
