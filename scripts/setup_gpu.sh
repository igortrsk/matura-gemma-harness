#!/usr/bin/env bash
# One-time setup on the Forgehand GPU machine (persists in /workspace across sessions).
# 1) NVIDIA CUDA compiler via conda  2) build llama.cpp for the L40S (sm_89)  3) download Gemma 4 12B QAT.
set -euo pipefail
LOG=/workspace/setup.log; exec > >(tee -a "$LOG") 2>&1
echo "=== setup start $(date)"

# Model download runs in parallel with the build.
(
  mkdir -p /workspace/models/gemma-4-12b-qat
  hf download google/gemma-4-12B-it-qat-q4_0-gguf --local-dir /workspace/models/gemma-4-12b-qat \
    && echo "=== MODEL DOWNLOAD DONE $(date)" || echo "=== MODEL DOWNLOAD FAILED"
) &

if [ ! -x /workspace/cuda/bin/nvcc ]; then
  /opt/conda/bin/conda create -y -q -p /workspace/cuda -c nvidia/label/cuda-12.8.1 cuda-nvcc cuda-cudart-dev libcublas-dev cuda-nvrtc-dev
fi
export CUDA_HOME=/workspace/cuda PATH=/workspace/cuda/bin:$PATH
nvcc --version | tail -1

if [ ! -d /workspace/llama.cpp ]; then git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp; fi
cd /workspace/llama.cpp
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCUDAToolkit_ROOT=/workspace/cuda \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4 --target llama-server llama-cli
ls -la build/bin/llama-server && echo "=== BUILD DONE $(date)"
wait
echo "=== setup end $(date)"
