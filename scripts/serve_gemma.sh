#!/usr/bin/env bash
# Start llama-server with Gemma 4 12B QAT (+ vision) on the GPU machine (used via serve_ctl.sh / keep_serving.sh).
# -np 4: four parallel slots; -c: total context shared across slots (page images use many tokens).
# -b/-ub 4096: an image must fit in one micro-batch (Gemma vision uses non-causal attention; default 512 crashes).
M=/workspace/models/gemma-4-12b-qat
exec /workspace/llama.cpp/build/bin/llama-server \
  -m $M/gemma-4-12b-it-qat-q4_0.gguf --mmproj $M/mmproj-gemma-4-12b-it-qat-q4_0.gguf \
  -ngl 999 -np 4 -c 131072 -b 4096 -ub 4096 --jinja --host 127.0.0.1 --port 8080 \
  --alias gemma-4-12b-qat "$@"
