#!/usr/bin/env bash
# (GPU machine) After the short-answer fine-tune: full-size Gemma server with both adapters
# (id 0 = essay-v2, id 1 = short-v2-rag, loaded but unused in the final setup). Copy lives in /scratch/app/.
set -e
A=/scratch/models/adapters; L=/scratch/logs/train-short-v2-rag.log
grep -q "ALL DONE" $L || { echo "fine-tune not finished"; exit 1; }
# checkpoint with the lowest validation loss (ties -> later step)
best=$(grep "VAL LOSS" $L | sed -E 's/.*step ([0-9]+)\/.*VAL LOSS ([0-9.]+).*snapshot ([^)]+)\).*/\2 \1 \3/' \
  | sort -k1,1g -k2,2nr | head -1)
echo "best by val loss (loss step snapshot): $best"
src=/scratch/ft/$(echo "$best" | awk '{print $3}').gguf
[ -f "$src" ] || src=/scratch/ft/adapters/short-v2-rag.gguf
cp "$src" $A/short-v2-rag.gguf && echo "short adapter <- $src"
cd /scratch/app
sed "s#--alias gemma-4-12b-qat#--alias gemma-4-12b-qat --lora $A/essay-v2.gguf --lora $A/short-v2-rag.gguf --lora-init-without-apply#" \
  serve_gemma.sh.full > serve_gemma.sh
grep -o -- "-np [0-9]* -c [0-9]*" serve_gemma.sh
bash serve_ctl.sh stop
bash serve_ctl.sh start
curl -s localhost:8080/lora-adapters; echo
