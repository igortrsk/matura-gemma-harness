#!/usr/bin/env bash
# (GPU machine) Fine-tune + convert on the LOCAL disk, then copy results to /workspace (persistent).
#   setsid nohup bash /scratch/app/ft_run.sh NAME SFT_NAME EPOCHS >/dev/null 2>&1 </dev/null &
# Log /scratch/logs/train-NAME.log; adapter /scratch/ft/adapters/NAME (+ ckpt-N) and .gguf; copied to /workspace/ft/adapters/.
NAME=$1; SFT=$2; EPOCHS=${3:-2}
LOG=/scratch/logs/train-$NAME.log
mkdir -p /scratch/ft/sft /scratch/ft/adapters && cp -r /workspace/ft/sft/$SFT /scratch/ft/sft/
cd /scratch/ft
/workspace/ft-venv/bin/python /scratch/app/train_lora.py --data sft/$SFT/train.jsonl --out adapters/$NAME --epochs $EPOCHS > $LOG 2>&1 \
  || { echo "TRAIN FAILED" >> $LOG; exit 1; }
for d in adapters/$NAME adapters/$NAME/ckpt-*; do
  [ -f $d/adapter_config.json ] || cp adapters/$NAME/adapter_config.json $d/
  /workspace/ft-venv/bin/python /workspace/llama.cpp/convert_lora_to_gguf.py --base /workspace/models/gemma-4-12b-qat-unq \
    --outtype f16 --outfile $d.gguf $d >> $LOG 2>&1 && echo "converted $d.gguf" >> $LOG
done
mkdir -p /workspace/ft/adapters && cp -r adapters/$NAME adapters/$NAME.gguf /workspace/ft/adapters/ 2>>$LOG && echo "copied to /workspace" >> $LOG
echo "ALL DONE $(date -u +%T)" >> $LOG
