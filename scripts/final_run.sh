#!/usr/bin/env bash
# Frozen final setup: harness flags chosen on 2023–25 by rules fixed before the results, stored in scripts/final_flags.txt.
#   scripts/final_run.sh RUN EXAM_DIR [KEY_YEAR]      (PARALLEL=4 by default; server needs adapters 0 = essay-v2, 1 = short-v2-rag)
cd /data
[ -s scripts/final_flags.txt ] || { echo "no scripts/final_flags.txt yet"; exit 1; }
mkdir -p runs/$1
PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_exam.py --run $1 --exam $2 ${3:+--key-year $3} --model gemma-4-12b-qat \
  $(cat scripts/final_flags.txt) --parallel ${PARALLEL:-4} 2>&1 | tee -a runs/$1/log.txt
