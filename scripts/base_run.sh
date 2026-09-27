#!/usr/bin/env bash
# Untouched base model on an exam package (the "base" answers for the progress score): plain Gemma, thinking ON,
# no fallback, no retrieval, no adapters (explicitly switched off) — same as runs/exam-base-* (123/180 on 2023–25).
#   scripts/base_run.sh RUN EXAM_DIR [KEY_YEAR]      (PARALLEL=4 by default)
cd /data; mkdir -p runs/$1
PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_exam.py --run $1 --exam $2 ${3:+--key-year $3} --model gemma-4-12b-qat \
  --think --no-no-think-fallback --parallel ${PARALLEL:-4} 2>&1 | tee -a runs/$1/log.txt
