#!/usr/bin/env bash
# Full synthetic dataset v1: fetch sources, then generate.
# Start detached so it survives logout:  setsid nohup scripts/run_synthetic_v1.sh >/dev/null 2>&1 </dev/null &
# Resumable: re-running skips excerpts/topics/items already generated (sources are re-fetched unless present).
cd /data
mkdir -p synthetic/v1
if [ ! -s sources/excerpts.jsonl ] || [ "$(wc -l < sources/excerpts.jsonl)" -lt 100 ]; then
  .venv/bin/python scripts/fetch_sources.py > synthetic/v1/fetch.log 2>&1 || { echo "FETCH FAILED $(date)" >> synthetic/v1/log.txt; exit 1; }
fi
.venv/bin/python scripts/generate_synthetic.py --run v1 --old-all --source-tasks 700 --new-essays-per-era 12 \
  --effort low --parallel 6 >> synthetic/v1/log.txt 2>&1
echo "EXIT $? $(date)" >> synthetic/v1/log.txt
