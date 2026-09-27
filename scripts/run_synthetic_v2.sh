#!/usr/bin/env bash
# Synthetic v2: ~650 extra essays with the deeper-argumentation prompt, new topics (not repeating v1 or the
# 2023–25 practice topics), hard cap $14.50 on the team LLM ledger.
# Start detached:  setsid nohup scripts/run_synthetic_v2.sh >/dev/null 2>&1 </dev/null &
# Resumable: rerunning skips essays already written (the cap counts only the new run's spend).
cd /data
mkdir -p synthetic/v2
.venv/bin/python scripts/generate_synthetic.py --run v2 --new-essays-per-era 28 --essay-v2 --effort medium \
  --parallel 8 --max-usd 14.5 --avoid-topics synthetic/v1/topics.jsonl >> synthetic/v2/log.txt 2>&1
echo "EXIT $? $(date)" >> synthetic/v2/log.txt
