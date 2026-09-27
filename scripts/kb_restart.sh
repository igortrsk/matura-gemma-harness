#!/usr/bin/env bash
# (GPU machine) (Re)start the KB search service from the local disk (/scratch/app, /scratch/kb). Logs local.
for p in $(pgrep -f "[k]b_server.py --dir"); do kill "$p"; done
for i in $(seq 20); do pgrep -f "[k]b_server.py --dir" >/dev/null || break; sleep 0.5; done
cd /scratch/app
setsid nohup /workspace/ft-venv/bin/python kb_server.py --dir /scratch/kb >/scratch/logs/kb_server.log 2>&1 </dev/null &
echo $! > /scratch/logs/kb_server.pid
for i in $(seq 60); do curl -s localhost:8090/health | grep -q ok && break; sleep 1; done
echo "kb_server pid $(cat /scratch/logs/kb_server.pid): $(curl -s localhost:8090/health)"
