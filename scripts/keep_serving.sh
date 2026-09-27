#!/usr/bin/env bash
# Keep the Gemma llama-server alive: restart it within seconds if it dies (twice today everything in tmux was killed).
# Start detached:  setsid nohup bash /workspace/keep_serving.sh >/dev/null 2>&1 </dev/null &
# Stop:            pkill -f keep_serving.sh; pkill -f llama-server
while true; do
  echo "=== start $(date -u +%FT%TZ)" >> /workspace/serve.log
  bash /workspace/serve_gemma.sh >> /workspace/serve.log 2>&1
  echo "=== exited $? $(date -u +%FT%TZ), restarting in 5 s" >> /workspace/serve.log
  sleep 5
done
