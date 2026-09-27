#!/usr/bin/env bash
# (local machine) Keep the forwards 8080 (Gemma) / 8090 (KB) to (GPU machine) alive: check every 20 s, re-open if down.
#   setsid nohup scripts/tunnel_keeper.sh >/dev/null 2>&1 </dev/null &      log: runs/tunnel.txt
cd /data; S="ssh -F /data/.ssh/config"
while true; do
  if ! curl -s -m 8 localhost:8080/health | grep -q ok || ! curl -s -m 8 localhost:8090/health | grep -q ok; then
    echo "$(date -u +%T) tunnel down -> reopening" >> runs/tunnel.txt
    $S -O check fh 2>/dev/null || { $S -O exit fh 2>/dev/null; $S -f -N fh </dev/null 2>/dev/null; }
    $S -O forward -L 8080:127.0.0.1:8080 fh 2>/dev/null; $S -O forward -L 8090:127.0.0.1:8090 fh 2>/dev/null
  fi
  sleep 20
done
