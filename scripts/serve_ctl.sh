#!/usr/bin/env bash
# (GPU machine) Gemma model server control (local-disk setup from boot_gpu.sh): serve_ctl.sh start|stop|status
case "$1" in
  stop)
    for p in $(pgrep -f "[k]eep_serving.sh"); do kill "$p"; done
    for p in $(pgrep -f "[l]lama-server -m /scratch/models/gemma"); do kill "$p"; done
    for i in $(seq 30); do pgrep -f "[l]lama-server -m /scratch/models/gemma" >/dev/null || break; sleep 1; done
    echo "stopped; server procs left: $(pgrep -fc '[l]lama-server -m /scratch/models/gemma')";;
  start)
    pgrep -f "[k]eep_serving.sh" >/dev/null || (setsid nohup bash /scratch/app/keep_serving.sh >/dev/null 2>&1 </dev/null &)
    for i in $(seq 120); do curl -s localhost:8080/health | grep -q ok && break; sleep 2; done
    echo "server: $(curl -s localhost:8080/health)";;
  *) echo "server: $(curl -s localhost:8080/health) | watchdog: $(pgrep -fc '[k]eep_serving.sh')";;
esac
