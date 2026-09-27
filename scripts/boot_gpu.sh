#!/usr/bin/env bash
# (GPU machine) After a (new) session start: stage hot files on the LOCAL disk /scratch so a /workspace (NFS) hiccup
# cannot freeze running jobs (a blocked log write on NFS can leave llama-server stuck in "D" state).
# Usage: bash /workspace/ft/boot_gpu.sh            (copies ~11 GB, a few minutes; idempotent)
# Afterwards: bash /scratch/app/serve_ctl.sh start|stop|status ; bash /scratch/app/kb_restart.sh
set -e
mkdir -p /scratch/models/gemma /scratch/kb /scratch/logs /scratch/app /scratch/ft
cp -u /workspace/ft/kb.py /workspace/ft/kb_server.py /workspace/ft/train_lora.py /workspace/ft/ft_run.sh \
      /workspace/ft/serve_ctl.sh /workspace/ft/kb_restart.sh /scratch/app/
M=/workspace/models/gemma-4-12b-qat
[ -s /scratch/models/gemma/gemma-4-12b-it-qat-q4_0.gguf ] || cp $M/gemma-4-12b-it-qat-q4_0.gguf $M/mmproj-gemma-4-12b-it-qat-q4_0.gguf /scratch/models/gemma/
[ -d /scratch/kb/tantivy ] || cp -r /workspace/kb/passages.sqlite /workspace/kb/tantivy /scratch/kb/
cat > /scratch/app/serve_gemma.sh <<'EOS'
#!/usr/bin/env bash
# Gemma 4 12B QAT Q4_0 + vision projector from the local disk; 4 slots x 32k context.
M=/scratch/models/gemma
exec /workspace/llama.cpp/build/bin/llama-server -m $M/gemma-4-12b-it-qat-q4_0.gguf --mmproj $M/mmproj-gemma-4-12b-it-qat-q4_0.gguf \
  -ngl 999 -np 4 -c 131072 -b 4096 -ub 4096 --jinja --host 127.0.0.1 --port 8080 --alias gemma-4-12b-qat "$@"
EOS
cat > /scratch/app/keep_serving.sh <<'EOS'
#!/usr/bin/env bash
# Watchdog: restart llama-server within 5 s if it dies. Logs on the local disk.
while true; do
  echo "=== start $(date -u +%FT%TZ)" >> /scratch/logs/serve.log
  bash /scratch/app/serve_gemma.sh >> /scratch/logs/serve.log 2>&1
  echo "=== exited $? $(date -u +%FT%TZ), restarting in 5 s" >> /scratch/logs/serve.log
  sleep 5
done
EOS
chmod +x /scratch/app/*.sh
echo "staged: $(du -sh /scratch/models/gemma | cut -f1) model, $(du -sh /scratch/kb | cut -f1) kb; free $(df -h /scratch | awk 'NR==2{print $4}')"
