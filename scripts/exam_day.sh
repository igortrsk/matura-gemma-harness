#!/usr/bin/env bash
# EXAM DAY — one command: scripts/exam_day.sh <package.zip | package dir> [NAME]        (default NAME=final-exam)
#  1. unpacks to exams/NAME (question JSON -> exam.json if named differently)
#  2. pre-flight: Gemma server + adapters (essay-v2 = id 0), KB server, SSH tunnel
#  3. base (untouched Gemma) and final setup run AT THE SAME TIME, 2 server slots each, + a resume pass for failed items
#  4. writes + validates runs/NAME-base/answers.json and runs/NAME-final/answers.json (upload these two)
# Optional: ONLY="1 2.1" limits to these item ids (smoke test).
set -u
cd /data
PKG=$1; NAME=${2:-final-exam}; E=exams/$NAME
if [ -f "$PKG" ]; then mkdir -p $E && unzip -o -q "$PKG" -d $E || { echo "unzip failed"; exit 1; }
elif [ -d "$PKG" ]; then [ "$(realpath "$PKG")" = "$(realpath -m $E)" ] || { mkdir -p $E && cp -r "$PKG"/. $E/; }
else echo "no such package: $PKG"; exit 1; fi
# package may be nested one folder deep inside the zip
if [ ! -f $E/answers-template.json ]; then d=$(dirname "$(find $E -name answers-template.json | head -1)"); [ -n "$d" ] && [ "$d" != "." ] && cp -r "$d"/. $E/; fi
if [ ! -f $E/exam.json ]; then
  q=$(find $E -maxdepth 1 -name '*.json' ! -name 'answers-template.json' | head -1)
  [ -n "$q" ] && cp "$q" $E/exam.json || { echo "no question JSON in $E"; exit 1; }
fi
[ -f $E/answers-template.json ] || echo "WARNING: no answers-template.json (write_answers may fail)"
.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); import run_exam as R; it=R.load_package('$E');
import collections; print('package:', len(it), 'items', dict(collections.Counter(i['type'] for i in it)), sum(i['image_count'] for i in it), 'images')" || exit 1

echo "== pre-flight"
curl -s -m 10 localhost:8080/health | grep -q ok || { echo "Gemma server NOT reachable on :8080 (tunnel? server_switch.sh?)"; exit 1; }
curl -s -m 10 localhost:8090/health | grep -q ok || { echo "KB server NOT reachable on :8090 (kb_restart.sh / tunnel)"; exit 1; }
curl -s localhost:8080/lora-adapters | grep -q '"id":0,"path":"[^"]*essay-v2.gguf"' || { echo "essay-v2 is not adapter 0 (run server_switch.sh)"; exit 1; }
pgrep -f "[t]unnel_keeper.sh" >/dev/null || (setsid nohup scripts/tunnel_keeper.sh >/dev/null 2>&1 </dev/null &)
echo "server, adapters, KB, tunnel OK — final flags: $(cat scripts/final_flags.txt)"

O=${ONLY:+--only $ONLY}
run() {  # $1 run, rest = flags
  local r=$1; shift; mkdir -p runs/$r
  for pass in 1 2; do
    PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_exam.py --run $r --exam $E --model gemma-4-12b-qat "$@" $O >> runs/$r/log.txt 2>&1
  done
}
echo "== running base + final ($(date +%H:%M)); logs: runs/$NAME-base/log.txt, runs/$NAME-final/log.txt"
run $NAME-base --think --no-no-think-fallback --parallel 2 &
run $NAME-final $(cat scripts/final_flags.txt) --parallel 2 &
wait
echo "== done $(date +%H:%M)"
for r in $NAME-base $NAME-final; do
  .venv/bin/python scripts/write_answers.py --run $r --exam $E 2>&1 | tail -n 4
done
echo "UPLOAD: solution = /data/runs/$NAME-final/answers.json, base model = /data/runs/$NAME-base/answers.json"
