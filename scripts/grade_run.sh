#!/usr/bin/env bash
# Grade harness runs non-interactively: one Claude Code CLI process (claude -p) per run, following
# scripts/GRADING.md. Runs are graded in parallel.
#   scripts/grade_run.sh exam-base-2023 exam-base-2024 exam-base-2025
# Writes runs/<run>/grades.jsonl + grade_summary.md (+ grade.log), then checks completeness.
# Runs on the machine with the data (needs a logged-in claude CLI).
cd /data
MODEL=${GRADER_MODEL:-claude-opus-5-5}

grade() {
  local run=$1
  if [ ! -s runs/$run/answers.jsonl ]; then echo "$run: no answers.jsonl" >&2; return 1; fi
  rm -f runs/$run/grades.jsonl runs/$run/grade_summary.md
  printf '%s' "Oceń uruchomienie RUN=$run dokładnie według instrukcji /data/scripts/GRADING.md (przeczytaj ją najpierw w całości). \
Konfiguracja uruchomienia: /data/runs/$run/config.json (rok z key_year; jeśli brak, z years). Oceń WSZYSTKIE zadania z answers.jsonl \
tego uruchomienia (także puste odpowiedzi = 0 pkt). Słowa w wypracowaniu policz poleceniem python3. Zapisz /data/runs/$run/grades.jsonl \
i /data/runs/$run/grade_summary.md w formacie z instrukcji. Na końcu wypisz tylko tabelę wyników." |
  claude -p --model "$MODEL" --allowedTools "Read,Write,Bash(python3:*),Bash(wc:*),Bash(ls:*)" > runs/$run/grade.log 2>&1
  python3 - "$run" <<'EOF'
import json, sys
run = sys.argv[1]
ans = {json.loads(l)["uid"] for l in open(f"/data/runs/{run}/answers.jsonl")}
auto = {json.loads(l)["uid"]: json.loads(l)["auto_points"] for l in open(f"/data/runs/{run}/auto_scores.jsonl")}
try:
    g = {json.loads(l)["uid"]: json.loads(l) for l in open(f"/data/runs/{run}/grades.jsonl")}
except FileNotFoundError:
    print(f"{run}: GRADING FAILED (no grades.jsonl) — see runs/{run}/grade.log"); sys.exit(1)
missing = sorted(ans - set(g))
changed = [u for u, p in auto.items() if p is not None and u in g and g[u]["points"] != p]
bad = [u for u, r in g.items() if not 0 <= r["points"] <= r["max"]]
pts, mx = sum(r["points"] for r in g.values()), sum(r["max"] for r in g.values())
print(f"{run}: {pts}/{mx} ({100 * pts / mx:.1f}%) | graded {len(g)}/{len(ans)} | missing {missing} | "
      f"auto changed {changed} | out of range {bad}")
EOF
}

for run in "$@"; do grade "$run" & done
wait
