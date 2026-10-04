# Matura z historii (rozszerzona) — Gemma 4 12B + harness
**🏆 Winner — Best exam score, Warsaw Model Trainers hackathon (Kolektyw3, 25–27.09.2026)** — 54/60 (90%) on the final exam.

Warsaw Model Trainers hackathon, Kolektyw3, 25–27.09.2026 (see `SOURCE.md`).
Goal: score well on the Polish history matura, extended level (CKE Formuła 2023, 60 pts), with open models only, fully
offline on exam day, all models together ≤ 8.8 GB.

## System

| Part | What | Size |
|---|---|---|
| Base model | `google/gemma-4-12B-it-qat-q4_0-gguf` (Gemma 4 12B, QAT Q4_0, Apache-2.0) + vision projector | 6.98 GB + 0.18 GB |
| Essay adapter | `essay-v2` LoRA (r=16) on the QAT-unquantized weights, used at **scale 0.5 with thinking ON**, essays only | 0.13 GB |
| Knowledge base | offline Polish Wikipedia + Wikisource (HF `wikimedia/wikipedia` 20231101.pl, `wikimedia/wikisource` 20231201.pl), BM25 (tantivy) — data, not a model | 3.9 GB |
| **Total models** | | **≈ 7.3 GB** |

Served by one `llama-server` (llama.cpp); adapters are switched per request, and every request sends explicit
scales for every loaded adapter (llama-server otherwise applies a loaded adapter to requests without a `lora` field).

### Final harness (`scripts/final_run.sh`, flags in `scripts/final_flags.txt`)

`--think --think-budget 4000 --retrieval --lora-for essay=0 --lora-think --lora-scale 0.5`

1. Exam-format prompt (`run_exam.exam_prompt`): sources, "Zadanie (N pkt)", question, the item's `answer_format`;
   pictures sent as images with an `[Obraz: …]` label. Everything model-facing is in Polish.
2. Thinking ON, capped at 4000 extra tokens. Long thinking never paid off (answers >9k thinking tokens: 0/13 points).
3. **No-thinking fallback:** if thinking runs out and the answer is empty, the question is asked again with thinking OFF.
4. **Retrieval** (non-essay): Gemma writes a Wikipedia search query (thinking off) + a names/years query from the
   task text → top 4 passages from the offline KB, marked as possibly off-topic.
5. **Essays:** essay adapter at half strength, thinking ON.

`scripts/base_run.sh` runs the untouched base model (thinking ON, no fallback, no retrieval, adapters off).

## Results 

### Final exam (official) 
**54/60 (90%)** in the organisers' grading — 🏆 Best exam score award.

### Development (2023–2025 papers in the organisers' exam format; 2026 held out, never used)

| Setup | Score |
|---|---|
| Untouched base model (thinking ON) | 123/180 = 68.3% (graded with Claude, `scripts/GRADING.md`) |
| Harness v1 (fallback + think cap 4000 + retrieval), non-essay | 110/135 vs base 99/135 (**+11**; fallback +8, retrieval ≈ +3) |
| Essays, 21 essays (18 older-format topics rewritten to Formuła 2023 + 3 practice essays), GPT-6 graded | base 210 & 208 (two runs) → **essay adapter ×0.5 + thinking 220 & 220** |
| **Final (estimate)** | ≈ 132–133/180 ≈ 73–74% → **≈ +5 pp over the base model** |

### Tried and rejected (measured, same grader on both sides)

| Idea (harness flags to reproduce) | Result |
|---|---|
| Short-answer LoRA `--lora-for open=1` (`short-v2-rag`, 781 synthetic tasks with retrieved passages, thinking OFF) | 67 vs 87 /109 on open questions → worse |
| Essay LoRA at full strength, thinking OFF `--lora-for essay=0` | 86 vs 167 /315 → confident but invented facts |
| Plan-then-write essays `--essay-plan --no-think-types essay` | 165 vs 167 → no gain |
| Gemma describes its own pictures first `--describer-url http://localhost:8080/v1` | 65 vs 67 /90 on picture questions → no gain |
| Zoomed picture halves for large images `--zoom-tiles 900` | 63 vs 67 /90 → worse |
| Detailed CKE essay checklist in the prompt `--essay-guide` | 214 vs 210 → within noise |
| Qwen3.5-9B picture describer | read small text better, but breaks the 8.8 GB total limit |

### Essay adapter recipe (exactly what produced `essay-v2`)

1. Data: `run_synthetic_v1.sh` and `run_synthetic_v2.sh` (plus two small earlier test runs of `generate_synthetic.py`:
   `synthetic/pilot`, `synthetic/test2`), then
   `build_sft.py --name v2-essay --kinds essay synthetic/pilot/essays.jsonl synthetic/test2/essays_new.jsonl synthetic/v1/essays_new.jsonl synthetic/v2/essays_new.jsonl`
   → 937 training + 49 validation essays (exam format: three topics, answer starts with "Temat nr N."; essays on the
   2023–25 practice topics are excluded).
2. Training: `ft_run.sh essay-v2 v2-essay 2` → QLoRA (NF4) on `google/gemma-4-12B-it-qat-q4_0-unquantized`, LoRA r=16,
   α=32, lr 1e-4, 2 epochs (235 steps), loss on the answer only, thinking-OFF chat format; validation loss 2.28 → 1.24.
   Converted with llama.cpp `convert_lora_to_gguf.py` (f16) → `essay-v2.gguf` (release v1).
3. Serving only what the final setup needs:
   `llama-server -m gemma-4-12b-it-qat-q4_0.gguf --mmproj mmproj-gemma-4-12b-it-qat-q4_0.gguf -ngl 999 -np 4 -c 131072 -b 4096 -ub 4096 --jinja --lora essay-v2.gguf --lora-init-without-apply`
   (adapter id 0). Our `server_switch.sh` additionally loads the unused short-answer adapter as id 1.

### Evaluation sets

The 2023–2025 CKE papers are fetched by `fetch_cke.sh` and converted with `build_practice_set.py` / `build_package.py`.
The 18-essay test set was assembled from older-format (2015–2022) CKE essay topics and marking rules
(`build_practice_set.py --old`), rewritten into the Formuła 2023 three-topic format; it is not redistributed (CKE content).

## Exam day (exactly what we run)

```
scripts/exam_day.sh <final-package.zip> final-exam
```
Unpacks the organisers' package, checks the model server (adapter 0 = essay-v2), the KB server and the tunnel, runs the
**untouched base model** (`--think --no-no-think-fallback`) and the **final setup** (`scripts/final_flags.txt`) at the same
time, and writes the two validated files to upload: `runs/final-exam-final/answers.json` (solution) and
`runs/final-exam-base/answers.json` (base model answers). Dry run on the 2023 mock: base 43/60, final 48/60 (GPT-6
graded), ~35 min for both.

## Reproduce

Scripts run from a data directory (`DATA_DIR`, default `/data`); GPU box: 1× L40S (Forgehand `gpu-l40s-small`).

1. `scripts/fetch_cke.sh` — downloads the CKE papers and marking guides (not redistributed: copyrighted).
2. `scripts/build_practice_set.py`, `scripts/build_package.py --year Y` — parse papers into items and exam packages
   (2026 refused unless `ALLOW_HELDOUT=1`).
3. GPU box: `scripts/setup_gpu.sh` (llama.cpp + Gemma), `scripts/setup_ft.sh` + `scripts/download_ft_kb.sh`
   (training env, unquantized weights, Wikipedia/Wikisource), `kb.py build` (offline index).
4. Synthetic training data (GPT-6 via the Forgehand team API; Polish): `run_synthetic_v1.sh` and `run_synthetic_v2.sh`
   (the exact commands; they call `fetch_sources.py` and `generate_synthetic.py`), then `build_sft.py` (`--retrieval` for passages). Train: `ft_run.sh NAME SFT EPOCHS` (`train_lora.py`, QLoRA NF4,
   loss on answers only) → GGUF adapter.
5. Serve: `boot_gpu.sh` (stages model + KB to local disk), `server_switch.sh` (llama-server with both adapters),
   `kb_restart.sh` (KB service on :8090).
6. Exam: `final_run.sh RUN EXAM_DIR` and `base_run.sh RUN EXAM_DIR`, then
   `write_answers.py --run RUN --exam EXAM_DIR` → `runs/RUN/answers.json`.
7. Grading: `grade_run.sh` (Claude CLI, `GRADING.md`) or `grade_gpt.py` (GPT-6 via Forgehand).

## Data and licences

- Code (scripts, harness, prompts): MIT licence, see `LICENSE`. Team: MakeNoMistakes.
- The essay adapter follows the base model's licence (Gemma, Apache-2.0).
- CKE exam papers and marking guides are fetched by script and never committed.
- Wikipedia/Wikisource-derived data (KB, synthetic tasks built on excerpts) is CC BY-SA 4.0; every passage keeps its
  source URL and revision.
- Gemma: Apache-2.0. Synthetic data was generated with GPT-6 (Forgehand hackathon credits) for training only.
