#!/usr/bin/env bash
# Downloads for fine-tuning (QAT unquantized weights) and the offline knowledge base (plwiki, plwikisource).
set -x
hf download wikimedia/wikipedia --repo-type dataset --include "20231101.pl/*" --local-dir /workspace/kb/raw/wikipedia
hf download wikimedia/wikisource --repo-type dataset --include "20231201.pl/*" --local-dir /workspace/kb/raw/wikisource
hf download google/gemma-4-12B-it-qat-q4_0-unquantized --local-dir /workspace/models/gemma-4-12b-qat-unq
echo DONE $(date)
