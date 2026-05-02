#!/usr/bin/env bash
set -euo pipefail

python -m ojgen.build_sft_from_csv \
  --csv data/raw/boj_problems_structured.csv \
  --out data/sft/full.jsonl \
  --no-require-samples
