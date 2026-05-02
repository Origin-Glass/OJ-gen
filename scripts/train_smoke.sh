#!/usr/bin/env bash
set -euo pipefail

python -m ojgen.build_sft_from_csv \
  --csv data/raw/boj_problems_structured.csv \
  --out data/sft/smoke_50.jsonl \
  --limit 50

python -m ojgen.train_sft \
  --data data/sft/smoke_50.jsonl \
  --out outputs/qwen3-8b-smoke \
  --epochs 1 \
  --max-seq-length 1024 \
  --batch-size 1 \
  --grad-accum 4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --target-modules qv \
  --no-packing
