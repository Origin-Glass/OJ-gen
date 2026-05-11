#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m ojgen.train_sft \
  --data data/sft/full.jsonl \
  --out outputs/qwen3-8b-ojgen-a40-v1 \
  --model unsloth/Qwen3-8B-unsloth-bnb-4bit \
  --epochs 1 \
  --max-seq-length 2048 \
  --batch-size 2 \
  --grad-accum 8 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.0 \
  --target-modules all \
  --packing
