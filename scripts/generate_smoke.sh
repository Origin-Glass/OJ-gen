#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/sample

python - <<'PY'
import json
from pathlib import Path

path = Path("data/sample/prompts_smoke.jsonl")
rows = [
    {
        "difficulty": "Silver 3",
        "tier_value": 8,
        "tags": ["bfs", "graph"],
        "time_limit": "2 seconds",
        "memory_limit": "256 MB",
    },
    {
        "difficulty": "Gold 4",
        "tier_value": 12,
        "tags": ["dynamic_programming"],
        "time_limit": "2 seconds",
        "memory_limit": "512 MB",
    },
]
with path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(path)
PY

python -m ojgen.generate \
  --adapter outputs/qwen3-8b-smoke \
  --base-model unsloth/Qwen3-8B-unsloth-bnb-4bit \
  --prompts data/sample/prompts_smoke.jsonl \
  --out outputs/generated_smoke.jsonl \
  --max-seq-length 2048 \
  --max-new-tokens 1024 \
  --temperature 0.8 \
  --top-p 0.95 \
  --num-return-sequences 1
