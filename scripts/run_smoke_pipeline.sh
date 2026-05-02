#!/usr/bin/env bash
set -euo pipefail
ZIP_PATH=${1:-/mnt/data/OJ.zip}
OUT_DIR=${2:-runs/smoke}
mkdir -p "$OUT_DIR"
python -m ojgen.parse_zip --zip "$ZIP_PATH" --out-dir "$OUT_DIR/parsed" --limit 50
python -m ojgen.build_sft_dataset \
  --problems "$OUT_DIR/parsed/problems.jsonl" \
  --samples "$OUT_DIR/parsed/samples.jsonl" \
  --out "$OUT_DIR/sft_smoke.jsonl" \
  --limit 50
python - <<'PY'
import json
from pathlib import Path
p=Path('runs/smoke/prompts_eval.jsonl')
p.parent.mkdir(parents=True, exist_ok=True)
items=[
 {'difficulty':'Bronze 2','tags':['implementation'],'time_limit':'1 second','memory_limit':'128 MB'},
 {'difficulty':'Silver 3','tags':['bfs','graph'],'time_limit':'2 seconds','memory_limit':'256 MB'},
 {'difficulty':'Gold 4','tags':['dynamic_programming'],'time_limit':'2 seconds','memory_limit':'512 MB'},
]
with p.open('w', encoding='utf-8') as f:
 for x in items:
  f.write(json.dumps(x, ensure_ascii=False)+'\n')
print(p)
PY
