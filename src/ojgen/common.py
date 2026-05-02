import json
import re
from pathlib import Path
from typing import Any, Iterable

SECTION_KEYS = [
    "description", "input", "output", "limit", "hint", "source"
]

DIFFICULTY_BUCKETS = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Ruby"]

TAG_ALIASES = {
    "dp": "dynamic_programming",
    "bfs": "breadth_first_search",
    "dfs": "depth_first_search",
    "graph": "graph_traversal",
    "math": "mathematics",
    "implementation": "implementation",
    "greedy": "greedy",
    "string": "string",
    "binary_search": "binary_search",
    "dijkstra": "dijkstra",
}

def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def read_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def tier_to_bucket(tier_value: int | None) -> str:
    if tier_value is None or tier_value <= 0:
        return "Unknown"
    if tier_value <= 5:
        return "Bronze"
    if tier_value <= 10:
        return "Silver"
    if tier_value <= 15:
        return "Gold"
    if tier_value <= 20:
        return "Platinum"
    if tier_value <= 25:
        return "Diamond"
    return "Ruby"
