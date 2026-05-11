import argparse
import json
from pathlib import Path
from typing import Any

from .difficulty import normalize_difficulty
from .package_md import accepted, get_generated_text, resolve_difficulty

DEFAULT_SYSTEM_PROMPT = (
    "You are an algorithm problem setter. Generate original Korean programming contest "
    "problems with precise input/output specifications."
)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert accepted evaluated generations back into SFT JSONL rows.")
    parser.add_argument("--evaluated", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-local-score", type=float, default=0.9)
    parser.add_argument("--min-llm-score", type=float, default=0.85)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--require-llm-eval", action="store_true")
    return parser.parse_args()


def build_user_content(row: dict[str, Any], difficulty: str) -> str:
    prompt = row.get("prompt") if isinstance(row.get("prompt"), dict) else {}
    tier_value = None
    llm_eval = row.get("llm_eval")
    if isinstance(llm_eval, dict):
        tier_value = llm_eval.get("tier_value")
    if tier_value is None:
        tier_value = prompt.get("tier_value")
    payload = {
        "task": "generate_algorithm_problem",
        "language": "ko",
        "difficulty": difficulty,
        "tier_value": tier_value,
        "tags": prompt.get("tags") or ["implementation"],
        "time_limit": prompt.get("time_limit") or "2 seconds",
        "memory_limit": prompt.get("memory_limit") or "256 MB",
        "requirements": [
            "Generate an original Korean programming contest problem.",
            "Keep the objective unambiguous and the output deterministic.",
            "Define all variables, constraints, and input format precisely.",
            "Include at least one valid sample input and sample output pair.",
            "Avoid copying or explicitly referencing existing online judge problems.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    ensure_parent_dir(out_path)

    written = 0
    with Path(args.evaluated).open("r", encoding="utf-8") as source, out_path.open("w", encoding="utf-8") as target:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not accepted(row, args):
                continue
            text = get_generated_text(row)
            difficulty = resolve_difficulty(row)
            llm_eval = row.get("llm_eval") if isinstance(row.get("llm_eval"), dict) else {}
            tier_value = llm_eval.get("tier_value")
            prompt = row.get("prompt") if isinstance(row.get("prompt"), dict) else {}
            if tier_value is None:
                tier_value = prompt.get("tier_value")
            item = {
                "problem_id": f"generated-{written + 1}",
                "difficulty": normalize_difficulty(difficulty, tier_value),
                "tier_value": tier_value,
                "tags": prompt.get("tags") or ["implementation"],
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_content(row, difficulty)},
                    {"role": "assistant", "content": text},
                ],
            }
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} distilled SFT rows to {out_path}")


if __name__ == "__main__":
    main()
