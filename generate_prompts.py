import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ojgen.difficulty import SOLVED_AC_BUCKETS, bucket_tier_values

TAG_POOLS = [
    ["implementation", "mathematics"],
    ["graph_traversal", "breadth_first_search", "depth_first_search"],
    ["dynamic_programming"],
    ["greedy"],
    ["string", "data_structures"],
    ["sorting", "binary_search"],
    ["geometry", "bruteforcing"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate balanced solved.ac-style prompt JSONL.")
    parser.add_argument("--out", default="data/prompts_50k.jsonl")
    parser.add_argument("--total", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--time-limit", default="2 seconds")
    parser.add_argument("--memory-limit", default="256 MB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_count, remainder = divmod(args.total, len(SOLVED_AC_BUCKETS))
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for bucket_index, bucket in enumerate(SOLVED_AC_BUCKETS):
            count = base_count + (1 if bucket_index < remainder else 0)
            tier_values = bucket_tier_values(bucket)
            for _ in range(count):
                tier_value = rng.choice(tier_values)
                row = {
                    "difficulty": bucket,
                    "tier_value": tier_value,
                    "tags": rng.choice(TAG_POOLS),
                    "time_limit": args.time_limit,
                    "memory_limit": args.memory_limit,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Generated {written} prompts at {out_path}")
    print(f"Difficulty buckets: {', '.join(SOLVED_AC_BUCKETS)}")


if __name__ == "__main__":
    main()
