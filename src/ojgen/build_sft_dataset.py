import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import pandas as pd
from .common import clean_text, read_jsonl, tier_to_bucket, write_jsonl

SYSTEM_PROMPT = (
    "You are an algorithm problem setter. Generate original Korean programming contest "
    "problems with precise input/output specifications. Do not copy or lightly paraphrase existing problems."
)

FALLBACK_TAGS = ["implementation"]


def load_enrichment(path: str | None) -> dict[int, dict]:
    if not path:
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        pid = int(r["problem_id"])
        tags = r.get("tags", "")
        if isinstance(tags, str):
            tags = [x.strip() for x in tags.replace(";", ",").split(",") if x.strip()]
        tier_value = None
        if "tier_value" in r and pd.notna(r["tier_value"]):
            tier_value = int(r["tier_value"])
        bucket = r.get("difficulty", None)
        if not isinstance(bucket, str) or not bucket:
            bucket = tier_to_bucket(tier_value)
        out[pid] = {"tier_value": tier_value, "difficulty": bucket, "tags": tags or FALLBACK_TAGS}
    return out


def load_samples(path: str | None) -> dict[int, list[dict]]:
    if not path or not Path(path).exists():
        return {}
    rows = list(read_jsonl(path)) if path.endswith(".jsonl") else pd.read_csv(path).to_dict("records")
    mp = defaultdict(list)
    for r in rows:
        try:
            pid = int(r["problem_id"])
            mp[pid].append({
                "sample_index": int(r.get("sample_index", len(mp[pid]) + 1)),
                "sample_input": clean_text(r.get("sample_input", "")),
                "sample_output": clean_text(r.get("sample_output", "")),
            })
        except Exception:
            continue
    return dict(mp)


def make_user_prompt(problem: dict, difficulty: str, tier_value: int | None, tags: list[str]) -> str:
    return f"""다음 조건을 만족하는 새로운 알고리즘 문제를 작성하라.

난이도: {difficulty}
예상 티어 값: {tier_value if tier_value is not None else 'unknown'}
알고리즘 태그: {', '.join(tags)}
시간 제한: {problem.get('time_limit', '')}
메모리 제한: {problem.get('memory_limit', '')}

요구사항:
1. 기존 온라인 저지 문제를 복사하거나 가볍게 변형하지 말 것.
2. 문제의 목표는 하나로 명확해야 한다.
3. 입력의 모든 변수는 범위와 의미가 정의되어야 한다.
4. 출력 형식은 결정적이어야 한다.
5. 예제 입력과 예제 출력은 실제로 일치해야 한다.
6. 제약은 요구 알고리즘 난이도와 맞아야 한다.
7. 한국어 온라인 저지 문제 스타일을 유지할 것.

반드시 다음 형식으로 출력하라:
제목:
문제:
입력:
출력:
제한:
예제 입력 1:
예제 출력 1:
예제 설명:"""


def make_assistant(problem: dict, samples: list[dict]) -> str:
    parts = [
        f"제목:\n{clean_text(problem.get('title', ''))}",
        f"문제:\n{clean_text(problem.get('description', ''))}",
        f"입력:\n{clean_text(problem.get('input', ''))}",
        f"출력:\n{clean_text(problem.get('output', ''))}",
        f"제한:\n{clean_text(problem.get('limit', ''))}",
    ]
    for s in samples[:3]:
        idx = int(s.get("sample_index", len(parts)))
        parts.append(f"예제 입력 {idx}:\n{s.get('sample_input', '')}")
        parts.append(f"예제 출력 {idx}:\n{s.get('sample_output', '')}")
    hint = clean_text(problem.get("hint", ""))
    parts.append(f"예제 설명:\n{hint}" if hint else "예제 설명:\n")
    return "\n\n".join(parts).strip()


def keep_problem(p: dict) -> bool:
    if p.get("parse_error"):
        return False
    if len(clean_text(p.get("description", ""))) < 40:
        return False
    if not clean_text(p.get("input", "")) or not clean_text(p.get("output", "")):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--enrichment", default=None, help="CSV columns: problem_id,tier_value,difficulty,tags")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    random.seed(args.seed)
    enrichment = load_enrichment(args.enrichment)
    samples = load_samples(args.samples)
    problems = list(read_jsonl(args.problems)) if args.problems.endswith(".jsonl") else pd.read_csv(args.problems).to_dict("records")
    rows = []
    for p in problems:
        if not keep_problem(p):
            continue
        pid = int(p["problem_id"])
        e = enrichment.get(pid, {"tier_value": None, "difficulty": "Unknown", "tags": FALLBACK_TAGS})
        ss = samples.get(pid, [])
        row = {
            "problem_id": pid,
            "difficulty": e["difficulty"],
            "tier_value": e["tier_value"],
            "tags": e["tags"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_prompt(p, e["difficulty"], e["tier_value"], e["tags"])},
                {"role": "assistant", "content": make_assistant(p, ss)},
            ],
        }
        rows.append(row)
    random.shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} SFT rows to {args.out}")

if __name__ == "__main__":
    main()
