import argparse
import json
import re
from pathlib import Path
from typing import Any

from .common import clean_text
from .difficulty import normalize_difficulty, parse_tier_value
from .validators import validate_generated_problem


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


EVAL_SYSTEM_PROMPT = (
    "You are a strict programming contest problem reviewer. "
    "Evaluate generated Korean online judge problems against solved.ac difficulty buckets."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM self-evaluation on generated problems.")
    parser.add_argument("--generated", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def get_generated_text(row: dict[str, Any]) -> str:
    return clean_text(row.get("generated_text") or row.get("output") or row.get("text") or "")


def build_eval_prompt(problem_text: str) -> str:
    return f"""다음 생성된 알고리즘 문제를 엄격하게 평가하라.

solved.ac 난이도 버킷은 Bronze, Silver, Gold, Platinum, Diamond, Ruby만 사용한다.
Master는 문제 난이도 버킷으로 사용하지 않는다.

다음 JSON만 출력하라:
{{
  "difficulty": "Bronze|Silver|Gold|Platinum|Diamond|Ruby",
  "tier_value": 1,
  "schema_score": 0.0,
  "sample_consistency_score": 0.0,
  "constraint_score": 0.0,
  "originality_score": 0.0,
  "confidence": 0.0,
  "issues": ["short reason"]
}}

평가 기준:
- schema_score: 제목/문제/입력/출력/제한/예제 입력/예제 출력 형식 완성도
- sample_consistency_score: 예제 입출력이 문제 설명과 모순 없이 성립할 가능성
- constraint_score: 제약이 난이도와 알고리즘 요구에 맞는 정도
- originality_score: 기존 문제명, BOJ, 백준, acmicpc 등을 드러내지 않고 독창적인 정도
- confidence: 위 평가를 신뢰할 수 있는 정도

문제:
{problem_text}
"""


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def normalize_eval(raw_eval: dict[str, Any]) -> dict[str, Any]:
    tier_value = parse_tier_value(raw_eval.get("tier_value"))
    difficulty = normalize_difficulty(raw_eval.get("difficulty"), tier_value)
    issues = raw_eval.get("issues")
    if not isinstance(issues, list):
        issues = [clean_text(issues)] if clean_text(issues) else []
    return {
        "difficulty": difficulty,
        "tier_value": tier_value,
        "schema_score": clamp_score(raw_eval.get("schema_score")),
        "sample_consistency_score": clamp_score(raw_eval.get("sample_consistency_score")),
        "constraint_score": clamp_score(raw_eval.get("constraint_score")),
        "originality_score": clamp_score(raw_eval.get("originality_score")),
        "confidence": clamp_score(raw_eval.get("confidence")),
        "issues": [clean_text(issue) for issue in issues if clean_text(issue)],
    }


def main() -> None:
    args = parse_args()
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("vllm is required for LLM self-evaluation. Install vllm or skip this step.") from exc

    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    rows = []
    prompts = []
    with Path(args.generated).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            text = get_generated_text(row)
            messages = [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": build_eval_prompt(text)},
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            rows.append(row)

    outputs = llm.generate(prompts, sampling_params)
    out_path = Path(args.out)
    ensure_parent_dir(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        for row, output in zip(rows, outputs):
            text = get_generated_text(row)
            raw_eval = extract_json(output.outputs[0].text)
            llm_eval = normalize_eval(raw_eval)
            validation = validate_generated_problem(text)
            payload = {
                **row,
                "validation": validation,
                "llm_eval": llm_eval,
                "judge_model": args.judge_model,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} self-evaluated rows to {out_path}")


if __name__ == "__main__":
    main()
