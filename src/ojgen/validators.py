import argparse
import json
import re
from pathlib import Path
from typing import Any

from .utils import clean_text, ensure_parent_dir

REQUIRED_SECTION_PATTERNS = [
    r"제목\s*:",
    r"문제\s*:",
    r"입력\s*:",
    r"출력\s*:",
    r"제한\s*:",
    r"예제 입력\s*1?\s*:",
    r"예제 출력\s*1?\s*:",
]

FORBIDDEN_LEAKAGE_TERMS = ["acmicpc", "BOJ", "백준", "Baekjoon"]


def has_required_sections(text: str) -> tuple[float, list[str]]:
    missing = []
    for pattern in REQUIRED_SECTION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE) is None:
            missing.append(pattern.replace(r"\s*", " ").replace("?", ""))
    score = (len(REQUIRED_SECTION_PATTERNS) - len(missing)) / len(REQUIRED_SECTION_PATTERNS)
    return round(score, 4), missing


def extract_section(text: str, section_name: str) -> str:
    pattern = re.compile(
        rf"(^|\n)\s*{re.escape(section_name)}\s*:\s*(.*?)(?=(\n\s*(제목|문제|입력|출력|제한|예제 입력 \d+|예제 출력 \d+)\s*:)|\Z)",
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    return clean_text(match.group(2)) if match else ""


def validate_generated_problem(text: str) -> dict[str, Any]:
    normalized = clean_text(text)
    issues: list[str] = []

    schema_score, missing_sections = has_required_sections(normalized)
    if missing_sections:
        issues.append("Missing required sections.")

    body_length = len(normalized)
    statement_length = len(extract_section(normalized, "문제"))
    if body_length >= 300 and statement_length >= 80:
        length_score = 1.0
    elif body_length >= 180 and statement_length >= 50:
        length_score = 0.5
        issues.append("Problem text is shorter than expected.")
    else:
        length_score = 0.0
        issues.append("Problem text is too short.")

    has_sample_input = bool(re.search(r"예제 입력\s*\d*\s*:\s*\S", normalized))
    has_sample_output = bool(re.search(r"예제 출력\s*\d*\s*:\s*\S", normalized))
    sample_score = 1.0 if has_sample_input and has_sample_output else 0.0
    if sample_score == 0.0:
        issues.append("Sample input/output is missing.")

    leaked_terms = [term for term in FORBIDDEN_LEAKAGE_TERMS if term.lower() in normalized.lower()]
    leakage_score = 1.0 if not leaked_terms else 0.0
    if leaked_terms:
        issues.append("Source leakage detected: " + ", ".join(leaked_terms))

    overall_score = round(
        0.35 * schema_score + 0.25 * length_score + 0.20 * sample_score + 0.20 * leakage_score,
        4,
    )

    return {
        "schema_score": round(schema_score, 4),
        "length_score": round(length_score, 4),
        "sample_score": round(sample_score, 4),
        "leakage_score": round(leakage_score, 4),
        "overall_score": overall_score,
        "issues": issues,
    }


def iter_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{"):
                payload = json.loads(stripped)
                text = payload.get("generated_text") or payload.get("output") or payload.get("text") or ""
                rows.append({**payload, "validation": validate_generated_problem(text)})
            else:
                rows.append({"generated_text": stripped, "validation": validate_generated_problem(stripped)})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated contest problem candidates.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.out)
    rows = iter_rows(input_path)
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} validated rows to {output_path}")


basic_validate = validate_generated_problem


if __name__ == "__main__":
    main()
