import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

from .utils import (
    DEFAULT_SYSTEM_PROMPT,
    FALLBACK_TAGS,
    clean_text,
    detect_text_language,
    ensure_parent_dir,
    parse_samples_from_record,
    parse_tag_list,
    pick_column,
    read_csv_with_fallback,
    safe_float,
    write_jsonl_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an SFT JSONL dataset from the structured BOJ CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-description-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--require-samples", dest="require_samples", action="store_true")
    parser.add_argument("--no-require-samples", dest="require_samples", action="store_false")
    parser.set_defaults(require_samples=True)
    parser.add_argument("--language-filter", choices=["ko", "korean", "en", "english"], default=None)
    parser.add_argument("--inspect", action="store_true")
    return parser.parse_args()


def normalize_language_filter(value: str | None) -> str | None:
    if value in {"ko", "korean"}:
        return "ko"
    if value in {"en", "english"}:
        return "en"
    return None


def coerce_limit_text(limit_text: str, time_limit: str, memory_limit: str) -> str:
    parts = []
    if limit_text:
        parts.append(limit_text)
    if time_limit:
        parts.append(f"시간 제한: {time_limit}")
    if memory_limit:
        parts.append(f"메모리 제한: {memory_limit}")
    return "\n".join(parts) if parts else "문제에서 주어진 입력 범위를 따른다."


def build_conditions(record: dict) -> str:
    payload = {
        "task": "generate_algorithm_problem",
        "language": "ko",
        "difficulty": record["difficulty"],
        "tier_value": record["tier_value"],
        "tags": record["tags"],
        "time_limit": record["time_limit"],
        "memory_limit": record["memory_limit"],
        "requirements": [
            "Generate an original Korean programming contest problem.",
            "Keep the objective unambiguous and the output deterministic.",
            "Define all variables, constraints, and input format precisely.",
            "Include at least one valid sample input and sample output pair when possible.",
            "Avoid copying or explicitly referencing existing online judge problems.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_completion(record: dict) -> str:
    parts = [
        f"제목:\n{record['title']}",
        f"문제:\n{record['description']}",
        f"입력:\n{record['input']}",
        f"출력:\n{record['output']}",
        f"제한:\n{record['limit']}",
    ]
    for index, sample in enumerate(record["samples"], start=1):
        parts.append(f"예제 입력 {index}:\n{sample['input']}")
        parts.append(f"예제 출력 {index}:\n{sample['output']}")
    return "\n\n".join(part for part in parts if part.strip()).strip()


def row_core_length(record: dict) -> int:
    keys = ["title", "description", "input", "output", "hint", "limit", "time_limit", "memory_limit"]
    return sum(len(record.get(key, "")) for key in keys)


def detect_columns(columns: list[str]) -> dict[str, str | None]:
    detected = {
        "problem_id": pick_column(columns, ["problem_id", "id", "number", "boj_id"]),
        "title": pick_column(columns, ["title", "problem_title", "name"]),
        "description": pick_column(columns, ["description", "problem", "problem_description", "statement"]),
        "input": pick_column(columns, ["input", "input_description"]),
        "output": pick_column(columns, ["output", "output_description"]),
        "limit": pick_column(columns, ["limit", "constraints", "constraint"]),
        "hint": pick_column(columns, ["hint"]),
        "time_limit": pick_column(columns, ["time_limit", "time"]),
        "memory_limit": pick_column(columns, ["memory_limit", "memory"]),
        "tags": pick_column(columns, ["tags", "tag"]),
        "difficulty": pick_column(columns, ["difficulty", "level", "tier"]),
        "tier_value": pick_column(columns, ["tier_value"]),
    }
    print("Detected columns:")
    for logical_name, actual_name in detected.items():
        print(f"  {logical_name}: {actual_name}")
    return detected


def build_record(raw_row: dict, columns: list[str], detected: dict[str, str | None]) -> dict:
    title = clean_text(raw_row.get(detected["title"], "")) if detected["title"] else ""
    description = clean_text(raw_row.get(detected["description"], "")) if detected["description"] else ""
    input_text = clean_text(raw_row.get(detected["input"], "")) if detected["input"] else ""
    output_text = clean_text(raw_row.get(detected["output"], "")) if detected["output"] else ""
    hint = clean_text(raw_row.get(detected["hint"], "")) if detected["hint"] else ""
    time_limit = clean_text(raw_row.get(detected["time_limit"], "")) if detected["time_limit"] else ""
    memory_limit = clean_text(raw_row.get(detected["memory_limit"], "")) if detected["memory_limit"] else ""
    raw_tags = raw_row.get(detected["tags"], "") if detected["tags"] else ""
    raw_tier_value = raw_row.get(detected["tier_value"], None) if detected["tier_value"] else None
    tier_value = safe_float(raw_tier_value)

    record = {
        "problem_id": clean_text(raw_row.get(detected["problem_id"], "")) if detected["problem_id"] else "",
        "title": title,
        "description": description,
        "input": input_text,
        "output": output_text,
        "hint": hint,
        "time_limit": time_limit,
        "memory_limit": memory_limit,
        "difficulty": clean_text(raw_row.get(detected["difficulty"], "")) if detected["difficulty"] else "",
        "tier_value": int(tier_value) if tier_value is not None and tier_value.is_integer() else tier_value,
        "tags": parse_tag_list(raw_tags) or list(FALLBACK_TAGS),
        "samples": parse_samples_from_record(raw_row, columns),
    }
    limit_text = clean_text(raw_row.get(detected["limit"], "")) if detected["limit"] else ""
    record["limit"] = coerce_limit_text(limit_text, time_limit, memory_limit)
    if not record["difficulty"]:
        record["difficulty"] = "Unknown"
    return record


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    language_filter = normalize_language_filter(args.language_filter)

    dataframe = read_csv_with_fallback(csv_path)
    columns = list(dataframe.columns)
    detected = detect_columns(columns)
    rows = dataframe.to_dict(orient="records")

    rng = random.Random(args.seed)
    if args.sample > 0 and args.sample < len(rows):
        rows = rng.sample(rows, args.sample)

    ensure_parent_dir(out_path)
    written = 0
    skipped = 0
    first_item = None

    with out_path.open("w", encoding="utf-8") as handle:
        for raw_row in tqdm(rows, desc="build sft", total=len(rows)):
            record = build_record(raw_row, columns, detected)

            if not record["problem_id"] or not record["title"]:
                skipped += 1
                continue
            if len(record["description"]) < args.min_description_chars:
                skipped += 1
                continue
            if not record["input"] or not record["output"]:
                skipped += 1
                continue
            if args.max_chars > 0 and row_core_length(record) > args.max_chars:
                skipped += 1
                continue
            if args.require_samples and not record["samples"]:
                skipped += 1
                continue

            detected_language = detect_text_language(f"{record['title']}\n{record['description']}")
            if language_filter is not None and detected_language != language_filter:
                skipped += 1
                continue

            item = {
                "problem_id": record["problem_id"],
                "title": record["title"],
                "difficulty": record["difficulty"],
                "tier_value": record["tier_value"],
                "tags": record["tags"],
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": build_conditions(record)},
                    {"role": "assistant", "content": build_completion(record)},
                ],
            }

            write_jsonl_row(handle, item)
            if first_item is None:
                first_item = item
            written += 1
            if args.limit > 0 and written >= args.limit:
                break

    print(f"Wrote {written} rows to {out_path}")
    print(f"Skipped {skipped} rows")

    if args.inspect and first_item is not None:
        print(json.dumps(first_item, ensure_ascii=False, indent=2))

    if written == 0:
        out_path.unlink(missing_ok=True)
        raise SystemExit(
            "No usable rows were written. Check the CSV columns, filters, sample requirements, and minimum lengths.",
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
