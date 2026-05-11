import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import clean_text
from .difficulty import SOLVED_AC_BUCKETS, normalize_difficulty
from .validators import validate_generated_problem


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package accepted generations as balanced Markdown files by difficulty.")
    parser.add_argument("--evaluated", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", default=None)
    parser.add_argument("--per-difficulty", type=int, default=0)
    parser.add_argument("--min-local-score", type=float, default=0.9)
    parser.add_argument("--min-llm-score", type=float, default=0.85)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--require-llm-eval", action="store_true")
    return parser.parse_args()


def get_generated_text(row: dict[str, Any]) -> str:
    return clean_text(row.get("generated_text") or row.get("output") or row.get("text") or "")


def get_local_score(row: dict[str, Any], text: str) -> float:
    validation = row.get("validation")
    if not isinstance(validation, dict):
        validation = validate_generated_problem(text)
        row["validation"] = validation
    try:
        return float(validation.get("overall_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def get_llm_quality(row: dict[str, Any]) -> tuple[float, float]:
    llm_eval = row.get("llm_eval")
    if not isinstance(llm_eval, dict):
        return 0.0, 0.0
    quality_keys = [
        "schema_score",
        "sample_consistency_score",
        "constraint_score",
        "originality_score",
    ]
    scores = []
    for key in quality_keys:
        try:
            scores.append(float(llm_eval.get(key, 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)
    confidence = float(llm_eval.get("confidence", 0.0) or 0.0)
    return sum(scores) / len(scores), confidence


def resolve_difficulty(row: dict[str, Any]) -> str:
    llm_eval = row.get("llm_eval")
    if isinstance(llm_eval, dict):
        difficulty = normalize_difficulty(llm_eval.get("difficulty"), llm_eval.get("tier_value"))
        if difficulty != "Unknown":
            return difficulty
    prompt = row.get("prompt")
    if isinstance(prompt, dict):
        return normalize_difficulty(prompt.get("difficulty"), prompt.get("tier_value"))
    return normalize_difficulty(row.get("difficulty"), row.get("tier_value"))


def accepted(row: dict[str, Any], args: argparse.Namespace) -> bool:
    text = get_generated_text(row)
    if not text:
        return False
    if get_local_score(row, text) < args.min_local_score:
        return False
    llm_quality, confidence = get_llm_quality(row)
    if args.require_llm_eval and "llm_eval" not in row:
        return False
    if "llm_eval" in row and (llm_quality < args.min_llm_score or confidence < args.min_confidence):
        return False
    return resolve_difficulty(row) in SOLVED_AC_BUCKETS


def extract_title(text: str) -> str:
    match = re.search(r"(^|\n)\s*제목\s*:\s*(.+?)(\n|$)", text)
    if not match:
        return "untitled"
    return clean_text(match.group(2)) or "untitled"


def safe_filename(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", value).strip("._-")
    return cleaned[:80] or fallback


def render_markdown(row: dict[str, Any], index: int, difficulty: str) -> str:
    text = get_generated_text(row)
    title = extract_title(text)
    prompt = row.get("prompt") if isinstance(row.get("prompt"), dict) else {}
    tier_value = None
    llm_eval = row.get("llm_eval")
    if isinstance(llm_eval, dict):
        tier_value = llm_eval.get("tier_value")
    if tier_value is None:
        tier_value = prompt.get("tier_value")
    header = [
        "---",
        f"id: {index}",
        f"title: {json.dumps(title, ensure_ascii=False)}",
        f"difficulty: {difficulty}",
        f"tier_value: {tier_value if tier_value is not None else ''}",
        f"tags: {json.dumps(prompt.get('tags', []), ensure_ascii=False)}",
        "---",
        "",
    ]
    return "\n".join(header) + text.strip() + "\n"


def write_zip(source_dir: Path, zip_path: Path) -> None:
    ensure_parent_dir(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(source_dir.rglob("*.md")):
            archive.write(file_path, file_path.relative_to(source_dir))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    with Path(args.evaluated).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            total += 1
            if accepted(row, args):
                buckets[resolve_difficulty(row)].append(row)

    available = {bucket: len(buckets.get(bucket, [])) for bucket in SOLVED_AC_BUCKETS}
    if args.per_difficulty > 0:
        per_difficulty = args.per_difficulty
    else:
        per_difficulty = min(available.values()) if available else 0
    if per_difficulty <= 0:
        raise SystemExit(f"No balanced output can be produced. Accepted counts: {available}")

    written = 0
    manifest = []
    for bucket in SOLVED_AC_BUCKETS:
        bucket_dir = out_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for offset, row in enumerate(buckets[bucket][:per_difficulty], start=1):
            index = written + 1
            text = get_generated_text(row)
            title = extract_title(text)
            filename = f"{offset:05d}_{safe_filename(title, f'problem_{offset}')}.md"
            path = bucket_dir / filename
            path.write_text(render_markdown(row, index, bucket), encoding="utf-8")
            manifest.append(
                {
                    "id": index,
                    "difficulty": bucket,
                    "title": title,
                    "path": str(path.relative_to(out_dir)),
                }
            )
            written += 1

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.zip:
        write_zip(out_dir, Path(args.zip))

    print(f"Read {total} rows")
    print(f"Accepted counts: {available}")
    print(f"Wrote {written} Markdown files to {out_dir}")
    if args.zip:
        print(f"Wrote zip to {args.zip}")


if __name__ == "__main__":
    main()
