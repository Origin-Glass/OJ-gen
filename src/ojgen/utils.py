import ast
import json
import re
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

DEFAULT_SYSTEM_PROMPT = (
    "You are an algorithm problem setter. Generate original Korean programming contest "
    "problems with precise input/output specifications."
)

FALLBACK_TAGS = ["implementation"]


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def pick_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        match = normalized.get(normalize_column_name(candidate))
        if match is not None:
            return match
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv_with_fallback(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="utf-8-sig")


def parse_possible_list(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None

    if isinstance(parsed, (list, tuple)):
        return [clean_text(item) for item in parsed if clean_text(item)]
    if isinstance(parsed, dict):
        return [clean_text(item) for item in parsed.values() if clean_text(item)]
    if ";" in text:
        return [clean_text(part) for part in text.split(";") if clean_text(part)]
    if "," in text:
        return [clean_text(part) for part in text.split(",") if clean_text(part)]
    return [text]


def parse_tag_list(value: Any) -> list[str]:
    return [tag for tag in parse_possible_list(value) if tag]


def parse_mixed_samples(value: Any) -> list[dict[str, str]]:
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []

    samples = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                samples.append(
                    {
                        "input": clean_text(item.get("input") or item.get("sample_input", "")),
                        "output": clean_text(item.get("output") or item.get("sample_output", "")),
                    }
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                samples.append({"input": clean_text(item[0]), "output": clean_text(item[1])})
    return samples


def parse_samples_from_record(record: dict[str, Any], columns: list[str]) -> list[dict[str, str]]:
    inputs: dict[int, str] = {}
    outputs: dict[int, str] = {}

    for column in columns:
        normalized = normalize_column_name(column)
        value = clean_text(record.get(column, ""))
        if not value:
            continue

        input_match = re.fullmatch(r"(sample|example)_?input_?(\d*)", normalized)
        output_match = re.fullmatch(r"(sample|example)_?output_?(\d*)", normalized)
        if input_match:
            index = int(input_match.group(2) or "1")
            inputs[index] = value
            continue
        if output_match:
            index = int(output_match.group(2) or "1")
            outputs[index] = value
            continue

        if normalized in {"sample_inputs", "example_inputs"}:
            for index, item in enumerate(parse_possible_list(value), start=1):
                inputs[index] = clean_text(item)
        elif normalized in {"sample_outputs", "example_outputs"}:
            for index, item in enumerate(parse_possible_list(value), start=1):
                outputs[index] = clean_text(item)
        elif normalized in {"samples", "examples", "sample_io"}:
            for index, item in enumerate(parse_mixed_samples(value), start=1):
                if item["input"]:
                    inputs[index] = item["input"]
                if item["output"]:
                    outputs[index] = item["output"]

    sample_pairs = []
    for index in sorted(set(inputs) | set(outputs)):
        sample_input = clean_text(inputs.get(index, ""))
        sample_output = clean_text(outputs.get(index, ""))
        if sample_input or sample_output:
            sample_pairs.append({"input": sample_input, "output": sample_output})
    return sample_pairs


def detect_text_language(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return "unknown"
    hangul_count = len(re.findall(r"[가-힣]", cleaned))
    latin_count = len(re.findall(r"[A-Za-z]", cleaned))
    if hangul_count == 0 and latin_count == 0:
        return "unknown"
    if hangul_count >= latin_count:
        return "ko"
    return "en"


def write_jsonl_row(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
