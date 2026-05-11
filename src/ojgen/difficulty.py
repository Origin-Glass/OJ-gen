import re
from typing import Any

SOLVED_AC_BUCKETS = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Ruby"]

BUCKET_ALIASES = {
    "bronze": "Bronze",
    "브론즈": "Bronze",
    "silver": "Silver",
    "실버": "Silver",
    "gold": "Gold",
    "골드": "Gold",
    "platinum": "Platinum",
    "plat": "Platinum",
    "플래티넘": "Platinum",
    "플레티넘": "Platinum",
    "diamond": "Diamond",
    "다이아몬드": "Diamond",
    "ruby": "Ruby",
    "루비": "Ruby",
}

BUCKET_TO_TIER_RANGE = {
    "Bronze": range(1, 6),
    "Silver": range(6, 11),
    "Gold": range(11, 16),
    "Platinum": range(16, 21),
    "Diamond": range(21, 26),
    "Ruby": range(26, 31),
}

LEVEL_TO_OFFSET = {
    "5": 0,
    "v": 0,
    "Ⅴ": 0,
    "4": 1,
    "iv": 1,
    "Ⅳ": 1,
    "3": 2,
    "iii": 2,
    "Ⅲ": 2,
    "2": 3,
    "ii": 3,
    "Ⅱ": 3,
    "1": 4,
    "i": 4,
    "Ⅰ": 4,
}


def tier_to_bucket(tier_value: int | float | None) -> str:
    if tier_value is None:
        return "Unknown"
    try:
        value = int(tier_value)
    except (TypeError, ValueError):
        return "Unknown"
    if value <= 0:
        return "Unknown"
    for bucket, values in BUCKET_TO_TIER_RANGE.items():
        if value in values:
            return bucket
    return "Ruby"


def normalize_bucket(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    lowered = text.lower()
    for alias, bucket in BUCKET_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered) or alias in lowered:
            return bucket
    return tier_to_bucket(parse_tier_value(text))


def parse_tier_value(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    numeric = re.search(r"\b([1-9]|[12][0-9]|30)\b", text)
    if numeric and re.fullmatch(r"\s*([1-9]|[12][0-9]|30)\s*", text):
        return int(numeric.group(1))

    bucket = "Unknown"
    for alias, resolved in BUCKET_ALIASES.items():
        if alias in text.lower():
            bucket = resolved
            break
    if bucket == "Unknown":
        return int(numeric.group(1)) if numeric else None

    level_match = re.search(r"\b([1-5]|iv|iii|ii|i|v)\b|[ⅠⅡⅢⅣⅤ]", text.lower())
    if not level_match:
        return None
    level = level_match.group(0)
    offset = LEVEL_TO_OFFSET.get(level)
    if offset is None:
        return None
    start = BUCKET_TO_TIER_RANGE[bucket].start
    return start + offset


def normalize_difficulty(value: Any, tier_value: int | float | None = None) -> str:
    parsed_tier = parse_tier_value(value)
    if parsed_tier is not None and 1 <= parsed_tier <= 30:
        return tier_to_bucket(parsed_tier)
    bucket = normalize_bucket(value)
    if bucket != "Unknown":
        return bucket
    return tier_to_bucket(tier_value)


def bucket_tier_values(bucket: str) -> list[int]:
    normalized = normalize_bucket(bucket)
    if normalized not in BUCKET_TO_TIER_RANGE:
        return []
    return list(BUCKET_TO_TIER_RANGE[normalized])
