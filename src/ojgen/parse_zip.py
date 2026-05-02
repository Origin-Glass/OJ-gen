import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from .common import clean_text, write_jsonl

INFO_FIELDS = ["시간 제한", "메모리 제한", "제출", "정답", "맞힌 사람", "정답 비율"]


def parse_problem_id(name: str) -> int | None:
    m = re.search(r"problem_(\d+)_", name)
    return int(m.group(1)) if m else None


def text_by_id(soup: BeautifulSoup, id_: str) -> str:
    node = soup.select_one(f"#{id_}")
    return clean_text(node.get_text("\n", strip=True)) if node else ""


def parse_info_table(soup: BeautifulSoup) -> dict:
    out = {"time_limit": "", "memory_limit": "", "submissions": "", "accepted": "", "solved_users": "", "acceptance_rate": ""}
    table = soup.select_one("#problem-info")
    if not table:
        return out
    cells = [clean_text(x.get_text(" ", strip=True)) for x in table.select("th,td")]
    if len(cells) >= 12:
        labels = cells[:6]
        vals = cells[6:12]
        mp = dict(zip(labels, vals))
        out.update({
            "time_limit": mp.get("시간 제한", ""),
            "memory_limit": mp.get("메모리 제한", ""),
            "submissions": mp.get("제출", ""),
            "accepted": mp.get("정답", ""),
            "solved_users": mp.get("맞힌 사람", ""),
            "acceptance_rate": mp.get("정답 비율", ""),
        })
    return out


def parse_samples(soup: BeautifulSoup) -> list[dict]:
    samples = []
    inputs = {}
    outputs = {}
    for pre in soup.select("pre"):
        pid = pre.get("id") or ""
        m_in = re.match(r"sample-input-(\d+)", pid)
        m_out = re.match(r"sample-output-(\d+)", pid)
        if m_in:
            inputs[int(m_in.group(1))] = clean_text(pre.get_text("\n", strip=False))
        elif m_out:
            outputs[int(m_out.group(1))] = clean_text(pre.get_text("\n", strip=False))
    for idx in sorted(set(inputs) | set(outputs)):
        samples.append({"sample_index": idx, "sample_input": inputs.get(idx, ""), "sample_output": outputs.get(idx, "")})
    return samples


def parse_html(name: str, html: bytes) -> tuple[dict, list[dict]] | None:
    pid = parse_problem_id(name)
    if pid is None:
        return None
    soup = BeautifulSoup(html, "lxml")
    title = text_by_id(soup, "problem_title")
    rec = {
        "problem_id": pid,
        "title": title,
        "url": f"https://www.acmicpc.net/problem/{pid}",
        "description": text_by_id(soup, "problem_description"),
        "input": text_by_id(soup, "problem_input"),
        "output": text_by_id(soup, "problem_output"),
        "limit": text_by_id(soup, "problem_limit"),
        "hint": text_by_id(soup, "problem_hint"),
        "source": text_by_id(soup, "source"),
        "saved_file": name,
    }
    rec.update(parse_info_table(soup))
    samples = parse_samples(soup)
    return rec, [{"problem_id": pid, **s} for s in samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = []
    samples = []
    with zipfile.ZipFile(args.zip) as zf:
        names = [n for n in zf.namelist() if n.endswith(".html") and "/archive/problem_" in n]
        names.sort(key=lambda n: parse_problem_id(n) or 0)
        if args.limit:
            names = names[:args.limit]
        for name in tqdm(names, desc="parse html"):
            try:
                parsed = parse_html(name, zf.read(name))
                if parsed:
                    p, ss = parsed
                    problems.append(p)
                    samples.extend(ss)
            except Exception as e:
                problems.append({"problem_id": parse_problem_id(name), "parse_error": str(e), "saved_file": name})
    write_jsonl(out_dir / "problems.jsonl", problems)
    write_jsonl(out_dir / "samples.jsonl", samples)
    if problems:
        with open(out_dir / "problems.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in problems for k in r.keys()}))
            writer.writeheader(); writer.writerows(problems)
    if samples:
        with open(out_dir / "samples.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["problem_id", "sample_index", "sample_input", "sample_output"])
            writer.writeheader(); writer.writerows(samples)
    print(f"wrote {len(problems)} problems and {len(samples)} samples to {out_dir}")

if __name__ == "__main__":
    main()
