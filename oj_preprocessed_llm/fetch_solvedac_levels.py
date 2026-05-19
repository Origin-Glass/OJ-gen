#!/usr/bin/env python3
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


def fetch_one(pid, timeout=20, retries=3, sleep_sec=0.12):
    url = "https://solved.ac/api/v3/problem/show?" + urllib.parse.urlencode(
        {"problemId": pid}
    )

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "boj-llm-preprocess/1.0"},
            )

            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))

            time.sleep(sleep_sec)
            return data

        except Exception as e:
            last_error = e

            if attempt < retries:
                backoff = sleep_sec * (2**attempt)
                time.sleep(backoff)

    raise RuntimeError(f"failed to fetch {pid}: {last_error}")


def load_done(outp):
    done = set()

    if not outp.exists():
        return done

    for line in outp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            obj = json.loads(line)
            pid = obj.get("problemId") or obj.get("problem_id")

            if pid is not None:
                done.add(int(pid))

        except Exception:
            pass

    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", default="solvedac_levels.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=0.12)
    args = ap.parse_args()

    ids = [
        int(x.strip())
        for x in Path(args.ids).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    outp = Path(args.out)

    done = set()
    if args.resume:
        done = load_done(outp)

    targets = [pid for pid in ids if pid not in done]

    mode = "a" if args.resume else "w"
    write_lock = Lock()

    total = len(ids)
    target_total = len(targets)
    success_count = 0
    fail_count = 0

    print(f"total ids: {total}")
    print(f"already done: {len(done)}")
    print(f"remaining: {target_total}")
    print(f"workers: {args.workers}")

    with outp.open(mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    fetch_one,
                    pid,
                    args.timeout,
                    args.retries,
                    args.sleep,
                ): pid
                for pid in targets
            }

            for idx, future in enumerate(as_completed(future_map), 1):
                pid = future_map[future]

                try:
                    obj = future.result()

                    with write_lock:
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        f.flush()

                    success_count += 1
                    print(f"[{idx}/{target_total}] fetched {pid}")

                except Exception as e:
                    fail_count += 1
                    print(f"[WARN] {pid}: {e}")
                    os._exit(0)

    print("done")
    print(f"success: {success_count}")
    print(f"failed: {fail_count}")


if __name__ == "__main__":
    main()
