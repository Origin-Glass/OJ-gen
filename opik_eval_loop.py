import argparse
import json
from pathlib import Path

from ojgen.validators import validate_generated_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local validation and optional Opik logging on generations.")
    parser.add_argument("--generated", required=True)
    parser.add_argument("--project", default="oj-problem-generator")
    parser.add_argument("--dataset", default="generated-problems")
    parser.add_argument("--out", default="outputs/evaluated.jsonl")
    parser.add_argument("--log-opik", action="store_true")
    return parser.parse_args()


def maybe_create_opik_dataset(log_opik: bool, project: str, dataset: str):
    if not log_opik:
        return None
    try:
        from opik import Opik
    except ImportError:
        print("Opik is not installed. Continuing with local evaluation only.")
        return None
    client = Opik(project_name=project)
    return client.get_or_create_dataset(name=dataset)


def main() -> None:
    args = parse_args()
    dataset = maybe_create_opik_dataset(args.log_opik, args.project, args.dataset)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with Path(args.generated).open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            text = row.get("generated_text") or row.get("output") or row.get("text") or ""
            validation = validate_generated_problem(text)
            payload = {**row, "validation": validation}
            target.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            if dataset is not None:
                dataset.insert(
                    [
                        {
                            "input": row.get("prompt", {}),
                            "output": text,
                            "scores": validation,
                        }
                    ]
                )
    print(f"Wrote {written} evaluated rows to {output_path}")


if __name__ == "__main__":
    main()
