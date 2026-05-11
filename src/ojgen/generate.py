import argparse
import json
from pathlib import Path

import unsloth
from unsloth import FastLanguageModel

from peft import PeftModel

from .difficulty import normalize_difficulty
from .utils import DEFAULT_SYSTEM_PROMPT, clean_text, ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate problems from a base model and LoRA adapter.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", default="unsloth/Qwen3-8B-unsloth-bnb-4bit")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    return parser.parse_args()


def build_user_prompt(payload: dict) -> str:
    tier_value = payload.get("tier_value", None)
    difficulty = normalize_difficulty(payload.get("difficulty"), tier_value)
    tags = payload.get("tags") or []
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.replace(";", ",").split(",") if tag.strip()]
    if not tags:
        tags = ["implementation"]
    time_limit = clean_text(payload.get("time_limit")) or "2 seconds"
    memory_limit = clean_text(payload.get("memory_limit")) or "256 MB"

    prompt_payload = {
        "task": "generate_algorithm_problem",
        "language": "ko",
        "difficulty": difficulty,
        "tier_value": tier_value,
        "tags": tags,
        "time_limit": time_limit,
        "memory_limit": memory_limit,
    }
    return json.dumps(prompt_payload, ensure_ascii=False)


def load_model(base_model: str, adapter_path: str, max_seq_length: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args.base_model, args.adapter, args.max_seq_length)
    output_path = Path(args.out)
    ensure_parent_dir(output_path)

    with Path(args.prompts).open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            prompt_object = json.loads(stripped)
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(prompt_object)},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
            )
            prompt_tokens = inputs["input_ids"].shape[1]
            for output_ids in outputs:
                generated_text = tokenizer.decode(output_ids[prompt_tokens:], skip_special_tokens=True).strip()
                row = {
                    "prompt": prompt_object,
                    "generated_text": generated_text,
                    "model": args.base_model,
                    "adapter": args.adapter,
                }
                target.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote generations to {output_path}")


if __name__ == "__main__":
    main()
