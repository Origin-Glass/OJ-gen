import argparse
import inspect
from pathlib import Path
from typing import Any

import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from .utils import ensure_parent_dir

ALL_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

ATTN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
QV_TARGET_MODULES = ["q_proj", "v_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen3-8B model with Unsloth SFT.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", choices=["all", "attn", "qv"], default="all")
    parser.add_argument("--packing", dest="packing", action="store_true")
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    parser.set_defaults(packing=True)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--max-trainable-params", type=int, default=500_000_000)
    parser.add_argument("--allow-large-lora", action="store_true")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--deepspeed", default=None)
    return parser.parse_args()


def resolve_target_modules(name: str) -> list[str]:
    if name == "all":
        return ALL_TARGET_MODULES
    if name == "attn":
        return ATTN_TARGET_MODULES
    if name == "qv":
        return QV_TARGET_MODULES
    raise ValueError(f"Unsupported target module preset: {name}")


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{index}] must be a dict, got {type(message).__name__}")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise TypeError(f"messages[{index}].role must be a non-empty string")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        normalized.append({"role": role, "content": content})
    return normalized


def _render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, list) and all(isinstance(part, str) for part in rendered):
        return "".join(rendered)
    raise TypeError(f"apply_chat_template must return text when tokenize=False, got {type(rendered).__name__}")


def _unwrap_input_ids(input_ids: Any) -> list[int]:
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError(f"Expected one tokenized sequence, got batch size {len(input_ids)}")
        input_ids = input_ids[0]
    return list(input_ids or [])


def _tokenize_text(tokenizer, text: str, max_seq_length: int) -> list[int]:
    kwargs = {
        "truncation": True,
        "max_length": max_seq_length,
        "add_special_tokens": False,
    }
    try:
        tokenized = tokenizer(text=text, **kwargs)
    except TypeError:
        tokenized = tokenizer(text, **kwargs)
    return _unwrap_input_ids(tokenized["input_ids"])


def _decode_tokens(tokenizer, input_ids: list[int]) -> str:
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    if not isinstance(decoded, str):
        raise TypeError(f"decode must return str, got {type(decoded).__name__}")
    return decoded


def get_text_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def format_and_truncate_dataset(dataset, tokenizer, max_seq_length: int):
    def transform_batch(batch):
        texts = []
        token_lengths = []
        for messages in batch["messages"]:
            normalized_messages = _normalize_messages(messages)
            rendered = _render_chat(tokenizer, normalized_messages)
            input_ids = _tokenize_text(tokenizer, rendered, max_seq_length)
            token_lengths.append(len(input_ids))
            texts.append(_decode_tokens(tokenizer, input_ids))
        return {"text": texts, "token_length": token_lengths}

    return dataset.map(
        transform_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="apply chat template",
    )


def count_trainable_parameters(model) -> int | None:
    try:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    ensure_parent_dir(output_dir / "adapter")

    raw_dataset = load_dataset("json", data_files=args.data, split="train")
    if "messages" not in raw_dataset.column_names:
        raise ValueError("Dataset must contain a 'messages' column.")

    model, processor = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_text_tokenizer(processor)
    target_modules = resolve_target_modules(args.target_modules)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    train_dataset = format_and_truncate_dataset(raw_dataset, tokenizer, args.max_seq_length)
    max_seen_length = max(train_dataset["token_length"]) if len(train_dataset) else 0
    if len(train_dataset) and max_seen_length <= 1:
        raise ValueError(
            "Tokenized training rows are unexpectedly length <= 1. "
            "Check that the JSONL messages contain non-empty string content.",
        )
    use_bf16 = is_bfloat16_supported()
    trainable_parameters = count_trainable_parameters(model)

    print(f"Model: {args.model}")
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Observed max tokenized length: {max_seen_length}")
    print(f"Target modules: {args.target_modules} -> {target_modules}")
    if trainable_parameters is not None:
        print(f"Trainable parameters: {trainable_parameters}")
        if trainable_parameters > args.max_trainable_params and not args.allow_large_lora:
            raise ValueError(
                f"Trainable parameters are too high for the default safety limit "
                f"({trainable_parameters:,} > {args.max_trainable_params:,}). "
                "For Qwen MoE models, avoid '--target-modules all' unless you have a larger "
                "multi-GPU setup. Try '--target-modules qv --lora-r 8 --lora-alpha 16 "
                "--max-seq-length 4096'. Use --allow-large-lora only if you intentionally "
                "want to risk the memory cost.",
            )
    print(f"Output path: {output_dir}")

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "packing": args.packing,
        "args": TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            warmup_ratio=0.03,
            weight_decay=0.01,
            optim=args.optim,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_strategy="steps",
            bf16=use_bf16,
            fp16=not use_bf16,
            report_to=args.report_to,
            seed=args.seed,
            lr_scheduler_type="cosine",
            gradient_checkpointing=args.gradient_checkpointing,
            deepspeed=args.deepspeed,
        ),
    }
    if "dataset_num_proc" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["dataset_num_proc"] = args.dataset_num_proc
    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"Saved adapter and processor to {output_dir}")


if __name__ == "__main__":
    main()
