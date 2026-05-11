import argparse
import json
import torch
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm
from .utils import DEFAULT_SYSTEM_PROMPT, ensure_parent_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Generate candidate problems using vLLM.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or path")
    parser.add_argument("--prompts", required=True, help="Path to input jsonl")
    parser.add_argument("--out", required=True, help="Path to output jsonl")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()

def build_prompt(payload):
    # Matches existing logic from original generate.py
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    return messages

def main():
    args = parse_args()
    
    # Initialize vLLM
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    
    # Load prompts
    prompts = []
    metadata = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(llm.get_tokenizer().apply_chat_template(build_prompt(data), tokenize=False, add_generation_prompt=True))
            metadata.append(data)
            
    # Generation params
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    
    # Generate
    outputs = llm.generate(prompts, sampling_params)
    
    # Write output
    ensure_parent_dir(Path(args.out))
    with open(args.out, "w", encoding="utf-8") as f:
        for output, meta in zip(outputs, metadata):
            f.write(json.dumps({
                "prompt": meta,
                "generated_text": output.outputs[0].text
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
