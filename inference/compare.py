#!/usr/bin/env python3
"""Compare base vs LoRA fine-tuned model responses side by side."""

import argparse
import json
import time
import urllib.request

PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "Write a Python function that reverses a linked list.",
    "What are the main causes of climate change?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Give me a recipe for a simple pasta dish.",
]


def query_model(url, prompt, model_name, max_tokens=256):
    payload = json.dumps({
        "model": model_name,
        "prompt": f"### Instruction:\n{prompt}\n\n### Response:\n",
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - start

    text = result["choices"][0]["text"].strip()
    tokens = result["usage"]["completion_tokens"]
    return text, elapsed, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-host", default="localhost")
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--lora-host", default="localhost")
    parser.add_argument("--lora-port", type=int, default=8001)
    args = parser.parse_args()

    base_url = f"http://{args.base_host}:{args.base_port}/v1/completions"
    lora_url = f"http://{args.lora_host}:{args.lora_port}/v1/completions"

    print("=" * 80)
    print("BASE vs FINE-TUNED MODEL COMPARISON")
    print("=" * 80)

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n{'─' * 80}")
        print(f"PROMPT {i}: {prompt}")
        print(f"{'─' * 80}")

        print("\n[BASE MODEL]")
        try:
            base_text, base_time, base_tokens = query_model(
                base_url, prompt, "/mnt/data/Llama-3.1-8B"
            )
            print(base_text)
            print(f"\n  ⏱ {base_time:.2f}s | {base_tokens} tokens | {base_tokens/base_time:.1f} tok/s")
        except Exception as e:
            print(f"  Error: {e}")

        print("\n[FINE-TUNED MODEL (LoRA)]")
        try:
            lora_text, lora_time, lora_tokens = query_model(
                lora_url, prompt, "lora-adapter"
            )
            print(lora_text)
            print(f"\n  ⏱ {lora_time:.2f}s | {lora_tokens} tokens | {lora_tokens/lora_time:.1f} tok/s")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'=' * 80}")
    print("COMPARISON COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
