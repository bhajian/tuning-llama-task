#!/usr/bin/env python3
"""Evaluate base vs LoRA fine-tuned model using GPT as a judge.

Samples 1000 examples from the Alpaca dataset, queries both models,
and uses GPT-4o to judge response quality with binary scoring.

Usage:
    OPENAI_API_KEY=sk-xxx python evaluate.py \
        --base-url http://<BASE_IP>:8000 \
        --lora-url http://<LORA_IP>:8000 \
        --samples 1000
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from datasets import load_dataset
from openai import OpenAI


JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating AI model responses.
Given an instruction (and optional input) and the model's response, decide whether
the response is correct, relevant, and helpful.

Rules:
- Answer ONLY "YES" or "NO".
- YES means the response correctly addresses the instruction, is factually reasonable,
  and provides a useful answer.
- NO means the response is wrong, irrelevant, incomplete, or nonsensical."""

JUDGE_USER_TEMPLATE = """### Instruction:
{instruction}

{input_section}### Model Response:
{response}

Is this response correct, relevant, and helpful? Answer YES or NO."""


@dataclass
class EvalResult:
    instruction: str
    input_text: str
    expected: str
    base_response: str
    lora_response: str
    base_score: int = 0
    lora_score: int = 0
    base_error: str = ""
    lora_error: str = ""


def format_prompt(instruction: str, input_text: str = "") -> str:
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def query_model(client: OpenAI, model_name: str, instruction: str, input_text: str) -> str:
    prompt = format_prompt(instruction, input_text)
    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].text.strip()


def judge_response(judge_client: OpenAI, instruction: str, input_text: str, response: str) -> int:
    input_section = f"### Input:\n{input_text}\n\n" if input_text else ""
    user_msg = JUDGE_USER_TEMPLATE.format(
        instruction=instruction,
        input_section=input_section,
        response=response,
    )

    result = judge_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=5,
        temperature=0,
    )
    answer = result.choices[0].message.content.strip().upper()
    return 1 if answer.startswith("YES") else 0


def evaluate_sample(
    sample: dict,
    base_client: OpenAI,
    lora_client: OpenAI,
    judge_client: OpenAI,
    base_model: str,
    lora_model: str,
) -> EvalResult:
    instruction = sample["instruction"]
    input_text = sample.get("input", "") or ""
    expected = sample.get("output", "") or ""

    result = EvalResult(
        instruction=instruction,
        input_text=input_text,
        expected=expected,
        base_response="",
        lora_response="",
    )

    try:
        result.base_response = query_model(base_client, base_model, instruction, input_text)
    except Exception as e:
        result.base_error = str(e)

    try:
        result.lora_response = query_model(lora_client, lora_model, instruction, input_text)
    except Exception as e:
        result.lora_error = str(e)

    if result.base_response and not result.base_error:
        try:
            result.base_score = judge_response(judge_client, instruction, input_text, result.base_response)
        except Exception as e:
            result.base_error = f"judge error: {e}"

    if result.lora_response and not result.lora_error:
        try:
            result.lora_score = judge_response(judge_client, instruction, input_text, result.lora_response)
        except Exception as e:
            result.lora_error = f"judge error: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs LoRA model with GPT judge")
    parser.add_argument("--base-url", required=True, help="Base model vLLM endpoint (e.g. http://IP:8000)")
    parser.add_argument("--lora-url", required=True, help="LoRA model vLLM endpoint (e.g. http://IP:8000)")
    parser.add_argument("--base-model", default="/mnt/data/Llama-3.1-8B", help="Base model name in vLLM")
    parser.add_argument("--lora-model", default="lora-adapter", help="LoRA model name in vLLM")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for evaluation")
    parser.add_argument("--output", default="eval_results.json", help="Output file for detailed results")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable is required.")
        sys.exit(1)

    base_client = OpenAI(api_key="EMPTY", base_url=f"{args.base_url}/v1")
    lora_client = OpenAI(api_key="EMPTY", base_url=f"{args.lora_url}/v1")
    judge_client = OpenAI(api_key=openai_key)

    print(f"Loading Alpaca dataset ({args.samples} samples)...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(args.samples, len(dataset))))
    samples = list(dataset)
    print(f"Loaded {len(samples)} samples.")

    print(f"\nBase model:  {args.base_url} ({args.base_model})")
    print(f"LoRA model:  {args.lora_url} ({args.lora_model})")
    print(f"Judge:       GPT-4o")
    print(f"Workers:     {args.workers}")
    print(f"\nStarting evaluation...\n")

    results: list[EvalResult] = []
    base_correct = 0
    lora_correct = 0
    errors = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_sample, sample, base_client, lora_client,
                judge_client, args.base_model, args.lora_model,
            ): i
            for i, sample in enumerate(samples)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                base_correct += result.base_score
                lora_correct += result.lora_score
                if result.base_error or result.lora_error:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"  Sample {idx} failed: {e}")

            done = len(results)
            if done % 50 == 0 or done == len(samples):
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                base_acc = base_correct / done * 100 if done > 0 else 0
                lora_acc = lora_correct / done * 100 if done > 0 else 0
                print(
                    f"  [{done:4d}/{len(samples)}] "
                    f"Base: {base_acc:5.1f}% | LoRA: {lora_acc:5.1f}% | "
                    f"Errors: {errors} | {rate:.1f} samples/s"
                )

    total_time = time.time() - start_time
    evaluated = len(results) - errors
    base_accuracy = base_correct / evaluated * 100 if evaluated > 0 else 0
    lora_accuracy = lora_correct / evaluated * 100 if evaluated > 0 else 0
    improvement = lora_accuracy - base_accuracy

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples evaluated: {evaluated} / {len(samples)}")
    print(f"Errors:            {errors}")
    print(f"Time:              {total_time:.1f}s")
    print(f"")
    print(f"Base model accuracy:   {base_accuracy:5.1f}%  ({base_correct}/{evaluated})")
    print(f"LoRA model accuracy:   {lora_accuracy:5.1f}%  ({lora_correct}/{evaluated})")
    print(f"Improvement:           {improvement:+5.1f}%")
    print("=" * 60)

    output_data = {
        "config": {
            "base_url": args.base_url,
            "lora_url": args.lora_url,
            "samples": args.samples,
            "judge": "gpt-4o",
        },
        "summary": {
            "evaluated": evaluated,
            "errors": errors,
            "base_accuracy": round(base_accuracy, 2),
            "lora_accuracy": round(lora_accuracy, 2),
            "improvement": round(improvement, 2),
            "time_seconds": round(total_time, 1),
        },
        "details": [
            {
                "instruction": r.instruction,
                "input": r.input_text,
                "expected": r.expected,
                "base_response": r.base_response,
                "lora_response": r.lora_response,
                "base_score": r.base_score,
                "lora_score": r.lora_score,
                "base_error": r.base_error,
                "lora_error": r.lora_error,
            }
            for r in results
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
