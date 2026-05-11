#!/usr/bin/env python3
"""Evaluate base vs LoRA fine-tuned model using Claude Sonnet as a judge via Vertex AI.

Samples 1000 examples from the Alpaca dataset, queries both models,
and uses Claude Sonnet to judge response quality with binary scoring.
Supports checkpointing for resumable evaluation.

Usage:
    python evaluate.py \
        --base-url http://<BASE_IP>:8000 \
        --lora-url http://<LORA_IP>:8000 \
        --gcp-project <PROJECT_ID> \
        --samples 1000
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
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

JUDGE_MODEL = "claude-sonnet-4-20250514"
JUDGE_DELAY = 1.0
_rate_lock = threading.Lock()


def format_prompt(instruction: str, input_text: str = "") -> str:
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def query_model(client: OpenAI, model_name: str, instruction: str, input_text: str) -> tuple:
    prompt = format_prompt(instruction, input_text)
    start = time.time()
    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=512,
        temperature=0.1,
    )
    elapsed = time.time() - start
    text = response.choices[0].text.strip()
    tokens = response.usage.completion_tokens
    return text, elapsed, tokens


def judge_response(judge_client, instruction: str, input_text: str, response: str) -> int:
    if not response:
        return 0
    input_section = f"### Input:\n{input_text}\n\n" if input_text else ""
    user_msg = JUDGE_USER_TEMPLATE.format(
        instruction=instruction,
        input_section=input_section,
        response=response,
    )
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with _rate_lock:
                time.sleep(JUDGE_DELAY)
            result = judge_client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=5,
                temperature=0,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            answer = result.content[0].text.strip().upper()
            return 1 if answer.startswith("YES") else 0
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt * 5
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"    Judge error: {e}")
                return 0
    print(f"    Max retries exceeded")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs LoRA model with Claude Sonnet judge")
    parser.add_argument("--base-url", required=True, help="Base model vLLM endpoint (e.g. http://IP:8000)")
    parser.add_argument("--lora-url", required=True, help="LoRA model vLLM endpoint (e.g. http://IP:8000)")
    parser.add_argument("--base-model", default="/mnt/data/Llama-3.1-8B", help="Base model name in vLLM")
    parser.add_argument("--lora-model", default="lora-adapter", help="LoRA model name in vLLM")
    parser.add_argument("--gcp-project", default=os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", ""), help="GCP project ID")
    parser.add_argument("--gcp-region", default=os.environ.get("CLOUD_ML_REGION", "europe-west1"), help="GCP region")
    parser.add_argument("--eval-data", default="dataset/eval/alpaca_eval.json", help="Path to eval split JSON")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    if not args.gcp_project:
        print("Error: --gcp-project or ANTHROPIC_VERTEX_PROJECT_ID required.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    base_client = OpenAI(api_key="EMPTY", base_url=f"{args.base_url}/v1")
    lora_client = OpenAI(api_key="EMPTY", base_url=f"{args.lora_url}/v1")
    judge_client = anthropic.AnthropicVertex(
        project_id=args.gcp_project,
        region=args.gcp_region,
    )

    # Load or resume samples
    samples_file = f"{args.output_dir}/samples.json"
    if os.path.exists(samples_file):
        with open(samples_file) as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} samples from checkpoint")
    else:
        print(f"Loading eval split from {args.eval_data}...")
        with open(args.eval_data) as f:
            eval_data = json.load(f)
        import random
        rng = random.Random(42)
        samples = rng.sample(eval_data, min(args.samples, len(eval_data)))
        with open(samples_file, "w") as f:
            json.dump(samples, f)
        print(f"Sampled {len(samples)} from {len(eval_data)} eval examples")

    print(f"\nBase model:  {args.base_url} ({args.base_model})")
    print(f"LoRA model:  {args.lora_url} ({args.lora_model})")
    print(f"Judge:       {JUDGE_MODEL} via Vertex AI")
    print(f"Workers:     {args.workers}")

    # Phase 1: Query models (with checkpointing)
    query_checkpoint = f"{args.output_dir}/query_results.json"
    if os.path.exists(query_checkpoint):
        with open(query_checkpoint) as f:
            query_results = json.load(f)
        completed = {r["index"] for r in query_results}
        remaining = [(i, s) for i, s in enumerate(samples) if i not in completed]
        print(f"\nQuery phase: {len(query_results)} done, {len(remaining)} remaining")
    else:
        query_results = []
        remaining = list(enumerate(samples))
        print(f"\nQuerying {len(remaining)} samples...")

    if remaining:
        def query_sample(i, sample):
            instruction = sample["instruction"]
            input_text = sample.get("input", "") or ""
            result = {"index": i, "instruction": instruction, "input": input_text, "expected": sample.get("output", "")}
            try:
                text, elapsed, tokens = query_model(base_client, args.base_model, instruction, input_text)
                result["base_response"] = text
                result["base_time"] = elapsed
                result["base_tokens"] = tokens
            except Exception as e:
                result["base_response"] = ""
                result["base_time"] = 0
                result["base_tokens"] = 0
                result["base_error"] = str(e)
            try:
                text, elapsed, tokens = query_model(lora_client, args.lora_model, instruction, input_text)
                result["lora_response"] = text
                result["lora_time"] = elapsed
                result["lora_tokens"] = tokens
            except Exception as e:
                result["lora_response"] = ""
                result["lora_time"] = 0
                result["lora_tokens"] = 0
                result["lora_error"] = str(e)
            return result

        start = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(query_sample, i, s): i for i, s in remaining}
            for future in as_completed(futures):
                query_results.append(future.result())
                done = len(query_results)
                if done % 50 == 0:
                    query_results.sort(key=lambda x: x["index"])
                    with open(query_checkpoint, "w") as f:
                        json.dump(query_results, f)
                    print(f"  [{done}/{len(samples)}] {time.time()-start:.0f}s — checkpointed")
        query_results.sort(key=lambda x: x["index"])
        with open(query_checkpoint, "w") as f:
            json.dump(query_results, f)
        print(f"  Query complete: {time.time()-start:.1f}s")

    # Phase 2: Judge (with checkpointing)
    judge_checkpoint = f"{args.output_dir}/judge_results.json"
    if os.path.exists(judge_checkpoint):
        with open(judge_checkpoint) as f:
            judged = json.load(f)
        judged_indices = {r["index"] for r in judged if "base_score" in r}
        remaining_judge = [r for r in query_results if r["index"] not in judged_indices]
        print(f"\nJudge phase: {len(judged)} done, {len(remaining_judge)} remaining")
    else:
        judged = []
        remaining_judge = list(query_results)
        print(f"\nJudging {len(remaining_judge)} samples...")

    if remaining_judge:
        def judge_sample(result):
            result = dict(result)
            result["base_score"] = judge_response(judge_client, result["instruction"], result["input"], result.get("base_response", ""))
            result["lora_score"] = judge_response(judge_client, result["instruction"], result["input"], result.get("lora_response", ""))
            return result

        start = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(judge_sample, r): r["index"] for r in remaining_judge}
            for future in as_completed(futures):
                judged.append(future.result())
                done = len(judged)
                if done % 25 == 0:
                    judged.sort(key=lambda x: x["index"])
                    with open(judge_checkpoint, "w") as f:
                        json.dump(judged, f)
                    base_acc = sum(r.get("base_score", 0) for r in judged) / done * 100
                    lora_acc = sum(r.get("lora_score", 0) for r in judged) / done * 100
                    print(f"  [{done}/{len(query_results)}] Base: {base_acc:.1f}% | LoRA: {lora_acc:.1f}% | {time.time()-start:.0f}s")
        judged.sort(key=lambda x: x["index"])
        with open(judge_checkpoint, "w") as f:
            json.dump(judged, f)

    # Summary
    n = len(judged)
    base_correct = sum(r.get("base_score", 0) for r in judged)
    lora_correct = sum(r.get("lora_score", 0) for r in judged)
    base_acc = base_correct / n * 100
    lora_acc = lora_correct / n * 100

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples:    {n}")
    print(f"Judge:      {JUDGE_MODEL} via Vertex AI")
    print(f"Base acc:   {base_acc:.1f}% ({base_correct}/{n})")
    print(f"LoRA acc:   {lora_acc:.1f}% ({lora_correct}/{n})")
    print(f"Improvement: {lora_acc - base_acc:+.1f}%")
    print("=" * 60)

    summary = {
        "num_samples": n,
        "judge_model": JUDGE_MODEL,
        "base_accuracy": round(base_acc, 2),
        "lora_accuracy": round(lora_acc, 2),
        "improvement": round(lora_acc - base_acc, 2),
    }
    with open(f"{args.output_dir}/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
