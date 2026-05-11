#!/usr/bin/env python3
"""Split the Alpaca dataset into 90% train / 10% eval.

Produces:
    dataset/train/alpaca_train.json
    dataset/eval/alpaca_eval.json

Usage:
    cd training-task
    python dataset/split_dataset.py
"""

import json
import os

from datasets import load_dataset

SEED = 42
TRAIN_RATIO = 0.9

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, "train", "alpaca_train.json")
EVAL_PATH = os.path.join(SCRIPT_DIR, "eval", "alpaca_eval.json")


def main():
    print("Loading tatsu-lab/alpaca...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=SEED)

    split = dataset.train_test_split(test_size=1 - TRAIN_RATIO, seed=SEED)
    train_data = [dict(row) for row in split["train"]]
    eval_data = [dict(row) for row in split["test"]]

    with open(TRAIN_PATH, "w") as f:
        json.dump(train_data, f)
    print(f"Train: {len(train_data)} examples -> {TRAIN_PATH}")

    with open(EVAL_PATH, "w") as f:
        json.dump(eval_data, f)
    print(f"Eval:  {len(eval_data)} examples -> {EVAL_PATH}")


if __name__ == "__main__":
    main()
