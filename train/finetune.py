#!/usr/bin/env python3
"""LoRA fine-tuning of Llama 3.1 8B with DDP over Ethernet."""

import json
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType


MODEL_ID = "/mnt/data/Llama-3.1-8B"
OUTPUT_DIR = "/mnt/data/llama-3.1-8b-lora-adapter"
TRAIN_DATA = "/mnt/data/dataset/train/alpaca_train.json"
MAX_SEQ_LEN = 2048
BATCH_SIZE = 2
EPOCHS = 3
LR = 2e-4
WARMUP_RATIO = 0.03
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def format_alpaca(example):
    if example.get("input"):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


def tokenize_and_pack(dataset, tokenizer):
    """Tokenize all examples and pack into fixed-length sequences."""
    all_ids = []
    for example in dataset:
        text = format_alpaca(example) + tokenizer.eos_token
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)

    n_chunks = len(all_ids) // MAX_SEQ_LEN
    all_ids = all_ids[: n_chunks * MAX_SEQ_LEN]
    packed = torch.tensor(all_ids).reshape(n_chunks, MAX_SEQ_LEN)
    return packed


class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, packed_ids):
        self.input_ids = packed_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if global_rank == 0:
        print(f"World size: {world_size}")
        print(f"Loading model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    if global_rank == 0:
        model.print_trainable_parameters()

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    if global_rank == 0:
        print("Loading and tokenizing dataset...")

    with open(TRAIN_DATA) as f:
        dataset = json.load(f)
    if global_rank == 0:
        print(f"Loaded {len(dataset)} training examples from {TRAIN_DATA}")
    packed = tokenize_and_pack(dataset, tokenizer)
    if global_rank == 0:
        print(f"Packed dataset: {len(packed)} sequences of {MAX_SEQ_LEN} tokens")

    train_dataset = PackedDataset(packed)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

    total_steps = len(dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if global_rank == 0:
        print(f"Training: {EPOCHS} epochs, {len(dataloader)} steps/epoch, {total_steps} total steps")

    model.train()
    global_step = 0
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if global_rank == 0 and step % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {lr_now:.2e}")

        avg_loss = epoch_loss / len(dataloader)
        if global_rank == 0:
            print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

    if global_rank == 0:
        print(f"Saving adapter to {OUTPUT_DIR}")
        unwrapped = model.module
        unwrapped.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
