#!/usr/bin/env python3
"""LoRA fine-tuning of Llama 3.1 8B with DDP over Ethernet."""

import argparse
import json
import os
import shutil

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    overrides = {
        "model_id": args.model_id,
        "output_dir": args.output_dir,
        "train_data": args.train_data,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val

    float_keys = ["lr", "warmup_ratio", "lora_dropout", "max_grad_norm", "weight_decay"]
    int_keys = ["max_seq_len", "batch_size", "epochs", "lora_r", "lora_alpha"]
    for k in float_keys:
        if k in cfg:
            cfg[k] = float(cfg[k])
    for k in int_keys:
        if k in cfg:
            cfg[k] = int(cfg[k])

    return cfg


def format_alpaca(example):
    if example.get("input"):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


def tokenize_and_pack(dataset, tokenizer, max_seq_len):
    all_ids = []
    for example in dataset:
        text = format_alpaca(example) + tokenizer.eos_token
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)

    n_chunks = len(all_ids) // max_seq_len
    all_ids = all_ids[: n_chunks * max_seq_len]
    packed = torch.tensor(all_ids).reshape(n_chunks, max_seq_len)
    return packed


class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, packed_ids):
        self.input_ids = packed_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def save_checkpoint(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(path)
    tokenizer.save_pretrained(path)


def setup_profiler(cfg, global_rank):
    if not cfg.get("profiler_enabled", False) or global_rank != 0:
        return None
    profiler_dir = os.path.join(cfg["output_dir"], "profiler")
    os.makedirs(profiler_dir, exist_ok=True)
    start = cfg.get("profiler_start_step", 10)
    end = cfg.get("profiler_end_step", 20)
    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=start, warmup=2, active=end - start, repeat=1),
        on_trace_ready=tensorboard_trace_handler(profiler_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    print(f"Profiler enabled: steps {start}-{end}, output: {profiler_dir}")
    return prof


def setup_mlflow(cfg, global_rank):
    if not MLFLOW_AVAILABLE or not os.environ.get("MLFLOW_TRACKING_URI"):
        return False
    if global_rank != 0:
        return False
    mlflow.set_experiment(cfg.get("mlflow_experiment", "llama-lora"))
    mlflow.start_run()
    mlflow.log_params({
        "model_id": cfg["model_id"],
        "lr": cfg["lr"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "max_seq_len": cfg["max_seq_len"],
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "lora_targets": str(cfg["lora_targets"]),
        "warmup_ratio": cfg["warmup_ratio"],
        "max_grad_norm": cfg["max_grad_norm"],
        "weight_decay": cfg["weight_decay"],
    })
    return True


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--weight_decay", type=float)
    args = parser.parse_args()

    cfg = load_config(args)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    use_mlflow = setup_mlflow(cfg, global_rank)

    if global_rank == 0:
        print(f"World size: {world_size}")
        print(f"Config: {json.dumps(cfg, indent=2)}")
        print(f"MLflow tracking: {'enabled' if use_mlflow else 'disabled'}")
        print(f"Loading model: {cfg['model_id']}")

        os.makedirs(cfg["output_dir"], exist_ok=True)
        with open(os.path.join(cfg["output_dir"], "config.yaml"), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_targets"],
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

    with open(cfg["train_data"]) as f:
        dataset = json.load(f)
    if global_rank == 0:
        print(f"Loaded {len(dataset)} training examples from {cfg['train_data']}")
    packed = tokenize_and_pack(dataset, tokenizer, cfg["max_seq_len"])
    if global_rank == 0:
        print(f"Packed dataset: {len(packed)} sequences of {cfg['max_seq_len']} tokens")

    train_dataset = PackedDataset(packed)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, pin_memory=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    total_steps = len(dataloader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    profiler = setup_profiler(cfg, global_rank)

    if global_rank == 0:
        print(f"Training: {cfg['epochs']} epochs, {len(dataloader)} steps/epoch, {total_steps} total steps")

    model.train()
    global_step = 0
    best_loss = float("inf")

    if profiler:
        profiler.start()

    for epoch in range(cfg["epochs"]):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if profiler:
                profiler.step()

            if global_rank == 0 and step % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{cfg['epochs']} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {lr_now:.2e}")
                if use_mlflow:
                    mlflow.log_metrics({"train_loss": loss.item(), "lr": lr_now}, step=global_step)

        avg_loss = epoch_loss / len(dataloader)
        if global_rank == 0:
            print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
            if use_mlflow:
                mlflow.log_metric("epoch_avg_loss", avg_loss, step=epoch + 1)

            if cfg.get("save_every_epoch", False):
                ckpt_path = os.path.join(cfg["output_dir"], f"checkpoint-epoch-{epoch+1}")
                print(f"Saving checkpoint: {ckpt_path}")
                save_checkpoint(model, tokenizer, ckpt_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(cfg["output_dir"], "best-adapter")
                print(f"New best loss ({avg_loss:.4f}), saving: {best_path}")
                save_checkpoint(model, tokenizer, best_path)

    if profiler:
        profiler.stop()

    if global_rank == 0:
        print(f"Saving final adapter to {cfg['output_dir']}")
        save_checkpoint(model, tokenizer, cfg["output_dir"])
        if use_mlflow:
            profiler_dir = os.path.join(cfg["output_dir"], "profiler")
            if os.path.isdir(profiler_dir):
                mlflow.log_artifacts(profiler_dir, artifact_path="profiler")
            mlflow.log_artifacts(cfg["output_dir"])
            mlflow.end_run()
        print("Done.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
