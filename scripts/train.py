"""
Monico Pretraining Script
Distributed training across H100 cluster.
Uses FSDP + gradient checkpointing + flash attention.
"""

import os
import math
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, IterableDataset
import wandb

from src.model.architecture import MonicoModel, MonicoConfig, MonicoDecoderLayer
from src.tokenizer.tokenizer import MonicoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # Model
    model_config: MonicoConfig = field(default_factory=MonicoConfig)

    # Data
    data_path: str = "data/processed"
    tokenizer_path: str = "tokenizer"
    seq_length: int = 4096        # Increase to 32768 for long-context phase

    # Training
    batch_size: int = 4            # Per GPU
    gradient_accumulation: int = 8 # Effective batch = 4 * 8 * num_gpus
    max_steps: int = 1_000_000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bf16: bool = True

    # Checkpointing
    output_dir: str = "checkpoints/monico-7b"
    save_every: int = 5000
    eval_every: int = 1000
    log_every: int = 10

    # W&B
    wandb_project: str = "monico-model"
    wandb_run_name: str = "monico-7b-pretrain-v1"


class StreamingDataset(IterableDataset):
    """Stream tokenized training data from disk"""
    def __init__(self, data_path: str, seq_length: int, rank: int, world_size: int):
        self.data_path = Path(data_path)
        self.seq_length = seq_length
        self.rank = rank
        self.world_size = world_size
        self.files = sorted(self.data_path.glob("*.bin"))[rank::world_size]

    def __iter__(self):
        buffer = []
        for file in self.files:
            data = torch.load(file, map_location="cpu")
            buffer.extend(data.tolist())
            while len(buffer) >= self.seq_length + 1:
                chunk = buffer[:self.seq_length + 1]
                buffer = buffer[self.seq_length:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def cosine_lr_schedule(step: int, config: TrainingConfig) -> float:
    if step < config.warmup_steps:
        return float(step) / float(max(1, config.warmup_steps))
    progress = float(step - config.warmup_steps) / float(max(1, config.max_steps - config.warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))


def train(config: TrainingConfig):
    # Init distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config))
        logger.info(f"Training Monico with {world_size} GPUs")

    # Model
    model = MonicoModel(config.model_config)
    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {total/1e9:.2f}B parameters")

    # FSDP wrap
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=device,
        auto_wrap_policy=transformer_auto_wrap_policy(
            transformer_layer_cls={MonicoDecoderLayer}
        ),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=True
    )

    # Data
    tokenizer = MonicoTokenizer.from_pretrained(config.tokenizer_path)
    dataset = StreamingDataset(config.data_path, config.seq_length, rank, world_size)
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

    # Training loop
    model.train()
    step = 0
    accumulated = 0
    optimizer.zero_grad()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"] / config.gradient_accumulation

        loss.backward()
        accumulated += 1

        if accumulated == config.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            lr_scale = cosine_lr_schedule(step, config)
            for pg in optimizer.param_groups:
                pg["lr"] = config.learning_rate * lr_scale

            optimizer.step()
            optimizer.zero_grad()
            accumulated = 0
            step += 1

            if step % config.log_every == 0 and rank == 0:
                logger.info(f"Step {step} | Loss: {loss.item() * config.gradient_accumulation:.4f} | LR: {config.learning_rate * lr_scale:.2e}")
                wandb.log({"loss": loss.item() * config.gradient_accumulation, "lr": config.learning_rate * lr_scale, "step": step})

            if step % config.save_every == 0 and rank == 0:
                Path(config.output_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), f"{config.output_dir}/checkpoint-{step}.pt")
                logger.info(f"Saved checkpoint at step {step}")

            if step >= config.max_steps:
                break

    dist.destroy_process_group()
    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    config = TrainingConfig()
    train(config)
