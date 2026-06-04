"""
Monico Pre-training Script
Supports: single GPU, DDP multi-GPU, DeepSpeed ZeRO-1/2/3

Usage:
  # Single GPU
  python train.py --config configs/7b_pretrain.yaml

  # Multi-GPU DDP
  torchrun --nproc_per_node=8 train.py --config configs/7b_pretrain.yaml

  # DeepSpeed ZeRO-3
  deepspeed --num_gpus=8 train.py --config configs/7b_pretrain.yaml --deepspeed configs/ds_zero3.json
"""
import os
import sys
import math
import time
import json
import yaml
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.monicoformer import MonicoForCausalLM, MonicoConfig, count_params

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    from torch.cuda.amp import GradScaler, autocast
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("monico")


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model_size: str = "7b"
    vocab_size: int = 128_000

    # Data
    data_dir: str = "data/tokenized"
    dataset_files: list = None
    seq_len: int = 4096
    num_workers: int = 4

    # Training
    batch_size: int = 1          # per-GPU micro batch
    gradient_accumulation: int = 8
    max_steps: int = 500_000
    warmup_steps: int = 2000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 10

    # Precision
    dtype: str = "bfloat16"       # bfloat16 | float16 | float32
    compile_model: bool = True     # torch.compile

    # DeepSpeed
    deepspeed_config: Optional[str] = None

    # Resume
    resume_from: Optional[str] = None


def load_config(path: str) -> TrainConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    return TrainConfig(**d)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class TokenizedDataset(IterableDataset):
    """Streams pre-tokenized binary files (uint32 .bin format like RedPajama/TinyLlama)."""

    def __init__(self, data_dir: str, seq_len: int, rank: int = 0, world_size: int = 1):
        self.files = sorted(Path(data_dir).glob("*.bin"))
        if not self.files:
            raise FileNotFoundError(f"No .bin files in {data_dir}")
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        import numpy as np
        buf = np.array([], dtype=np.uint32)
        chunk_size = self.seq_len + 1

        for i, path in enumerate(self.files):
            if i % self.world_size != self.rank:
                continue  # Shard files across workers
            data = np.fromfile(path, dtype=np.uint32)
            buf = np.concatenate([buf, data])
            while len(buf) >= chunk_size:
                chunk = buf[:chunk_size]
                buf = buf[chunk_size:]
                x = torch.from_numpy(chunk[:-1].astype(np.int64))
                y = torch.from_numpy(chunk[1:].astype(np.int64))
                yield {"input_ids": x, "labels": y}


# ─── LR Scheduler ─────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


# ─── Training Loop ─────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.rank == 0

        # Setup distributed
        if self.world_size > 1:
            dist.init_process_group("nccl")
            torch.cuda.set_device(self.local_rank)

        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]

        # Build model
        model_cfg = getattr(MonicoConfig, f"monico_{cfg.model_size}")()
        model_cfg.vocab_size = cfg.vocab_size
        self.model = MonicoForCausalLM(model_cfg).to(self.device)

        if cfg.compile_model and hasattr(torch, "compile"):
            log.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        if self.is_main:
            count_params(self.model)

        # Mixed precision
        self.scaler = GradScaler() if cfg.dtype == "float16" and HAS_AMP else None

        # Optimizer
        decay_params = [p for n, p in self.model.named_parameters()
                        if p.requires_grad and len(p.shape) >= 2]
        no_decay_params = [p for n, p in self.model.named_parameters()
                           if p.requires_grad and len(p.shape) < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay_params, "weight_decay": cfg.weight_decay},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=True,
        )

        # DDP
        if self.world_size > 1 and not cfg.deepspeed_config:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        # Data
        dataset = TokenizedDataset(cfg.data_dir, cfg.seq_len, self.rank, self.world_size)
        self.loader = DataLoader(dataset, batch_size=cfg.batch_size,
                                 num_workers=cfg.num_workers, pin_memory=True)

        # State
        self.step = 0
        self.best_loss = float("inf")

        if cfg.resume_from:
            self._load_checkpoint(cfg.resume_from)

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self, path: str):
        ck = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.step = ck["step"]
        log.info(f"Resumed from step {self.step}")

    def _save_checkpoint(self, tag: str = "latest"):
        if not self.is_main:
            return
        path = Path(self.cfg.save_dir) / f"monico-{self.cfg.model_size}-{tag}.pt"
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({
            "step": self.step,
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
        }, path)
        log.info(f"Checkpoint saved: {path}")

    def train(self):
        log.info(f"Starting training — rank {self.rank}/{self.world_size}")
        t0 = time.time()

        for batch in self.loader:
            if self.step >= self.cfg.max_steps:
                break

            # LR schedule
            lr = get_lr(self.step, self.cfg)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            # Accumulate gradients
            self.model.train()
            total_loss = 0.0

            for micro_step in range(self.cfg.gradient_accumulation):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with (autocast(dtype=self.dtype) if HAS_AMP else torch.autocast("cuda", dtype=self.dtype)):
                    out = self.model(input_ids, labels=labels)
                    loss = out["loss"] / self.cfg.gradient_accumulation

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item()

            # Grad clip + step
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            # Logging
            if self.step % self.cfg.log_every == 0 and self.is_main:
                elapsed = time.time() - t0
                tokens_per_sec = (self.cfg.log_every * self.cfg.batch_size *
                                  self.cfg.gradient_accumulation * self.cfg.seq_len *
                                  self.world_size) / elapsed
                log.info(
                    f"step {self.step:>7} | loss {total_loss:.4f} | "
                    f"lr {lr:.2e} | {tokens_per_sec/1e6:.2f}M tok/s"
                )
                t0 = time.time()

            # Save
            if self.step % self.cfg.save_every == 0:
                self._save_checkpoint(str(self.step))
                self._save_checkpoint("latest")

        self._save_checkpoint("final")
        log.info("Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/7b_pretrain.yaml")
    parser.add_argument("--deepspeed", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.deepspeed:
        cfg.deepspeed_config = args.deepspeed

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
