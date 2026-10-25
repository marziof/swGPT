# src/gpt/config.py
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import yaml


@dataclass
class Config:
    max_steps: int = 5000
    eval_interval: int = 250
    eval_batches: int = 20
    ckpt_interval: int = 1000
    run_dir: str = "runs"
    run_name: str | None = None        # auto-filled in load_config
    resume: str | None = None          # path to a checkpoint to resume from
    min_lr_ratio: float = 0.1
    lr: float = 1.0e-3
    weight_decay: float = 0.1
    optimizer: str = "adamw"
    warmup_steps: int = 100  
    min_lr_ratio: float = 0.1  
    grad_clip: float = 1.0
    seed: int=42
    overfit: bool = False

    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    context_len: int = 256
    batch_size: int = 64
    mlp_mult: int = 4
    data_dir: str = "data"
    vocab_size: int | None = None      # filled from meta.json if left as None

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert (self.d_model // self.n_head) % 2 == 0, "head_dim must be even for RoPE"
        assert isinstance(self.lr, float), "lr must be a float; write 1.0e-3, not 1e-3, in YAML"
        assert self.optimizer in ("adamw", "sgd"), f"unknown optimizer {self.optimizer}"
        assert self.warmup_steps < self.max_steps, "warmup_steps must be < max_steps"


def load_config(path, **overrides):
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    d.update(overrides)                # e.g. load_config(p, data_dir="../data") in a notebook
    cfg = Config(**d)                  # an unknown key in the YAML raises TypeError
    if cfg.vocab_size is None:
        meta = json.loads((Path(cfg.data_dir) / "meta.json").read_text())
        cfg.vocab_size = meta["vocab_size"]
    if cfg.run_name is None:
        cfg.run_name = f"{datetime.now():%Y-%m-%d_%H%M}_{Path(path).stem}"
    return cfg

