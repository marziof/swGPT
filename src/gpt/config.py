# src/gpt/config.py
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
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


def load_config(path, **overrides):
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    d.update(overrides)                # e.g. load_config(p, data_dir="../data") in a notebook
    cfg = Config(**d)                  # an unknown key in the YAML raises TypeError
    if cfg.vocab_size is None:
        meta = json.loads((Path(cfg.data_dir) / "meta.json").read_text())
        cfg.vocab_size = meta["vocab_size"]
    return cfg