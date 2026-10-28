
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

def get_device(pref="auto"):
    if pref == "cuda":
        assert torch.cuda.is_available(), "device=cuda requested but CUDA is not available"
        return "cuda"
    if pref == "mps":
        assert torch.backends.mps.is_available(), "device=mps requested but MPS is not available"
        return "mps"
    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():            # "auto"
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_checkpoint(path, model, opt, sched, step, cfg, g, best_val=math.inf, scaler=None):
    ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "step": step, "best_val": best_val,
            "cfg": asdict(cfg), "gen_state": g.get_state(), 
            "scaler": scaler.state_dict() if scaler is not None else None}
    tmp = str(path) + ".tmp"
    torch.save(ckpt, tmp)
    Path(tmp).replace(path)             # atomic


def load_checkpoint(path, model, opt, sched, g, device, scaler=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    g.set_state(ckpt["gen_state"].cpu())   # generator state must live on CPU
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt["best_val"]


def load_pretrained(path, model, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def log(metrics, step, run_path=None, trace=True, use_wandb=False):
    if trace:
        print(f"step {step}: " + ", ".join(f"{k} {v:.4g}" for k, v in metrics.items()))
    if run_path is not None:
        with open(Path(run_path) / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
    if use_wandb:
        import wandb
        wandb.log(metrics, step=step)


import subprocess

def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def amp_context(device, cfg):
    enabled = cfg.amp == "fp16" and device == "cuda"
    return torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16, enabled=enabled)


