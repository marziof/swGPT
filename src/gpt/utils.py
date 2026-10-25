
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"



def save_checkpoint(path, model, opt, sched, step, cfg, g, best_val=math.inf):
    ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "step": step, "best_val": best_val,
            "cfg": asdict(cfg), "gen_state": g.get_state()}
    tmp = str(path) + ".tmp"
    torch.save(ckpt, tmp)
    Path(tmp).replace(path)             # atomic


def load_checkpoint(path, model, opt, sched, g, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    g.set_state(ckpt["gen_state"].cpu())   # generator state must live on CPU
    return ckpt["step"], ckpt["best_val"]



def log(metrics, step, run_path=None, trace=True):
    if trace:
        print(f"step {step}: " + ", ".join(f"{k} {v:.4g}" for k, v in metrics.items()))
    if run_path is not None:
        with open(Path(run_path) / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
    # later: wandb.log(metrics, step=step)


