import torch
from gpt.model import GPT
from gpt.optim import build_optimizer, build_scheduler
from gpt.data import get_batch, get_val_batches
from gpt.utils import get_device, save_checkpoint, load_checkpoint, log
from gpt.config import load_config

from pathlib import Path
import argparse
import json
import yaml
from dataclasses import asdict


def train(cfg, trace=True, save=True):
    device = get_device()
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    model = GPT(cfg=cfg).to(device)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)

    data_dir = Path(cfg.data_dir)
    train_path, val_path = data_dir / "train.bin", data_dir / "val.bin"
    val_data = get_val_batches(val_path,  cfg.batch_size, cfg.context_len, cfg.eval_batches, device, seed=cfg.seed)
    run_path = Path(cfg.resume).parent if cfg.resume else Path(cfg.run_dir) / cfg.run_name
    run_path.mkdir(parents=True, exist_ok=True)

    if not cfg.resume:
        (run_path / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    start_step, best_val = 0, float("inf")
    if cfg.resume:
        start_step, best_val = load_checkpoint(cfg.resume, model, opt, sched, g, device)
        print(f"resumed from {cfg.resume} at step {start_step}")
    if cfg.overfit:
        x, y = get_batch(train_path, cfg.batch_size, cfg.context_len, g, device)

    history = []
    for step in range(start_step, cfg.max_steps):
        if not cfg.overfit:
            x, y = get_batch(train_path, cfg.batch_size, cfg.context_len, g, device)
        train_loss, grad_norm = train_step(model, opt, x, y, cfg.grad_clip)
        sched.step()

        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            val_loss = evaluate(model, val_data)
            history.append((step, train_loss, val_loss))
            log({"train_loss": train_loss, "val_loss": val_loss, "grad_norm": grad_norm, "lr": sched.get_last_lr()[0]},
                step, run_path, trace)
            if save and val_loss < best_val:
                best_val = val_loss
                save_checkpoint(run_path / "best.pt", model, opt, sched, step + 1, cfg, g, best_val)

        if save and (step + 1) % cfg.ckpt_interval == 0:
            save_checkpoint(run_path / "last.pt", model, opt, sched, step + 1, cfg, g, best_val)

    if save:
        save_checkpoint(run_path / "last.pt", model, opt, sched, cfg.max_steps, cfg, g, best_val)
    return history



def train_step(model, optimizer, x, y, grad_clip):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, targets=y)
    
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item(), grad_norm.item()


@torch.no_grad()
def evaluate(model, val_data):
    model.eval()
    losses = [model(x, targets=y)[1].item() for x,y in val_data]
    model.train()
    return sum(losses) / len(losses)






def parse_overrides(pairs):
    out = {}
    for s in pairs:
        k, v = s.split("=", 1)
        try:
            out[k] = json.loads(v)       # numbers, true/false, null (1e-3 parses as a float)
        except json.JSONDecodeError:
            out[k] = v                   # plain strings like paths
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    train(load_config(args.config, **parse_overrides(args.set)))


if __name__ == "__main__":
    main()