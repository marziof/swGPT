import torch
from gpt.model import GPT
from gpt.optim import build_optimizer, build_scheduler
from gpt.data import get_batch, get_val_batches
from gpt.utils import get_device, save_checkpoint, load_checkpoint, log, amp_context, get_git_hash
from gpt.config import load_config

from pathlib import Path
import argparse
import json
import yaml
import time
from dataclasses import asdict


def train(cfg, trace=True, save=True):
    device = get_device()
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    model = GPT(cfg=cfg).to(device)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)

    amp_ctx = amp_context(device, cfg)
    use_scaler = cfg.amp == "fp16" and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    if cfg.amp == "fp16" and not use_scaler:
        print("amp=fp16 requested but device is not CUDA; running in fp32")

    data_dir = Path(cfg.data_dir)
    train_path, val_path = data_dir / "train.bin", data_dir / "val.bin"
    val_data = get_val_batches(val_path,  cfg.batch_size, cfg.context_len, cfg.eval_batches, device, seed=cfg.seed)
    run_path = Path(cfg.resume).parent if cfg.resume else Path(cfg.run_dir) / cfg.run_name
    run_path.mkdir(parents=True, exist_ok=True)

    if cfg.wandb:
        import wandb
        id_file = run_path / "wandb_id.txt"
        run_id = id_file.read_text() if id_file.exists() else wandb.util.generate_id()
        id_file.write_text(run_id)
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, id=run_id, resume="allow",
                config={**asdict(cfg), "git_commit": get_git_hash()})

    if not cfg.resume:
        (run_path / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
    start_step, best_val = 0, float("inf")
    if cfg.resume:
        start_step, best_val = load_checkpoint(cfg.resume, model, opt, sched, g, device, scaler)
        print(f"resumed from {cfg.resume} at step {start_step}")
    if cfg.overfit:
        x, y = get_batch(train_path, cfg.batch_size, cfg.context_len, g, device)
    t_last, step_last = time.time(), start_step - 1
    history = []
    for step in range(start_step, cfg.max_steps):
        if not cfg.overfit:
            x, y = get_batch(train_path, cfg.batch_size, cfg.context_len, g, device)
        train_loss, grad_norm = train_step(model, opt, x, y, cfg.grad_clip, amp_ctx, scaler)
        sched.step()

        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            tok_s = cfg.batch_size * cfg.context_len * (step - step_last) / (time.time() - t_last)
            val_loss = evaluate(model, val_data, amp_ctx)
            metrics = {"train_loss": train_loss, "val_loss": val_loss, "grad_norm": grad_norm,
                    "lr": sched.get_last_lr()[0], "tok_s": tok_s}
            if scaler is not None:
                metrics["amp_scale"] = scaler.get_scale()
            history.append((step, train_loss, val_loss))
            log(metrics, step, run_path, trace, cfg.wandb)
            if save and val_loss < best_val:
                best_val = val_loss
                save_checkpoint(run_path / "best.pt", model, opt, sched, step + 1, cfg, g, best_val, scaler=scaler)
            t_last, step_last = time.time(), step

        if save and (step + 1) % cfg.ckpt_interval == 0:
            save_checkpoint(run_path / "last.pt", model, opt, sched, step + 1, cfg, g, best_val, scaler=scaler)

    if save:
        save_checkpoint(run_path / "last.pt", model, opt, sched, cfg.max_steps, cfg, g, best_val, scaler=scaler)
    if cfg.wandb:
        wandb.finish()
    return history


def train_step(model, opt, x, y, grad_clip, amp_ctx, scaler=None):
    model.train()
    opt.zero_grad(set_to_none=True)
    with amp_ctx:
        _, loss = model(x, targets=y)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                  # must unscale BEFORE clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)                      # silently skips the step if grads have inf/nan
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    return loss.item(), grad_norm.item()


@torch.no_grad()
def evaluate(model, val_data, amp_ctx):
    model.eval()
    with amp_ctx:
        losses = [model(x, targets=y)[1].item() for x, y in val_data]
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