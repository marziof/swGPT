import torch

import torch.optim.adamw as adamw
import torch.optim.sgd as SGD


def build_optimizer(model, cfg):
    # weight decay on 2D+ weight matrices only; none on 1D params (norm gains)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def build_scheduler(opt, cfg):
    warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_steps)
    cosine = CosineAnnealingLR(opt, T_max=cfg.max_steps - cfg.warmup_steps,
                               eta_min=cfg.lr * cfg.min_lr_ratio)
    return SequentialLR(opt, [warmup, cosine], milestones=[cfg.warmup_steps])