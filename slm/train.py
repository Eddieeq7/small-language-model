"""Training loop.

    python -m slm.train --steps 3000 --block_size 128 --n_embd 256 --lr 1e-3 --out runs/main

AdamW with decoupled weight decay (not applied to biases/LayerNorm/embeddings'
position table), linear warmup then cosine decay, gradient clipping, periodic
validation-loss estimation on held-out text, best-checkpoint saving, and a JSONL
log of every eval so curves can be plotted later.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch

from .data import get_batch, prepare
from .model import GPT, Config


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict, cfg: Config, batch_size: int, iters: int, device: str) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # same eval batches every time → comparable curves
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, cfg.block_size, batch_size, device, gen)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    tok, train_ids, val_ids = prepare(args.vocab_size)
    cfg = Config(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                 n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(device)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "pos_emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "pos_emb" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")
    open(log_path, "w").close()
    print(f"device={device} params={model.num_params()/1e6:.2f}M train_tokens={len(train_ids):,} "
          f"val_tokens={len(val_ids):,} cfg={asdict(cfg)}")

    best_val, t0 = float("inf"), time.time()
    splits = {"train": train_ids, "val": val_ids}
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, splits, cfg, args.batch_size, args.eval_iters, device)
            rec = {"step": step, "train_loss": round(losses["train"], 4), "val_loss": round(losses["val"], 4),
                   "lr": lr_at(step, args.lr, args.warmup, args.steps), "elapsed_s": round(time.time() - t0, 1)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec)
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "step": step,
                            "val_loss": best_val}, os.path.join(args.out, "best.pt"))
        if step == args.steps:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
        x, y = get_batch(train_ids, cfg.block_size, args.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    summary = {"cfg": asdict(cfg), "lr": args.lr, "steps": args.steps, "params": model.num_params(),
               "best_val_loss": round(best_val, 4), "seconds": round(time.time() - t0, 1), "device": device}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/main")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--vocab_size", type=int, default=512)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=40)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
