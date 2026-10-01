"""Hyperparameter sweep over context length, embedding size, and learning rate.

    python -m slm.sweep --steps 1500

Varies one axis at a time around a baseline (rather than a full grid) so each
result isolates one effect for the same compute budget. Writes runs/sweep/results.json.
"""
from __future__ import annotations

import argparse
import json
import os

from .train import parser as train_parser
from .train import train

BASELINE = dict(block_size=128, n_embd=256, lr=1e-3)
AXES = {
    "block_size": [32, 64, 128, 256],
    "n_embd": [128, 256, 384],
    "lr": [3e-4, 1e-3, 3e-3],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--out", default="runs/sweep")
    a = p.parse_args()

    results, done = [], set()
    for axis, values in AXES.items():
        for v in values:
            cfg = dict(BASELINE, **{axis: v})
            key = tuple(sorted(cfg.items()))
            name = f"ctx{cfg['block_size']}_emb{cfg['n_embd']}_lr{cfg['lr']:g}"
            if key not in done:
                args = train_parser().parse_args([
                    "--out", os.path.join(a.out, name), "--steps", str(a.steps),
                    "--block_size", str(cfg["block_size"]), "--n_embd", str(cfg["n_embd"]),
                    "--lr", str(cfg["lr"]), "--eval_every", str(max(1, a.steps // 10)),
                ])
                summary = train(args)
                done.add(key)
                cache = summary
            else:
                with open(os.path.join(a.out, name, "summary.json")) as f:
                    cache = json.load(f)
            results.append({"axis": axis, "value": v, "run": name, "best_val_loss": cache["best_val_loss"],
                            "params": cache["params"], "seconds": cache["seconds"]})
            print(results[-1])

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"baseline": BASELINE, "steps": a.steps, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
