"""Export artifacts from a trained checkpoint for visualization.

    python -m slm.export --ckpt runs/main/best.pt --out export.json

Writes: config, loss curve, tokenization of a prompt (showing BPE pieces), the
per-layer/per-head attention matrices for that prompt, and sampled generations.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from .data import prepare
from .model import GPT, Config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main/best.pt")
    p.add_argument("--out", default="export.json")
    p.add_argument("--prompt", default="ROMEO:\nBut soft, what light through yonder window breaks?")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = Config(**ck["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    tok, _, _ = prepare(cfg.vocab_size)

    ids = tok.encode(a.prompt)[: cfg.block_size]
    for blk in model.blocks:
        blk.attn.record_attn = True
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids]))
    attn = [[[[round(v, 4) for v in row] for row in head] for head in blk.attn.last_attn[0].tolist()]
            for blk in model.blocks]
    top_next = torch.topk(torch.softmax(logits[0, -1], -1), 8)

    torch.manual_seed(a.seed)
    start = torch.tensor([tok.encode("\n")])
    samples = [tok.decode(model.generate(start, 200, temperature=0.8, top_k=40)[0].tolist()[1:])
               for _ in range(a.samples)]

    run_dir = os.path.dirname(a.ckpt)
    curve = [json.loads(line) for line in open(os.path.join(run_dir, "log.jsonl"))]
    summary = json.load(open(os.path.join(run_dir, "summary.json")))

    out = {
        "config": ck["cfg"], "params": model.num_params(), "best_step": ck["step"],
        "best_val_loss": round(ck["val_loss"], 4), "summary": summary, "curve": curve,
        "prompt": a.prompt, "tokens": [tok.decode([i]) for i in ids], "token_ids": ids,
        "attention": attn,  # [layer][head][query][key]
        "next_token": [{"token": tok.decode([i]), "p": round(pv, 4)}
                       for pv, i in zip(top_next.values.tolist(), top_next.indices.tolist())],
        "samples": samples,
        "merges_preview": [tok.vocab[256 + i].decode("utf-8", "replace") for i in range(40)],
    }
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}: {len(ids)} tokens, {cfg.n_layer}×{cfg.n_head} attention maps")
    for s in samples:
        print("---\n" + s)


if __name__ == "__main__":
    main()
