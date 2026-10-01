# Small Language Model from Scratch

A GPT-style decoder-only transformer in PyTorch, written without
`nn.MultiheadAttention`, `scaled_dot_product_attention`, or a pretrained tokenizer.

| piece | file | what's in it |
|---|---|---|
| Tokenizer | `slm/tokenizer.py` | byte-level BPE trained from scratch; GPT-2-style regex pre-split; merges never cross chunks |
| Model | `slm/model.py` | token + learned position embeddings → N pre-norm blocks (hand-written causal multi-head attention, GELU MLP) → LayerNorm → weight-tied head |
| Training | `slm/train.py` | AdamW (no decay on biases/LN/pos-emb), linear warmup + cosine decay, grad clipping, periodic train/val loss on fixed eval batches, best-val checkpointing, JSONL log |
| Sweep | `slm/sweep.py` | context length × embedding size × learning rate, one axis at a time around a baseline |
| Export | `slm/export.py` | loss curve, per-layer/per-head attention matrices for a prompt, next-token distribution, samples |
| Tests | `tests/test_slm.py` | tokenizer round-trip (incl. unicode), causal-mask leak test, attention rows sum to 1, hand-written attention == PyTorch SDPA, initial loss ≈ ln V, single-batch overfit |

```
tokens ─► tok_emb + pos_emb ─► N × [ x += Attn(LN(x)) ; x += MLP(LN(x)) ] ─► LN ─► head (tied) ─► logits
```

## Run

```sh
python -m venv .venv && .venv/bin/pip install torch numpy pytest
.venv/bin/python -m pytest -q tests
.venv/bin/python -m slm.train --steps 5000 --out runs/main      # downloads Tiny Shakespeare on first run
.venv/bin/python -m slm.export --ckpt runs/main/best.pt --out export.json
.venv/bin/python -m slm.sweep --steps 1500
```

Data: Tiny Shakespeare (~1.1M characters). The validation set is the contiguous last
10% of the corpus, and the tokenizer is trained on the training split only, so
validation text is genuinely unseen.

See [`RESULTS.md`](RESULTS.md) for measured numbers.
