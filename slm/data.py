"""Corpus download, tokenizer training, and train/val split."""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import torch

from .tokenizer import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_corpus() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(CORPUS_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def prepare(vocab_size: int = 512, val_frac: float = 0.1) -> tuple[BPETokenizer, np.ndarray, np.ndarray]:
    """Train (or load) the tokenizer and return (tokenizer, train_ids, val_ids).

    The split is a contiguous tail of the corpus, not random lines, so validation
    text is genuinely unseen. The tokenizer is trained on the training split only.
    """
    text = load_corpus()
    cut = int(len(text) * (1 - val_frac))
    train_text, val_text = text[:cut], text[cut:]

    tok_path = os.path.join(DATA_DIR, f"bpe_{vocab_size}.json")
    if os.path.exists(tok_path):
        tok = BPETokenizer.load(tok_path)
    else:
        tok = BPETokenizer.train(train_text, vocab_size, verbose=True)
        tok.save(tok_path)

    ids_path = os.path.join(DATA_DIR, f"ids_{vocab_size}.npz")
    if os.path.exists(ids_path):
        z = np.load(ids_path)
        return tok, z["train"], z["val"]
    train = np.array(tok.encode(train_text), dtype=np.uint16)
    val = np.array(tok.encode(val_text), dtype=np.uint16)
    np.savez(ids_path, train=train, val=val)
    return tok, train, val


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str, gen: torch.Generator | None = None):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
