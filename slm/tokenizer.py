"""Byte-level BPE tokenizer, trained from scratch.

Starts from the 256 raw byte values (so any UTF-8 text is encodable, no <unk>),
then repeatedly merges the most frequent adjacent pair into a new token.

Like GPT-2, text is first split into chunks with a regex (words, numbers,
punctuation runs, whitespace) and merges never cross chunk boundaries. That
also makes training fast: we count pairs over *unique* chunks weighted by
frequency instead of over the whole corpus.
"""
from __future__ import annotations

import json
import re
from collections import Counter

SPLIT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def _merge(ids: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = merges or []
        self._rebuild()

    def _rebuild(self) -> None:
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        assert vocab_size >= 256
        words = Counter(tuple(w.encode("utf-8")) for w in SPLIT.findall(text))
        merges: list[tuple[int, int]] = []
        for step in range(vocab_size - 256):
            pairs: Counter[tuple[int, int]] = Counter()
            for w, freq in words.items():
                for p in zip(w, w[1:]):
                    pairs[p] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.__getitem__)
            new_id = 256 + step
            merges.append(best)
            words = Counter({_merge(w, best, new_id): f for w, f in words.items()})
            if verbose and step % 64 == 0:
                tok = cls(merges)
                print(f"merge {step:4d}: {tok.vocab[new_id]!r} (count {pairs[best]})")
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = tuple(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge present, exactly as in training.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, 256 + self.ranks[pair])
        self._cache[chunk] = list(ids)
        return self._cache[chunk]

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in SPLIT.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            return cls([tuple(m) for m in json.load(f)["merges"]])
