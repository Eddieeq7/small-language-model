import torch

from slm.model import GPT, CausalSelfAttention, Config
from slm.tokenizer import BPETokenizer

TEXT = "the cat sat on the mat. the cat ate the rat. that cat is fat!\n" * 20


def tiny_cfg(**kw) -> Config:
    base = dict(vocab_size=300, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    base.update(kw)
    return Config(**base)


# ── tokenizer ────────────────────────────────────────────────────────────────

def test_tokenizer_roundtrip_ascii_and_unicode():
    tok = BPETokenizer.train(TEXT, 300)
    for s in [TEXT, "unseen words here", "héllo wörld — ünïcode ✓", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_tokenizer_learns_merges_that_compress():
    tok = BPETokenizer.train(TEXT, 300)
    # Stops early once every chunk is a single token (merges never cross chunks).
    assert 256 < tok.vocab_size <= 300
    assert all(len(tok._encode_chunk(w)) == 1 for w in [" cat", " the", " mat"])
    assert len(tok.encode(TEXT)) < len(TEXT.encode()) / 2
    assert b" cat" in tok.vocab.values()


def test_tokenizer_save_load(tmp_path):
    tok = BPETokenizer.train(TEXT, 280)
    p = tmp_path / "tok.json"
    tok.save(str(p))
    tok2 = BPETokenizer.load(str(p))
    assert tok2.encode(TEXT) == tok.encode(TEXT)


# ── attention ────────────────────────────────────────────────────────────────

def test_causal_mask_blocks_future_tokens():
    """Changing token t must not change the outputs at positions < t."""
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 300, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 300
    lx, _ = model(x)
    ly, _ = model(y)
    assert torch.allclose(lx[0, :10], ly[0, :10], atol=1e-6)
    assert not torch.allclose(lx[0, 10:], ly[0, 10:])


def test_attention_rows_are_distributions_and_upper_triangle_zero():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    attn = CausalSelfAttention(cfg).eval()
    attn.record_attn = True
    attn(torch.randn(2, 12, cfg.n_embd))
    a = attn.last_attn
    assert a.shape == (2, cfg.n_head, 12, 12)
    assert torch.allclose(a.sum(-1), torch.ones(2, cfg.n_head, 12), atol=1e-5)
    assert torch.all(torch.triu(a, diagonal=1) == 0)


def test_matches_pytorch_reference_attention():
    """Hand-written attention == F.scaled_dot_product_attention(is_causal=True)."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 12, cfg.n_embd)
    q, k, v = attn.qkv(x).split(cfg.n_embd, dim=2)
    sh = lambda t: t.view(2, 12, cfg.n_head, -1).transpose(1, 2)
    ref = torch.nn.functional.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
    ref = attn.proj(ref.transpose(1, 2).reshape(2, 12, cfg.n_embd))
    assert torch.allclose(attn(x), ref, atol=1e-5)


# ── model / training ─────────────────────────────────────────────────────────

def test_initial_loss_is_near_uniform():
    torch.manual_seed(0)
    model = GPT(tiny_cfg())
    x = torch.randint(0, 300, (4, 16))
    _, loss = model(x, x)
    assert abs(loss.item() - torch.log(torch.tensor(300.0)).item()) < 0.5


def test_weight_tying():
    model = GPT(tiny_cfg())
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_can_overfit_a_single_batch():
    torch.manual_seed(0)
    model = GPT(tiny_cfg())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 300, (2, 16))
    y = torch.roll(x, -1, dims=1)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_extends_sequence_and_respects_block_size():
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=40)
    assert out.shape == (1, 41)
