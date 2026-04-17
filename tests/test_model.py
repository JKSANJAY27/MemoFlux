"""
tests/test_model.py — 10 unit tests for ContextAwareLSTM and sub-modules
"""

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from predictor.model           import ContextAwareLSTM
from predictor.temporal_gating import TemporalGatingModule
from predictor.app_attention   import AppAttentionModule
from predictor.user_profile    import UserProfileModule


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def small_model():
    return ContextAwareLSTM(vocab_size=50, n_users=20, hidden_dim=64, seq_len=10)


@pytest.fixture
def batch():
    B, S = 4, 10
    return {
        "app_ids":  torch.randint(0, 50, (B, S)),
        "ctx_vecs": torch.randn(B, S, 12),
        "user_ids": torch.randint(0, 20, (B,)),
    }


# ── Tests ─────────────────────────────────────────────────

def test_model_output_shape(small_model, batch):
    """Logits shape must be (batch, vocab_size)."""
    logits, _ = small_model(batch["app_ids"], batch["ctx_vecs"], batch["user_ids"])
    assert logits.shape == (4, 50), f"Expected (4,50), got {logits.shape}"


def test_model_parameter_count_under_2m():
    """Full-size model must have < 2M parameters."""
    model = ContextAwareLSTM(vocab_size=100, n_users=300, hidden_dim=128, seq_len=10)
    n = model.count_parameters()
    assert n < 2_000_000, f"Parameter count {n:,} exceeds 2M limit"


def test_predict_top_k_returns_valid_probs(small_model, batch):
    """top-K probs must all be in [0,1] and sorted descending."""
    indices, probs = small_model.predict_top_k(
        batch["app_ids"], batch["ctx_vecs"], batch["user_ids"], k=5
    )
    assert indices.shape == (4, 5)
    assert probs.shape   == (4, 5)
    # All probs in valid range
    assert (probs >= 0).all() and (probs <= 1).all()
    # Descending order
    for i in range(4):
        p = probs[i].tolist()
        assert p == sorted(p, reverse=True), "Probs not sorted descending"


def test_temporal_gating_changes_output():
    """Morning (hour=8) and night (hour=22) gates must produce different outputs."""
    gate = TemporalGatingModule(hidden_dim=64)
    gate.eval()
    h = torch.randn(1, 64)

    def hour_enc(h_val):
        s = math.sin(2 * math.pi * h_val / 24)
        c = math.cos(2 * math.pi * h_val / 24)
        return torch.tensor([[s, c]])

    with torch.no_grad():
        out_morning = gate(h, hour_enc(8))
        out_night   = gate(h, hour_enc(22))

    assert not torch.allclose(out_morning, out_night, atol=1e-4), \
        "Gating produced identical output for hour=8 vs hour=22"


def test_app_attention_causal_mask():
    """Each position must NOT attend to any future position."""
    attn = AppAttentionModule(hidden_dim=32, seq_len=8)
    attn.eval()

    # Feed distinct inputs so attention weights are non-trivial
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        out = attn(x)   # (1, 8, 32)

    # The causal mask is internal; verify output shape and no NaN
    assert out.shape == (1, 8, 32)
    assert not torch.isnan(out).any(), "NaN in attention output"

    # Verify mask is properly upper-triangular in the registered buffer
    mask = attn._causal_mask   # (S, S) bool — True = masked (blocked)
    S = mask.shape[0]
    for i in range(S):
        for j in range(S):
            if j > i:
                assert mask[i, j].item(), f"mask[{i},{j}] should be True (blocked)"
            else:
                assert not mask[i, j].item(), f"mask[{i},{j}] should be False (allowed)"


def test_user_profile_cold_start():
    """Unknown user_id=0 must produce non-NaN output."""
    up    = UserProfileModule(n_users=50, hidden_dim=64)
    h     = torch.randn(2, 64)
    uids  = torch.zeros(2, dtype=torch.long)    # both are "unknown"
    out   = up(h, uids)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


def test_no_nan_in_forward(small_model, batch):
    """Forward pass must never produce NaN."""
    logits, _ = small_model(batch["app_ids"], batch["ctx_vecs"], batch["user_ids"])
    assert not torch.isnan(logits).any(), "NaN in model logits"
    assert not torch.isinf(logits).any(), "Inf in model logits"


def test_padding_token_does_not_dominate(small_model):
    """Sequences consisting entirely of padding (app_id=0) should still produce
    a valid distribution (no collapsed output)."""
    B, S = 2, 10
    app_ids  = torch.zeros(B, S, dtype=torch.long)   # all padding
    ctx_vecs = torch.zeros(B, S, 12)
    user_ids = torch.zeros(B, dtype=torch.long)
    logits, _ = small_model(app_ids, ctx_vecs, user_ids)
    probs = torch.softmax(logits, dim=-1)
    assert not torch.isnan(probs).any()
    assert (probs > 0).any()


def test_training_one_epoch_no_error():
    """Training for 1 epoch on tiny synthetic data must not raise."""
    from torch.utils.data import DataLoader, TensorDataset
    from predictor.trainer import train_one_epoch, TopKCrossEntropyLoss
    import torch.optim as optim

    V = 20; B = 8; S = 10
    ds = TensorDataset(
        torch.randint(0, V, (B*4, S)),         # app_ids
        torch.randn(B*4, S, 12),               # ctx_vecs
        torch.randint(0, 5, (B*4,)),           # user_ids
        torch.randint(1, V, (B*4,)),           # labels
    )

    def collate(batch):
        app_ids, ctx_vecs, user_ids, labels = zip(*batch)
        return {
            "app_ids":  torch.stack(app_ids),
            "ctx_vecs": torch.stack(ctx_vecs),
            "user_id":  torch.stack(user_ids),
            "label":    torch.stack(labels),
        }

    dl    = DataLoader(ds, batch_size=B, collate_fn=collate)
    model = ContextAwareLSTM(vocab_size=V, n_users=5, hidden_dim=32, seq_len=S)
    opt   = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = TopKCrossEntropyLoss()

    loss = train_one_epoch(model, dl, opt, loss_fn, device="cpu")
    assert isinstance(loss, float) and loss > 0


def test_model_repr():
    """__repr__ should return a non-empty string."""
    m = ContextAwareLSTM(vocab_size=30, n_users=10, hidden_dim=32)
    r = repr(m)
    assert "ContextAwareLSTM" in r
    assert "vocab=" in r
