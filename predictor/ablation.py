"""
predictor/ablation.py — Ablation Study: 4 Model Variants

Trains 4 variants to isolate each component's contribution to HR@3.
Proves to judges that each component adds measurable value.

Variants:
  V1: Vanilla LSTM (no attention, no gating, no profile)
  V2: + App Attention
  V3: + Temporal Gating
  V4: Full model (all components)

Usage:
    python -m predictor.ablation
    python -m predictor.ablation --epochs 10 --quick
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from predictor.dataset             import build_vocab, build_dataloaders, split_sessions_by_user
from predictor.trainer             import train
from predictor.evaluator           import evaluate


# ─────────────────────────────────────────────────────────
# Stripped-down variants
# ─────────────────────────────────────────────────────────

class VanillaLSTM(nn.Module):
    """V1: bare LSTM, no attention/gating/profile."""

    def __init__(self, vocab_size, n_users, hidden_dim=128, dropout=0.3, seq_len=10, **kw):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len    = seq_len

        self.app_emb  = nn.Embedding(vocab_size + 1, 32, padding_idx=0)
        self.ctx_proj = nn.Sequential(nn.Linear(12, 32), nn.GELU())
        self.fuse     = nn.Sequential(nn.Linear(64, hidden_dim), nn.GELU())
        self.lstm     = nn.LSTM(hidden_dim, hidden_dim, num_layers=2,
                                dropout=dropout, batch_first=True)
        self.drop     = nn.Dropout(dropout)
        self.head     = nn.Linear(hidden_dim, vocab_size)
        self._init()

    def _init(self):
        for n, p in self.named_parameters():
            if "weight" in n and p.dim() >= 2: nn.init.xavier_uniform_(p)
            elif "bias" in n: nn.init.zeros_(p)

    def forward(self, app_ids, ctx_vecs, user_ids, hidden=None):
        e = self.app_emb(app_ids)
        c = self.ctx_proj(ctx_vecs)
        x = self.fuse(torch.cat([e, c], -1))
        out, hidden = self.lstm(x, hidden)
        last = self.drop(out[:, -1, :])
        return self.head(last), hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMWithAttention(VanillaLSTM):
    """V2: + App Attention."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        from predictor.app_attention import AppAttentionModule
        self.attn = AppAttentionModule(self.hidden_dim, self.seq_len)

    def forward(self, app_ids, ctx_vecs, user_ids, hidden=None):
        e = self.app_emb(app_ids)
        c = self.ctx_proj(ctx_vecs)
        x = self.fuse(torch.cat([e, c], -1))
        x = self.attn(x)                          # ← app attention
        out, hidden = self.lstm(x, hidden)
        last = self.drop(out[:, -1, :])
        return self.head(last), hidden


class LSTMWithAttentionAndGating(LSTMWithAttention):
    """V3: + Temporal Gating."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        from predictor.temporal_gating import TemporalGatingModule
        self.gate = TemporalGatingModule(self.hidden_dim)

    def forward(self, app_ids, ctx_vecs, user_ids, hidden=None):
        e = self.app_emb(app_ids)
        c = self.ctx_proj(ctx_vecs)
        x = self.fuse(torch.cat([e, c], -1))
        x = self.attn(x)
        out, hidden = self.lstm(x, hidden)
        last = self.drop(out[:, -1, :])
        hour = ctx_vecs[:, -1, :2]
        last = self.gate(last, hour)               # ← temporal gating
        return self.head(last), hidden


# V4 = full ContextAwareLSTM (imported directly)


# ─────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────

VARIANTS = {
    "V1_VanillaLSTM":           VanillaLSTM,
    "V2_PlusAttention":         LSTMWithAttention,
    "V3_PlusTemporalGating":    LSTMWithAttentionAndGating,
    "V4_FullModel":             None,   # loaded below
}


def run_ablation(n_epochs=15, patience=5, quick=False):
    from predictor.model   import ContextAwareLSTM

    # ── Data ──────────────────────────────────────
    try:
        from data_pipeline.lsapp_loader import LSAppLoader
        loader = LSAppLoader()
        t, v, te = loader.load_splits()
        sessions = t + v + te
        if not sessions: raise ValueError("empty")
    except Exception:
        from data_pipeline.synthetic_generator import SyntheticGenerator
        sessions = SyntheticGenerator(n_users=100, days_per_user=20, seed=42).generate_all()

    if quick:
        sessions = sessions[: max(len(sessions) // 5, 20)]
        n_epochs = 5; patience = 3

    train_s, val_s, _ = split_sessions_by_user(sessions, seed=42)
    app_vocab, user_vocab = build_vocab(train_s, min_count=3)
    V = len(app_vocab); N = len(user_vocab)
    train_dl, val_dl, _ = build_dataloaders(train_s, val_s, [], app_vocab, user_vocab,
                                             batch_size=64, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    VARIANTS["V4_FullModel"] = ContextAwareLSTM

    results = {}
    for variant_name, ModelCls in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Training {variant_name}")
        print(f"{'='*60}")

        model = ModelCls(vocab_size=V, n_users=N, hidden_dim=128, dropout=0.3, seq_len=10)
        history = train(
            model, train_dl, val_dl,
            device=device, n_epochs=n_epochs, lr=3e-3, patience=patience,
            checkpoint_dir=f"checkpoints/ablation/{variant_name}",
            verbose=True,
        )
        best_hr3 = max(history.get("val_hr3", [0]))
        results[variant_name] = {
            "best_val_hr3": round(best_hr3, 4),
            "params":       model.count_parameters(),
        }
        print(f"  {variant_name}: HR@3 = {best_hr3:.1%}")

    # ── Report ────────────────────────────────────
    print("\n=== ABLATION RESULTS ===")
    print(f"{'Variant':<35} {'HR@3':>8} {'Params':>10}")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<35} {res['best_val_hr3']:.1%}  {res['params']:>10,}")

    out_path = "checkpoints/ablation_results.json"
    Path("checkpoints").mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results → {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="AX Memory — Ablation Study")
    parser.add_argument("--epochs",  type=int, default=15)
    parser.add_argument("--patience",type=int, default=5)
    parser.add_argument("--quick",   action="store_true")
    args = parser.parse_args()
    run_ablation(args.epochs, args.patience, args.quick)


if __name__ == "__main__":
    main()
