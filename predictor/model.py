"""
predictor/model.py — ContextAwareLSTM Architecture

Next-app predictor for Samsung AX Memory.
Target: HR@3 ≥ 75% on LSApp test split.

Architecture:
  App Embedding → Context Projection → Input Fusion
  → App Attention → LSTM (2-layer) → Temporal Gating
  → User Profile → Output Head (logits over vocab)

Parameter budget: < 2M total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_gating import TemporalGatingModule
from .app_attention   import AppAttentionModule
from .user_profile    import UserProfileModule


class ContextAwareLSTM(nn.Module):
    """
    Four-component next-app predictor.

    Input per timestep:
      app_id      int        — index of app just launched (0 = padding/UNK)
      context_vec float[12]  — from feature_engineer.py
      user_id     int        — per-user profile lookup

    Components (in forward order):
      1. App Embedding     nn.Embedding(V+1, app_emb_dim)
      2. Context Proj      Linear(12 → ctx_dim) + GELU
      3. Input Fusion      Linear(app_emb_dim + ctx_dim → hidden_dim) + GELU
      4. App Attention     causal self-attention over sequence
      5. LSTM              2-layer bidirectional-like, uses last timestep
      6. Temporal Gating   rescales hidden state by time-of-day
      7. User Profile      adds per-user bias vector
      8. Output Head       Linear(hidden_dim → V)
    """

    def __init__(
        self,
        vocab_size:    int,
        n_users:       int,
        app_emb_dim:   int = 32,
        ctx_dim:       int = 32,
        hidden_dim:    int = 128,
        n_lstm_layers: int = 2,
        dropout:       float = 0.3,
        seq_len:       int = 10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.seq_len    = seq_len

        # ── 1. App Embedding ────────────────────────────────
        self.app_embedding = nn.Embedding(vocab_size + 1, app_emb_dim, padding_idx=0)

        # ── 2. Context Projection ───────────────────────────
        self.ctx_proj = nn.Sequential(
            nn.Linear(12, ctx_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── 3. Input Fusion ─────────────────────────────────
        self.input_fusion = nn.Sequential(
            nn.Linear(app_emb_dim + ctx_dim, hidden_dim),
            nn.GELU(),
        )

        # ── 4. App Attention ────────────────────────────────
        self.app_attention = AppAttentionModule(hidden_dim, seq_len, dropout=0.1)

        # ── 5. LSTM ─────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # ── 6. Temporal Gating ──────────────────────────────
        self.temporal_gate = TemporalGatingModule(hidden_dim)

        # ── 7. User Profile ─────────────────────────────────
        self.user_profile = UserProfileModule(n_users, hidden_dim)

        # ── 8. Output Head ──────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, vocab_size + 1),
        )

        self._init_weights()

    # ───────────────────────────────────────────────────────
    # Weight Initialisation
    # ───────────────────────────────────────────────────────
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "lstm" in name and "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "lstm" in name and "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Embedding — uniform small
        nn.init.uniform_(self.app_embedding.weight, -0.05, 0.05)
        with torch.no_grad():
            self.app_embedding.weight[0].fill_(0)   # zero out padding

    # ───────────────────────────────────────────────────────
    # Forward
    # ───────────────────────────────────────────────────────
    def forward(
        self,
        app_ids:  torch.Tensor,   # (B, S)
        ctx_vecs: torch.Tensor,   # (B, S, 12)
        user_ids: torch.Tensor,   # (B,)
        hidden=None,
    ):
        B, S = app_ids.shape

        # 1. App embedding
        app_embs  = self.app_embedding(app_ids)            # (B, S, E)

        # 2. Context projection
        ctx_proj  = self.ctx_proj(ctx_vecs)                # (B, S, C)

        # 3. Input fusion
        fused     = torch.cat([app_embs, ctx_proj], dim=-1) # (B, S, E+C)
        fused     = self.input_fusion(fused)                # (B, S, H)

        # 4. App attention (causal)
        fused     = self.app_attention(fused)               # (B, S, H)

        # 5. LSTM
        lstm_out, hidden = self.lstm(fused, hidden)         # (B, S, H)
        lstm_out  = self.lstm_dropout(lstm_out)

        # Take last timestep
        last_out  = lstm_out[:, -1, :]                      # (B, H)

        # 6. Temporal gating (use last step's hour features)
        hour_feat = ctx_vecs[:, -1, :2]                     # (B, 2) — hour_sin, hour_cos
        last_out  = self.temporal_gate(last_out, hour_feat) # (B, H)

        # 7. User profile bias
        last_out  = self.user_profile(last_out, user_ids)   # (B, H)

        # 8. Output head
        logits    = self.output_head(last_out)               # (B, V)

        return logits, hidden

    # ───────────────────────────────────────────────────────
    # Inference helpers
    # ───────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_top_k(
        self,
        app_ids:  torch.Tensor,
        ctx_vecs: torch.Tensor,
        user_ids: torch.Tensor,
        k: int = 5,
    ):
        """Returns top-k (index, probability) tensors."""
        self.eval()
        logits, _ = self.forward(app_ids, ctx_vecs, user_ids)
        probs     = F.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, k, dim=-1)
        return top_idx, top_probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"ContextAwareLSTM("
            f"vocab={self.vocab_size}, "
            f"hidden={self.hidden_dim}, "
            f"params={self.count_parameters():,})"
        )
