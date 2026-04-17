"""
predictor/user_profile.py — Per-User Embedding Layer

Learns a unique D-dimensional bias vector per user.
At inference, adds user-specific offset to LSTM output (personalisation).

Cold-start: unknown user → falls back to learned global embedding (index 0).
Scale parameter keeps user bias from dominating sequence features.

Parameters: (n_users + 1) × hidden_dim
  For n_users=292, hidden_dim=128 → 37,504 params (negligible).
"""

import torch
import torch.nn as nn


class UserProfileModule(nn.Module):
    """
    Adds a learned per-user bias to the shared LSTM representation.

    user_ids == 0  →  global / cold-start user (padding_idx ignored for embedding,
                       but the 0-indexed embedding is effectively the "no-user" prior).
    """

    def __init__(self, n_users: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        # Index 0 = global/unknown-user embedding
        self.embedding  = nn.Embedding(n_users + 1, hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(hidden_dim)
        # Learnable scale — start small so pre-training on sequence features dominates
        self.scale      = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        """
        hidden:   (B, hidden_dim)
        user_ids: (B,)  — 0 = unknown/global
        returns:  (B, hidden_dim)
        """
        user_emb = self.embedding(user_ids)      # (B, hidden_dim)
        user_emb = self.dropout(user_emb)
        return self.norm(hidden + self.scale * user_emb)
