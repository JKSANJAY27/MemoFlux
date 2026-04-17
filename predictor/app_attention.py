"""
predictor/app_attention.py — Causal Self-Attention over App History

Intuition: if user opened WhatsApp → Instagram → WhatsApp, the second
WhatsApp event should weight higher — it signals an ongoing conversation.
The attention mechanism learns these sequential patterns.

Single-head causal attention with residual + LayerNorm.
Causal mask prevents future leakage (step i cannot see step j > i).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AppAttentionModule(nn.Module):
    """
    Lightweight single-head causal self-attention over the app sequence.

    Input/output: (B, S, hidden_dim)
    Parameters:   4 × hidden_dim² (no bias in projections = fewer params)
    """

    def __init__(self, hidden_dim: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len    = seq_len
        self.scale      = math.sqrt(hidden_dim)

        self.q_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(hidden_dim)

        # Pre-build causal mask (upper triangle = -inf)
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, hidden_dim)
        returns: (B, S, hidden_dim) — attention-enhanced sequence
        """
        B, S, H = x.shape

        Q = self.q_proj(x)   # (B, S, H)
        K = self.k_proj(x)   # (B, S, H)
        V = self.v_proj(x)   # (B, S, H)

        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale   # (B, S, S)

        # Apply causal mask (handle dynamic seq lengths gracefully)
        if S <= self.seq_len:
            mask = self._causal_mask[:S, :S]
        else:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        weights  = F.softmax(scores, dim=-1)
        weights  = self.dropout(weights)

        attended = torch.bmm(weights, V)           # (B, S, H)
        attended = self.out_proj(attended)

        # Residual + LayerNorm
        return self.norm(x + attended)
