"""
predictor/predictor_interface.py — Clean predict() API consumed by MemorySimEnv

Usage:
    predictor = PredictorInterface.load("checkpoints/best_model.pt", app_vocab, user_vocab)
    predictions = predictor.predict("user_42", context_vec_12d)
    # → [("YouTube", 0.76), ("WhatsApp", 0.51), ...]

    predictor.update_history("user_42", "Chrome", context_vec_12d)  # after each app open
"""

import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .dataset import SEQ_LEN


class PredictorInterface:
    """
    Thread-safe inference wrapper around ContextAwareLSTM.

    Maintains rolling (app_id, context) buffers per user so the model
    always sees the last SEQ_LEN events — no external state management required.
    """

    def __init__(
        self,
        model,
        app_vocab:  Dict[str, int],
        user_vocab: Dict[str, int],
        device:     str = "cpu",
        top_k:      int = 5,
    ):
        self.model      = model.to(device)
        self.model.eval()
        self.device     = device
        self.top_k      = top_k
        self.app_vocab  = app_vocab
        self.user_vocab = user_vocab
        self.idx_to_app = {v: k for k, v in app_vocab.items()}

        # Per-user rolling buffers  (user_id → deque)
        self._app_bufs: Dict[str, deque] = {}
        self._ctx_bufs: Dict[str, deque] = {}

    # ── Factory ───────────────────────────────────────────

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        app_vocab:   Dict[str, int],
        user_vocab:  Dict[str, int],
        device:      str = "cpu",
    ) -> "PredictorInterface":
        """Reconstruct model from checkpoint and return a ready predictor."""
        from .model import ContextAwareLSTM

        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg  = ckpt.get("model_config", {})

        model = ContextAwareLSTM(
            vocab_size = cfg.get("vocab_size",  len(app_vocab)),
            n_users    = cfg.get("n_users",     len(user_vocab)),
            hidden_dim = cfg.get("hidden_dim",  128),
            seq_len    = cfg.get("seq_len",     SEQ_LEN),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, app_vocab, user_vocab, device)

    @classmethod
    def from_model(
        cls,
        model,
        app_vocab:  Dict[str, int],
        user_vocab: Dict[str, int],
        device:     str = "cpu",
    ) -> "PredictorInterface":
        """Wrap an already-instantiated model."""
        return cls(model, app_vocab, user_vocab, device)

    # ── Buffer management ─────────────────────────────────

    def _init_user(self, user_id: str):
        if user_id not in self._app_bufs:
            self._app_bufs[user_id] = deque([0]      * SEQ_LEN, maxlen=SEQ_LEN)
            self._ctx_bufs[user_id] = deque([[0.]*12]* SEQ_LEN, maxlen=SEQ_LEN)

    def update_history(self, user_id: str, app_name: str, context_vec: List[float]):
        """Call after each app launch to keep rolling buffer current."""
        self._init_user(user_id)
        app_idx = self.app_vocab.get(app_name, 0)
        ctx     = list(context_vec)[:12]
        ctx    += [0.0] * (12 - len(ctx))
        self._app_bufs[user_id].append(app_idx)
        self._ctx_bufs[user_id].append(ctx)

    def reset_user(self, user_id: str):
        """Clear history for a user (e.g. on episode reset)."""
        if user_id in self._app_bufs:
            del self._app_bufs[user_id]
            del self._ctx_bufs[user_id]

    # ── Inference ─────────────────────────────────────────

    def predict(
        self,
        user_id:     str,
        context_vec: List[float],
    ) -> List[Tuple[str, float]]:
        """
        Returns top-K (app_name, probability) for user_id given current context.
        Unknown users → global embedding (cold-start).
        Target: < 15ms CPU.
        """
        self._init_user(user_id)

        app_ids_list = list(self._app_bufs[user_id])
        ctx_list     = list(self._ctx_bufs[user_id])

        # Build tensors (batch=1)
        app_t  = torch.tensor([app_ids_list], dtype=torch.long)         # (1, S)
        ctx_t  = torch.tensor([ctx_list],     dtype=torch.float32)      # (1, S, 12)
        uid_t  = torch.tensor([self.user_vocab.get(user_id, 0)],
                               dtype=torch.long)                          # (1,)

        with torch.no_grad():
            logits, _ = self.model(
                app_t.to(self.device),
                ctx_t.to(self.device),
                uid_t.to(self.device),
            )
            probs       = F.softmax(logits[0], dim=-1)
            top_probs, top_idx = torch.topk(probs, min(self.top_k, probs.shape[-1]))

        results = []
        for idx, prob in zip(top_idx.tolist(), top_probs.tolist()):
            app = self.idx_to_app.get(idx, "UNKNOWN")
            results.append((app, round(float(prob), 4)))
        return results

    # ── Explainability ────────────────────────────────────

    def get_explanation(
        self,
        user_id:     str,
        predictions: List[Tuple[str, float]],
        context_vec: List[float],
    ) -> str:
        """
        Human-readable explanation of the top prediction.
        Powers the Decision Log in the dashboard (Samsung's explainability requirement).
        """
        if not predictions:
            return "No prediction available."

        top_app, top_prob = predictions[0]

        # Decode hour from sinusoidal encoding
        hour_sin = context_vec[0] if len(context_vec) > 0 else 0.0
        hour_cos = context_vec[1] if len(context_vec) > 1 else 1.0
        hour = int((math.atan2(hour_sin, hour_cos) / (2 * math.pi) * 24) % 24)

        time_ctx = (
            "morning commute" if 6 <= hour <= 9 else
            "lunch break"     if 11 <= hour <= 13 else
            "evening commute" if 17 <= hour <= 19 else
            "evening wind-down" if 20 <= hour <= 23 else
            "off-peak"
        )

        # Last 3 known apps from buffer
        recent = [
            self.idx_to_app.get(aid, "?")
            for aid in list(self._app_bufs.get(user_id, deque()))[-3:]
            if aid != 0
        ]
        recent_str = " → ".join(recent) if recent else "no recent history"

        battery   = context_vec[4] if len(context_vec) > 4 else 0.5
        is_wifi   = context_vec[6] if len(context_vec) > 6 else 0.0
        network   = "WiFi" if is_wifi > 0.5 else "cellular"

        return (
            f"Pre-loading {top_app} ({top_prob:.0%} confidence) · "
            f"{time_ctx} (hour≈{hour}h) · "
            f"Recent: {recent_str} · "
            f"Battery={battery:.0%} · {network}"
        )

    # ── Vocab accessors ───────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.app_vocab)

    def app_name(self, idx: int) -> str:
        return self.idx_to_app.get(idx, "UNKNOWN")
