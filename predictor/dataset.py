"""
predictor/dataset.py — PyTorch Dataset for Next-App Prediction

Sliding window extraction from LSApp sessions.
Splits by USER (not time) to test true cold-start generalisation.

Each sample:
  app_ids   LongTensor (seq_len,)      — last N app IDs (0 = pad)
  ctx_vecs  FloatTensor (seq_len, 12)  — context at each step
  user_id   LongTensor ()              — user index
  label     LongTensor ()              — next app index

Augmentation (training only):
  - Gaussian context noise (σ=0.02)
  - Random app dropout (30% chance, 1 step zeroed)
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SEQ_LEN = 10   # look-back window


# ─────────────────────────────────────────────────────────
# Vocabulary builders
# ─────────────────────────────────────────────────────────

def build_vocab(sessions: List[Dict], min_count: int = 5) -> Tuple[Dict, Dict]:
    """
    Build app and user vocabularies from a list of session dicts.

    A session dict must contain:
      'user_id' : str
      'events'  : list of dicts with keys 'app' and 'context'

    Returns:
      app_vocab  : {app_name: int}   — 0 = padding/UNK
      user_vocab : {user_id: int}    — 0 = unknown/global
    """
    app_counts: Dict[str, int] = defaultdict(int)
    user_ids: set = set()

    for sess in sessions:
        if isinstance(sess, dict):
            uid = sess.get("user_id") or sess.get("user") or "anon"
            events = sess.get("events", [])
        else:
            uid = sess[0].get("user_id", "anon") if sess else "anon"
            events = sess

        user_ids.add(uid)
        for ev in events:
            app = ev.get("app") or ev.get("app_name", "UNKNOWN")
            app_counts[app] += 1

    valid_apps = {a for a, c in app_counts.items() if c >= min_count and a != "UNKNOWN"}
    app_vocab  = {app: i + 1 for i, app in enumerate(sorted(valid_apps))}   # 1-indexed
    user_vocab = {uid: i + 1 for i, uid in enumerate(sorted(user_ids))}      # 1-indexed

    return app_vocab, user_vocab


def save_vocab(app_vocab: Dict, user_vocab: Dict, out_dir: str = "data/processed"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/app_vocab.json", "w") as f:
        json.dump(app_vocab, f, indent=2)
    with open(f"{out_dir}/user_vocab.json", "w") as f:
        json.dump(user_vocab, f, indent=2)


def load_vocab(out_dir: str = "data/processed") -> Tuple[Dict, Dict]:
    with open(f"{out_dir}/app_vocab.json") as f:
        app_vocab = json.load(f)
    with open(f"{out_dir}/user_vocab.json") as f:
        user_vocab = json.load(f)
    return app_vocab, user_vocab


# ─────────────────────────────────────────────────────────
# Session splitter — by USER
# ─────────────────────────────────────────────────────────

def split_sessions_by_user(
    sessions: List[Dict],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> Tuple[List, List, List]:
    """
    Split by user (not time) to avoid data leakage and test cold-start.
    Returns train_sessions, val_sessions, test_sessions.
    """
    # Group sessions by user
    by_user: Dict[str, List] = defaultdict(list)
    for sess in sessions:
        uid = sess[0].get("user_id", "anon") if isinstance(sess, list) else sess.get("user_id", "anon")
        by_user[uid].append(sess)

    users = sorted(by_user.keys())
    rng   = random.Random(seed)
    rng.shuffle(users)

    n        = len(users)
    n_train  = int(n * train_frac)
    n_val    = int(n * val_frac)

    train_users = set(users[:n_train])
    val_users   = set(users[n_train:n_train + n_val])
    test_users  = set(users[n_train + n_val:])

    def collect(uids):
        out = []
        for uid in uids:
            out.extend(by_user[uid])
        return out

    return collect(train_users), collect(val_users), collect(test_users)


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────

class AppSequenceDataset(Dataset):
    """
    Sliding window dataset over LSApp sessions.

    Each sample extracts a (seq_len) look-back window and the next app as label.
    Handles sessions stored as plain lists of event dicts (from synthetic_generator
    or lsapp_loader) — no wrapper dict required.
    """

    def __init__(
        self,
        sessions:   List,
        app_vocab:  Dict[str, int],
        user_vocab: Dict[str, int],
        seq_len:    int  = SEQ_LEN,
        augment:    bool = False,
    ):
        self.seq_len   = seq_len
        self.app_vocab = app_vocab
        self.user_vocab= user_vocab
        self.augment   = augment
        self.idx_to_app= {v: k for k, v in app_vocab.items()}
        self.samples: List[Dict] = []
        self._build(sessions)

    # ── Internal ──────────────────────────────────────────

    def _ev_app(self, ev: dict) -> str:
        return ev.get("app") or ev.get("app_name", "UNKNOWN")

    def _ev_ctx(self, ev: dict) -> List[float]:
        raw = ev.get("context", {})
        if isinstance(raw, dict):
            vals = list(raw.values())
        elif isinstance(raw, (list, tuple)):
            vals = list(raw)
        else:
            vals = []
        # Pad / trim to 12
        vals = vals[:12]
        vals += [0.0] * (12 - len(vals))
        return [float(v) for v in vals]

    def _build(self, sessions: List):
        for sess in sessions:
            # Normalise: session can be a list of events or a dict with 'events' key
            if isinstance(sess, dict):
                events = sess.get("events", [])
                uid    = sess.get("user_id", "anon")
            else:
                events = sess
                uid    = events[0].get("user_id", "anon") if events else "anon"

            if len(events) < 2:
                continue

            user_idx = self.user_vocab.get(uid, 0)

            for i in range(1, len(events)):
                start   = max(0, i - self.seq_len)
                window  = events[start:i]
                pad_len = self.seq_len - len(window)

                app_ids  = [0] * pad_len + [
                    self.app_vocab.get(self._ev_app(e), 0) for e in window
                ]
                ctx_vecs = [[0.0] * 12] * pad_len + [
                    self._ev_ctx(e) for e in window
                ]

                label_app = self._ev_app(events[i])
                label     = self.app_vocab.get(label_app, 0)
                if label == 0:
                    continue   # skip unknown next-app labels

                self.samples.append({
                    "app_ids":  torch.LongTensor(app_ids),
                    "ctx_vecs": torch.FloatTensor(ctx_vecs),
                    "user_id":  torch.tensor(user_idx, dtype=torch.long),
                    "label":    torch.tensor(label,    dtype=torch.long),
                    "raw_app":  label_app,
                })

    # ── Augmentation ─────────────────────────────────────

    def _augment(self, sample: dict) -> dict:
        s = {k: v.clone() if isinstance(v, torch.Tensor) else v
             for k, v in sample.items()}
        # 1. Gaussian context noise
        s["ctx_vecs"] = s["ctx_vecs"] + torch.randn_like(s["ctx_vecs"]) * 0.02
        # 2. Random app dropout (30% chance)
        if torch.rand(1).item() < 0.30:
            idx = torch.randint(0, self.seq_len, (1,)).item()
            s["app_ids"][idx] = 0
        return s

    # ── Dataset protocol ──────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self._augment(sample) if self.augment else sample


# ─────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────

def build_dataloaders(
    train_sessions, val_sessions, test_sessions,
    app_vocab:   Dict,
    user_vocab:  Dict,
    batch_size:  int = 64,
    seq_len:     int = SEQ_LEN,
    num_workers: int = 0,   # 0 = main process (safe on Windows)
):
    train_ds = AppSequenceDataset(train_sessions, app_vocab, user_vocab, seq_len, augment=True)
    val_ds   = AppSequenceDataset(val_sessions,   app_vocab, user_vocab, seq_len, augment=False)
    test_ds  = AppSequenceDataset(test_sessions,  app_vocab, user_vocab, seq_len, augment=False)

    kw = dict(num_workers=num_workers, pin_memory=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size,  shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False, **kw)
    test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, **kw)

    return train_dl, val_dl, test_dl
