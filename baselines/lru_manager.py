"""
baselines/lru_manager.py — Least Recently Used Baseline Memory Manager

AX Memory | Samsung AX Hackathon 2026 | PS-03

This is the "naive phone" baseline — exactly how Android manages memory
today without any ML assistance. It reacts to app launches and evicts
whoever was used least recently. No prediction, no awareness of context.

Purpose: establish the "before" KPIs that the RL+LSTM system will beat.
"""

import numpy as np
from typing import List, Optional, Dict
import time


class LRUMemoryManager:
    """
    Deterministic Least Recently Used (LRU) memory manager.

    Policy:
      - On each step: keep all apps unless RAM is full
      - When RAM is full: evict the slot with the oldest last_access_time
      - Never pre-loads (purely reactive)
      - Produces human-readable action log entries (Samsung AX requirement)

    Interface mirrors MemorySimEnv's action-space expectations:
      act() returns np.ndarray of shape (n_slots,) with values 0=keep, 1=evict, 2=preload
    """

    def __init__(self, n_slots: int = 10, name: str = "LRU Baseline"):
        self.n_slots = n_slots
        self.name = name
        self._last_access: List[int] = [-1] * n_slots  # step index
        self._slot_apps: List[Optional[str]] = [None] * n_slots
        self._step = 0
        self.action_log: List[Dict] = []

    def act(self, observation: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        """
        Compute action array from the flattened observation.

        observation: flat float32 vector from MemorySimEnv._get_observation()
        info: optional last step info dict (used for logging)

        Returns: np.ndarray of shape (n_slots,) with values 0 (keep) or 1 (evict)
        """
        action = np.zeros(self.n_slots, dtype=np.int64)

        # Extract last_access times from observation (slot feature index 1 = steps_since)
        # observation layout: [slot0_f0..f4, slot1_f0..f4, ..., ctx..., global...]
        slot_feat_dim = 5
        steps_since = []
        for i in range(self.n_slots):
            base = i * slot_feat_dim
            app_norm = float(observation[base])    # 0 = empty slot
            since    = float(observation[base + 1])
            steps_since.append((i, app_norm, since))

        # Determine if any slot is occupied
        occupied = [(i, app, since) for i, app, since in steps_since if app > 0.0]

        if len(occupied) >= self.n_slots:
            # RAM full — evict LRU slot (largest steps_since = oldest)
            lru_idx = max(occupied, key=lambda x: x[2])[0]
            action[lru_idx] = 1   # evict

            self._log(f"LRU eviction at slot {lru_idx} (steps_since={steps_since[lru_idx][2]:.2f})")

        self._step += 1
        return action

    def reset(self):
        """Reset internal state between episodes."""
        self._last_access = [-1] * self.n_slots
        self._slot_apps = [None] * self.n_slots
        self._step = 0
        self.action_log = []

    def _log(self, msg: str):
        self.action_log.append({
            "step": self._step,
            "timestamp": time.strftime("%H:%M:%S"),
            "manager": self.name,
            "msg": msg,
        })

    @property
    def manager_name(self) -> str:
        return self.name
