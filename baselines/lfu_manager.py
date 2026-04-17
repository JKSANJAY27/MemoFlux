"""
baselines/lfu_manager.py — Least Frequently Used Baseline Memory Manager

AX Memory | Samsung AX Hackathon 2026 | PS-03

Maintains a frequency counter per slot. When RAM is full, evicts the app
that has been accessed fewest times since it was loaded. Slight improvement
over LRU for apps with predictable high-frequency patterns (e.g. Chrome).
"""

import numpy as np
from typing import List, Optional, Dict
import time


class LFUMemoryManager:
    """
    Least Frequently Used (LFU) memory manager.

    Policy:
      - Track access count per slot (from observation freq_norm feature)
      - When RAM is full: evict slot with lowest access frequency
      - No pre-loading (reactive only)
    """

    def __init__(self, n_slots: int = 10, name: str = "LFU Baseline"):
        self.n_slots = n_slots
        self.name = name
        self._step = 0
        self.action_log: List[Dict] = []

    def act(self, observation: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        action = np.zeros(self.n_slots, dtype=np.int64)

        slot_feat_dim = 5
        slots = []
        for i in range(self.n_slots):
            base = i * slot_feat_dim
            app_norm  = float(observation[base])      # 0 = empty
            freq_norm = float(observation[base + 2])  # usage frequency
            slots.append((i, app_norm, freq_norm))

        occupied = [(i, app, freq) for i, app, freq in slots if app > 0.0]

        if len(occupied) >= self.n_slots:
            # Evict slot with lowest frequency
            lfu_idx = min(occupied, key=lambda x: x[2])[0]
            action[lfu_idx] = 1
            self._log(f"LFU eviction at slot {lfu_idx} (freq_norm={slots[lfu_idx][2]:.3f})")

        self._step += 1
        return action

    def reset(self):
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
