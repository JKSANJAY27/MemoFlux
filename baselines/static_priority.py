"""
baselines/static_priority.py — Frequency-Ranked Static Cache Baseline

AX Memory | Samsung AX Hackathon 2026 | PS-03

Pre-computes a global frequency ranking of apps from training data and
always keeps the top-N apps in RAM. Never evicts high-frequency apps
regardless of recency. The "best traditional algorithm" in the comparison.

Why it still loses to the ML system:
  - It doesn't adapt to per-user patterns
  - It ignores time-of-day context (Chrome at 8am vs 11pm)
  - It cannot pre-load based on predicted intent
  - It treats all users identically
"""

import numpy as np
from typing import Dict, List, Optional
import time

from env.memory_sim_env import APP_LOAD_PROFILES

# Global app frequency ranking (derived from LSApp statistics)
# Higher score = more frequently used across the dataset
DEFAULT_PRIORITY_SCORES: Dict[str, float] = {
    "Chrome":      0.95,
    "WhatsApp":    0.92,
    "YouTube":     0.88,
    "Instagram":   0.84,
    "Gmail":       0.80,
    "Maps":        0.75,
    "Spotify":     0.70,
    "TikTok":      0.68,
    "Twitter":     0.65,
    "Telegram":    0.62,
    "Facebook":    0.58,
    "Snapchat":    0.52,
    "Netflix":     0.50,
    "Samsung Pay": 0.45,
    "LinkedIn":    0.42,
    "Camera":      0.40,
    "Calendar":    0.35,
    "Messages":    0.30,
    "Gallery":     0.25,
    "Settings":    0.20,
    "Phone":       0.15,
    "Contacts":    0.12,
    "Clock":       0.10,
    "Bixby":       0.08,
    "Kakao":       0.05,
    "UNKNOWN":     0.02,
}


class StaticPriorityManager:
    """
    Static priority cache. Evicts the lowest-priority app when RAM is full.
    Priority = global frequency rank (same for every user, every context).

    This is the best achievable without any per-user or per-context learning.
    """

    def __init__(
        self,
        n_slots: int = 10,
        priority_scores: Optional[Dict[str, float]] = None,
        name: str = "Static Priority Baseline",
    ):
        self.n_slots = n_slots
        self.name = name
        self.priority_scores = priority_scores or DEFAULT_PRIORITY_SCORES
        self._app_list = sorted(APP_LOAD_PROFILES.keys())
        self._step = 0
        self.action_log: List[Dict] = []

    def act(self, observation: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        action = np.zeros(self.n_slots, dtype=np.int64)

        slot_feat_dim = 5
        n_apps = len(self._app_list)
        slots = []

        for i in range(self.n_slots):
            base = i * slot_feat_dim
            app_norm = float(observation[base])
            if app_norm > 0.0:
                # Recover approximate app_idx from normalised value
                app_idx = min(int(round(app_norm * (n_apps - 1))), n_apps - 1)
                app = self._app_list[app_idx]
            else:
                app = None
            slots.append((i, app))

        occupied = [(i, app) for i, app in slots if app is not None]

        if len(occupied) >= self.n_slots:
            # Evict lowest priority
            lowest = min(occupied, key=lambda x: self.priority_scores.get(x[1], 0.0))
            action[lowest[0]] = 1
            self._log(
                f"Static eviction slot {lowest[0]}: {lowest[1]} "
                f"(priority={self.priority_scores.get(lowest[1], 0):.2f})"
            )

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
