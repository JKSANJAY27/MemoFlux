"""
tests/test_baselines.py — Baseline Manager Tests
AX Memory | Samsung AX Hackathon 2026
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.lru_manager import LRUMemoryManager
from baselines.lfu_manager import LFUMemoryManager
from baselines.static_priority import StaticPriorityManager
from env.memory_sim_env import MemorySimEnv, DEVICE_PROFILES


def _make_obs(n_slots=10, occupied_slots=None):
    """Build a synthetic observation vector for testing managers."""
    slot_feat_dim = 5
    ctx_dim = 12
    global_dim = 3
    obs = np.zeros(n_slots * slot_feat_dim + ctx_dim + global_dim, dtype=np.float32)
    occupied_slots = occupied_slots or []
    for idx in occupied_slots:
        base = idx * slot_feat_dim
        obs[base]     = 0.5    # app_idx_norm (non-zero = occupied)
        obs[base + 1] = float(idx) / n_slots  # steps_since
        obs[base + 2] = 0.3    # freq_norm
    return obs


def test_lru_no_eviction_below_capacity():
    """LRU should not evict when RAM has free slots."""
    mgr = LRUMemoryManager(n_slots=10)
    # Only 3 of 10 slots occupied
    obs = _make_obs(n_slots=10, occupied_slots=[0, 1, 2])
    action = mgr.act(obs)
    assert sum(action == 1) == 0, "Should not evict when slots free"


def test_lru_evicts_when_full():
    """LRU should evict exactly one slot when all slots are occupied."""
    mgr = LRUMemoryManager(n_slots=10)
    obs = _make_obs(n_slots=10, occupied_slots=list(range(10)))
    action = mgr.act(obs)
    assert sum(action == 1) == 1, f"Should evict exactly one slot, evicted: {sum(action==1)}"


def test_lfu_evicts_lowest_frequency():
    """LFU should prefer evicting the slot with the lowest frequency."""
    n_slots = 6
    mgr = LFUMemoryManager(n_slots=n_slots)
    obs = np.zeros(n_slots * 5 + 12 + 3, dtype=np.float32)
    # Fill all slots with different frequencies
    freqs = [0.9, 0.7, 0.05, 0.8, 0.6, 0.4]  # slot 2 has lowest freq
    for i, f in enumerate(freqs):
        base = i * 5
        obs[base]     = 0.5  # occupied
        obs[base + 2] = f    # freq_norm

    action = mgr.act(obs)
    # Should evict slot 2 (lowest frequency = 0.05)
    assert action[2] == 1, f"Expected eviction at slot 2, got actions: {action}"


def test_static_priority_evicts_lowest_priority():
    """Static priority evicts lowest-priority app when full."""
    mgr = StaticPriorityManager(n_slots=4)
    # All 4 slots occupied; we construct obs so slot 3 maps to "Settings" (low priority)
    all_apps = sorted({"Chrome", "WhatsApp", "YouTube", "Settings"})
    app_to_idx = {a: i for i, a in enumerate(sorted(mgr._app_list))}

    obs = np.zeros(4 * 5 + 12 + 3, dtype=np.float32)
    apps_in_slots = ["Chrome", "WhatsApp", "YouTube", "Settings"]
    for i, app in enumerate(apps_in_slots):
        base = i * 5
        idx = app_to_idx.get(app, 0)
        n = len(mgr._app_list)
        obs[base] = idx / max(n - 1, 1)  # app_idx_norm (non-zero)

    action = mgr.act(obs)
    # Settings has the lowest priority score → should be evicted
    evicted_slot = np.argwhere(action == 1).flatten()
    assert len(evicted_slot) == 1


def test_all_managers_return_correct_shape():
    """All managers must return action of shape (n_slots,)."""
    n_slots = 8
    obs = _make_obs(n_slots=n_slots, occupied_slots=list(range(n_slots)))
    for mgr_cls in [LRUMemoryManager, LFUMemoryManager, StaticPriorityManager]:
        mgr = mgr_cls(n_slots=n_slots)
        action = mgr.act(obs)
        assert action.shape == (n_slots,), f"{mgr_cls.__name__}: wrong shape {action.shape}"
        assert all(a in [0, 1, 2] for a in action), f"{mgr_cls.__name__}: invalid values {action}"


def test_managers_survive_empty_obs():
    """Managers should handle completely empty observation gracefully."""
    n_slots = 10
    obs = np.zeros(n_slots * 5 + 12 + 3, dtype=np.float32)  # all zeros = all empty
    for mgr_cls in [LRUMemoryManager, LFUMemoryManager, StaticPriorityManager]:
        mgr = mgr_cls(n_slots=n_slots)
        action = mgr.act(obs)
        assert action.shape == (n_slots,)
        assert sum(action == 1) == 0, f"{mgr_cls.__name__} should not evict empty slots"
