"""
tests/test_data_pipeline.py — Data Pipeline Tests
AX Memory | Samsung AX Hackathon 2026
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.feature_engineer import FeatureEngineer
from data_pipeline.synthetic_generator import SyntheticGenerator
from data_pipeline.session_builder import SessionBuilder
from env.memory_sim_env import APP_LOAD_PROFILES


def test_feature_engineer_context_dim():
    fe = FeatureEngineer(seed=0)
    ctx = fe.build_context(hour=12, day_of_week=3)
    assert ctx.shape == (12,)


def test_feature_engineer_cyclical_encoding():
    """Cyclical hour encoding must be in [-1, 1]."""
    fe = FeatureEngineer()
    for h in range(24):
        ctx = fe.build_context(hour=h, day_of_week=0)
        assert -1 <= ctx[0] <= 1  # hour_sin
        assert -1 <= ctx[1] <= 1  # hour_cos


def test_feature_engineer_weekend_flag():
    fe = FeatureEngineer()
    weekday = fe.build_context(hour=10, day_of_week=0)  # Monday
    weekend = fe.build_context(hour=10, day_of_week=5)  # Saturday
    assert weekday[11] == 0.0
    assert weekend[11] == 1.0


def test_synthetic_generator_archetype_distribution():
    """All generated sessions have a valid archetype."""
    gen = SyntheticGenerator(n_users=10, days_per_user=5, seed=1)
    sessions = gen.generate_all()
    valid_archetypes = {"morning_commuter", "social", "work", "night_owl", "mixed"}
    for sess in sessions:
        for ev in sess:
            assert ev.get("archetype") in valid_archetypes


def test_session_builder_splits_on_inactivity():
    """Events >5min apart should create new sub-sessions."""
    builder = SessionBuilder()
    events = [
        {"user_id": "u1", "app": "Chrome", "timestamp_unix": 1000, "hour_of_day": 8, "day_of_week": 1, "inter_arrival_s": 0},
        {"user_id": "u1", "app": "WhatsApp", "timestamp_unix": 1060, "hour_of_day": 8, "day_of_week": 1, "inter_arrival_s": 60},
        # 10 min gap — should be new sub-session
        {"user_id": "u1", "app": "YouTube", "timestamp_unix": 1660, "hour_of_day": 8, "day_of_week": 1, "inter_arrival_s": 600},
        {"user_id": "u1", "app": "Instagram", "timestamp_unix": 1720, "hour_of_day": 8, "day_of_week": 1, "inter_arrival_s": 60},
        {"user_id": "u1", "app": "Maps", "timestamp_unix": 1780, "hour_of_day": 8, "day_of_week": 1, "inter_arrival_s": 60},
    ]
    # Direct sub-session test
    sub_sessions = builder._split_sub_sessions(events)
    assert len(sub_sessions) == 2  # split at gap


def test_session_builder_filters_short_sessions():
    """Sessions with <3 events are filtered."""
    builder = SessionBuilder()
    events = [
        {"user_id": "u1", "app": "Chrome", "timestamp_unix": 1000, "hour_of_day": 8, "day_of_week": 1, "inter_arrival_s": 0},
        {"user_id": "u1", "app": "Gmail", "timestamp_unix": 1060, "hour_of_day": 8, "day_of_week": 1, "inter_arrival_s": 60},
    ]
    episodes = builder.build(events)
    assert len(episodes) == 0  # too short


def test_synthetic_generator_inter_arrival_realistic():
    """Inter-arrival times should be positive."""
    gen = SyntheticGenerator(n_users=5, days_per_user=3, seed=77)
    sessions = gen.generate_all()
    for sess in sessions:
        for i, ev in enumerate(sess[1:], 1):
            assert ev["inter_arrival_s"] >= 0, f"Negative inter-arrival at pos {i}"
