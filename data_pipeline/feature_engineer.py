"""
data_pipeline/feature_engineer.py — Context Vector Builder

AX Memory | Samsung AX Hackathon 2026 | PS-03

Builds a 12-dimensional context vector for every app-switch event.
For LSApp (which lacks battery/network/location), realistic values
are simulated from timestamps — demonstrating on-device deployment
readiness where these signals ARE available.

Context vector:
  [0]  sin(2π * hour / 24)           — cyclical hour encoding
  [1]  cos(2π * hour / 24)
  [2]  sin(2π * day_of_week / 7)     — cyclical day encoding
  [3]  cos(2π * day_of_week / 7)
  [4]  battery_level / 100.0         — normalized [0,1]
  [5]  is_charging (0 or 1)
  [6]  network_type_wifi (0 or 1)    — one-hot of 3 network types
  [7]  network_type_4g (0 or 1)
  [8]  network_type_5g (0 or 1)
  [9]  session_app_count / 10.0      — apps used so far this session
  [10] log1p(inter_arrival_s) / log1p(3600)  — capped at 1h
  [11] is_weekend (0 or 1)
"""

import numpy as np
from typing import Dict, Any, Optional


class FeatureEngineer:
    """
    Converts raw event metadata into a fixed 12-dimensional context vector.

    Simulation rules for missing signals (LSApp):
    - Battery: 100 - (hour * 3) + N(0, 5), clipped to [10, 100]
    - Network: 70% WiFi during 20:00-08:00, 60% 4G/5G during commute hours
    - is_charging: True with 40% probability during 22:00-07:00
    """

    CONTEXT_DIM = 12

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────

    def build_context(
        self,
        hour: int,
        day_of_week: int,
        inter_arrival_s: float = 0.0,
        session_app_count: int = 0,
        battery: Optional[float] = None,
        is_charging: Optional[bool] = None,
        network: Optional[str] = None,  # "wifi", "4g", "5g"
    ) -> np.ndarray:
        """
        Build the 12-dim context vector.

        Args:
            hour: 0-23
            day_of_week: 0=Monday … 6=Sunday
            inter_arrival_s: seconds since last app open
            session_app_count: apps opened so far in session
            battery: battery % (simulated if None)
            is_charging: charging state (simulated if None)
            network: "wifi" | "4g" | "5g" (simulated if None)
        """
        hour = int(np.clip(hour, 0, 23))
        day_of_week = int(np.clip(day_of_week, 0, 6))

        # Cyclical time encodings
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin  = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos  = np.cos(2 * np.pi * day_of_week / 7)

        # Battery
        if battery is None:
            battery = self._simulate_battery(hour)
        battery_norm = float(np.clip(battery / 100.0, 0.0, 1.0))

        # Charging state
        if is_charging is None:
            is_charging = self._simulate_charging(hour)
        is_charging_f = 1.0 if is_charging else 0.0

        # Network
        if network is None:
            network = self._simulate_network(hour)
        wifi = 1.0 if network == "wifi" else 0.0
        net4g = 1.0 if network == "4g" else 0.0
        net5g = 1.0 if network == "5g" else 0.0

        # Session features
        session_norm = float(np.clip(session_app_count / 10.0, 0.0, 1.0))

        # Inter-arrival time (log-normalised)
        inter_arrival_s = max(0.0, inter_arrival_s)
        inter_arrival_norm = float(np.log1p(inter_arrival_s) / np.log1p(3600.0))
        inter_arrival_norm = float(np.clip(inter_arrival_norm, 0.0, 1.0))

        # Weekend flag
        is_weekend = 1.0 if day_of_week >= 5 else 0.0

        vec = np.array([
            hour_sin, hour_cos,
            dow_sin,  dow_cos,
            battery_norm,
            is_charging_f,
            wifi, net4g, net5g,
            session_norm,
            inter_arrival_norm,
            is_weekend,
        ], dtype=np.float32)

        assert vec.shape == (self.CONTEXT_DIM,), f"Bug: expected {self.CONTEXT_DIM}, got {vec.shape}"
        return vec

    def build_context_dict(self, **kwargs) -> Dict[str, Any]:
        """Returns a dict version of build_context for serialisation."""
        vec = self.build_context(**kwargs)
        keys = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "battery_norm", "is_charging", "network_type_wifi",
            "network_type_4g", "network_type_5g",
            "session_app_count", "inter_arrival_norm", "is_weekend",
        ]
        return dict(zip(keys, vec.tolist()))

    # ─────────────────────────────────────────────────────
    # Simulation helpers (for LSApp missing signals)
    # ─────────────────────────────────────────────────────

    def _simulate_battery(self, hour: int) -> float:
        """
        Simulate battery level using a realistic daily drain model.
        Peak at 07:00 after overnight charge (~95%), trough at 20:00 (~30%).
        """
        base = 100.0 - (hour * 3.0)
        noise = self.rng.normal(0, 5)
        return float(np.clip(base + noise, 10.0, 100.0))

    def _simulate_charging(self, hour: int) -> bool:
        """True with 40% probability during 22:00-07:00 (night charging)."""
        if hour >= 22 or hour <= 7:
            return bool(self.rng.random() < 0.40)
        return False

    def _simulate_network(self, hour: int) -> str:
        """
        Simulate network type based on time-of-day behaviour patterns.
        - 20:00-08:00 (home hours): 70% WiFi
        - 07:00-09:00, 17:00-19:00 (commute): 60% cellular
        - Other: 50/50
        """
        r = self.rng.random()
        is_home = (hour >= 20) or (hour <= 8)
        is_commute = (7 <= hour <= 9) or (17 <= hour <= 19)

        if is_home:
            if r < 0.70: return "wifi"
            if r < 0.85: return "4g"
            return "5g"
        elif is_commute:
            if r < 0.40: return "wifi"
            if r < 0.70: return "4g"
            return "5g"
        else:
            if r < 0.50: return "wifi"
            if r < 0.75: return "4g"
            return "5g"

    # ─────────────────────────────────────────────────────
    # Batch processing
    # ─────────────────────────────────────────────────────

    def process_events(self, events: list) -> list:
        """Add context vector to a list of event dicts in-place."""
        for i, ev in enumerate(events):
            hour = ev.get("hour_of_day", 12)
            dow  = ev.get("day_of_week", 0)
            ia   = ev.get("inter_arrival_s", 0.0)
            cnt  = ev.get("session_position", i)
            ctx  = self.build_context_dict(
                hour=hour,
                day_of_week=dow,
                inter_arrival_s=ia,
                session_app_count=cnt,
            )
            ctx["is_morning"] = 6 <= hour <= 11
            ctx["is_evening"] = 18 <= hour <= 23
            ctx["session_length_so_far"] = cnt
            ev["context"] = ctx
        return events
