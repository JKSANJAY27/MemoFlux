"""
data_pipeline/synthetic_generator.py — Synthetic App Usage Generator

AX Memory | Samsung AX Hackathon 2026 | PS-03

Generates statistically realistic app-switch sessions when real data is
insufficient for RL training or when stress-testing edge cases.

Statistical basis (LSApp-derived):
  - Session length:     Poisson(λ=5.46) — LSApp mean app switches / session
  - Inter-arrival time: Weibull(shape=0.7, scale=120) seconds
  - Battery decay:      Linear 85% → 30% with N(0,3) noise
  - Archetype mix:      morning_commuter=20%, social=30%, work=25%,
                        night_owl=15%, mixed=10%

Usage:
  python data_pipeline/synthetic_generator.py --n_users 100 --days 30
"""

import argparse
import json
import logging
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# App pools per archetype
# ─────────────────────────────────────────────────────────
ARCHETYPE_APP_POOLS: Dict[str, List[Tuple[str, float]]] = {
    "morning_commuter": [
        ("Chrome", 0.20), ("Gmail", 0.18), ("Maps", 0.15),
        ("Spotify", 0.12), ("WhatsApp", 0.10), ("Calendar", 0.08),
        ("LinkedIn", 0.07), ("Messages", 0.05), ("Settings", 0.03),
        ("Instagram", 0.02),
    ],
    "social": [
        ("Instagram", 0.25), ("TikTok", 0.20), ("Twitter", 0.15),
        ("WhatsApp", 0.12), ("Snapchat", 0.10), ("YouTube", 0.08),
        ("Chrome", 0.05), ("Facebook", 0.03), ("Messages", 0.02),
    ],
    "work": [
        ("Chrome", 0.25), ("Gmail", 0.22), ("Calendar", 0.12),
        ("LinkedIn", 0.10), ("Messages", 0.08), ("Maps", 0.07),
        ("Settings", 0.05), ("WhatsApp", 0.04), ("Phone", 0.04),
        ("Contacts", 0.03),
    ],
    "night_owl": [
        ("YouTube", 0.28), ("Netflix", 0.22), ("Spotify", 0.18),
        ("Instagram", 0.12), ("TikTok", 0.10), ("Twitter", 0.05),
        ("WhatsApp", 0.03), ("Chrome", 0.02),
    ],
    "mixed": [
        ("Chrome", 0.18), ("WhatsApp", 0.15), ("YouTube", 0.12),
        ("Instagram", 0.10), ("Gmail", 0.10), ("Settings", 0.08),
        ("Maps", 0.07), ("Spotify", 0.07), ("TikTok", 0.07),
        ("Camera", 0.06),
    ],
}

# Peak hour distributions per archetype (Gaussian mixture params)
ARCHETYPE_HOUR_PARAMS: Dict[str, List[Tuple[float, float, float]]] = {
    # (mean_hour, std, weight)
    "morning_commuter": [(8.0, 1.5, 0.5), (18.0, 1.5, 0.3), (12.5, 1.0, 0.2)],
    "social":           [(12.5, 1.0, 0.3), (20.0, 1.5, 0.5), (16.0, 1.0, 0.2)],
    "work":             [(10.0, 2.0, 0.5), (14.5, 1.5, 0.3), (17.5, 1.0, 0.2)],
    "night_owl":        [(22.5, 1.5, 0.6), (13.0, 1.5, 0.25), (9.0, 1.0, 0.15)],
    "mixed":            [(12.0, 4.0, 1.0)],
}

ARCHETYPE_DIST = {
    "morning_commuter": 0.20,
    "social":           0.30,
    "work":             0.25,
    "night_owl":        0.15,
    "mixed":            0.10,
}


class SyntheticGenerator:
    """
    Produces synthetic user-day sessions matching LSApp statistical patterns.

    Output schema (per event):
      {"user_id", "app", "timestamp_unix", "hour_of_day", "day_of_week",
       "inter_arrival_s", "session_position", "context"}
    """

    def __init__(
        self,
        n_users: int = 500,
        days_per_user: int = 30,
        archetype_distribution: Optional[Dict[str, float]] = None,
        seed: int = 42,
        output_dir: str = "data/synthetic",
    ):
        self.n_users = n_users
        self.days_per_user = days_per_user
        self.archetype_dist = archetype_distribution or ARCHETYPE_DIST
        self.rng = np.random.default_rng(seed)
        self.fe = FeatureEngineer(seed=seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────

    def generate_all(self) -> List[List[Dict]]:
        """Generate sessions for all users and save to output_dir."""
        all_sessions: List[List[Dict]] = []
        archetypes = list(self.archetype_dist.keys())
        weights = [self.archetype_dist[a] for a in archetypes]

        logger.info(
            "Generating synthetic data: %d users × %d days", self.n_users, self.days_per_user
        )

        for uid in range(self.n_users):
            archetype = self.rng.choice(archetypes, p=weights)
            user_id = f"synthetic_u{uid:04d}"
            base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

            for day_offset in range(self.days_per_user):
                date = base_date + timedelta(days=day_offset)
                session = self._generate_day(user_id, archetype, date)
                if len(session) >= 3:
                    all_sessions.append(session)

        logger.info("Generated %d sessions total", len(all_sessions))

        # Save
        out_path = self.output_dir / "synthetic_sessions.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(all_sessions, f)
        logger.info("Saved synthetic sessions to %s", out_path)

        return all_sessions

    def generate_user_day(self, archetype: str = "mixed") -> List[Dict]:
        """Generate a single user-day on demand."""
        date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        return self._generate_day("demo_user", archetype, date)

    # ─────────────────────────────────────────────────────
    # Internal generation
    # ─────────────────────────────────────────────────────

    def _generate_day(
        self, user_id: str, archetype: str, date: datetime
    ) -> List[Dict]:
        """Generate one user-day session."""
        n_events = max(3, int(self.rng.poisson(5.46)))  # LSApp mean
        pool = ARCHETYPE_APP_POOLS.get(archetype, ARCHETYPE_APP_POOLS["mixed"])
        apps    = [a[0] for a in pool]
        probs   = np.array([a[1] for a in pool])
        probs  /= probs.sum()

        # Generate event timestamps using Weibull inter-arrivals
        hour_params = ARCHETYPE_HOUR_PARAMS.get(archetype, ARCHETYPE_HOUR_PARAMS["mixed"])
        first_hour  = self._sample_hour(hour_params)
        base_ts     = int(date.timestamp()) + first_hour * 3600

        events: List[Dict] = []
        ts = base_ts
        prev_ts = None

        for pos in range(n_events):
            # Inter-arrival: Weibull(shape=0.7, scale=120)
            if pos > 0:
                ia = float(self.rng.weibull(0.7) * 120)
                ts = ts + int(ia)
            else:
                ia = 0.0

            hour = (datetime.utcfromtimestamp(ts).hour) % 24
            dow  = datetime.utcfromtimestamp(ts).weekday()
            app  = str(self.rng.choice(apps, p=probs))

            ctx = self.fe.build_context_dict(
                hour=hour,
                day_of_week=dow,
                inter_arrival_s=ia,
                session_app_count=pos,
            )
            ctx["is_morning"] = 6 <= hour <= 11
            ctx["is_evening"] = 18 <= hour <= 23
            ctx["session_length_so_far"] = pos

            ev = {
                "user_id": user_id,
                "app": app,
                "app_name": app,
                "timestamp_unix": ts,
                "hour_of_day": hour,
                "day_of_week": dow,
                "inter_arrival_s": ia,
                "session_position": pos,
                "archetype": archetype,
                "context": ctx,
            }
            events.append(ev)

        return events

    def _sample_hour(self, params: List[Tuple[float, float, float]]) -> int:
        """Sample an hour from a Gaussian mixture."""
        weights = np.array([p[2] for p in params])
        weights /= weights.sum()
        comp_idx = self.rng.choice(len(params), p=weights)
        mean, std, _ = params[comp_idx]
        hour = self.rng.normal(mean, std)
        return int(np.clip(round(hour), 0, 23))


# ─────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="AX Memory Synthetic Data Generator")
    parser.add_argument("--n_users", type=int, default=500)
    parser.add_argument("--days",    type=int, default=30)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  type=str, default="data/synthetic")
    args = parser.parse_args()

    gen = SyntheticGenerator(
        n_users=args.n_users,
        days_per_user=args.days,
        seed=args.seed,
        output_dir=args.output,
    )
    sessions = gen.generate_all()
    print(f"\n✓ Generated {len(sessions)} sessions → {args.output}/synthetic_sessions.pkl")
    # Print a sample session
    if sessions:
        sample = sessions[0]
        print(f"\nSample session ({len(sample)} events):")
        for ev in sample[:3]:
            print(f"  {ev['hour_of_day']:02d}:xx  {ev['app']:<15}  arch={ev['archetype']}")


if __name__ == "__main__":
    main()
