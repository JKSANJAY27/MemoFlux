"""
data_pipeline/lsapp_loader.py — LSApp Dataset Parser

AX Memory | Samsung AX Hackathon 2026 | PS-03

Loads and parses the LSApp dataset (Aliannejadi et al., ACM TOIS 2021)
into simulation-ready session lists for MemorySimEnv.

LSApp TSV columns: userId, sessionId, timestamp, appName, eventType
eventType values: "open", "close", "interaction"

Processing pipeline:
  1. Load TSV, parse timestamps as Unix epoch (ms → s)
  2. Filter: keep only eventType == "open" (app launches only)
  3. Filter: remove system apps via SYSTEM_APP_BLACKLIST
  4. Map rare apps (< 50 occurrences) → "UNKNOWN"
  5. Normalize app names to APP_LOAD_PROFILES keys
  6. Build sessions: group by (userId, date) — one user-day = one episode
  7. Split 70/15/15 by user (no temporal leakage)
  8. Save processed output to data/processed/
"""

import os
import pickle
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .feature_engineer import FeatureEngineer
from env.app_registry import LSAPP_NAME_MAP, SYSTEM_APP_BLACKLIST

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
RARE_APP_THRESHOLD = 50  # apps with fewer occurrences → UNKNOWN
RAW_PATH   = "data/raw/lsapp.tsv"
PROC_DIR   = "data/processed"

# Column names observed in LSApp TSV
# Actual columns: userId  sessionId  timestamp  appName  eventType
LSAPP_COLS = ["userId", "sessionId", "timestamp", "appName", "eventType"]


def normalize_app_name(raw_name: str, freq_map: Optional[Dict[str, int]] = None) -> str:
    """
    Map an LSApp raw app identifier to a canonical APP_LOAD_PROFILES key.

    Strategy:
      1. Direct lookup in LSAPP_NAME_MAP
      2. Package prefix matching
      3. Rare app → UNKNOWN (if freq_map provided and count < threshold)
      4. Otherwise → UNKNOWN
    """
    raw_lower = str(raw_name).lower().strip()

    # Blacklisted system processes
    if raw_lower in SYSTEM_APP_BLACKLIST:
        return None  # caller should skip this event

    # Direct lookup (case-insensitive)
    for pkg, canon in LSAPP_NAME_MAP.items():
        if raw_lower == pkg.lower():
            return canon

    # Prefix matching
    for pkg, canon in LSAPP_NAME_MAP.items():
        if raw_lower.startswith(pkg.lower()[:len(pkg) // 2]):
            return canon

    # Frequency filter
    if freq_map is not None:
        if freq_map.get(raw_name, 0) < RARE_APP_THRESHOLD:
            return "UNKNOWN"

    return "UNKNOWN"


class LSAppLoader:
    """
    Loads and processes the LSApp dataset into MemorySimEnv-compatible sessions.

    Usage:
        loader = LSAppLoader()
        train, val, test = loader.load_splits()
    """

    def __init__(
        self,
        tsv_path: str = RAW_PATH,
        proc_dir: str = PROC_DIR,
        seed: int = 42,
    ):
        self.tsv_path = tsv_path
        self.proc_dir = Path(proc_dir)
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.fe = FeatureEngineer(seed=seed)
        self._df: Optional[pd.DataFrame] = None
        self._freq_map: Dict[str, int] = {}

    # ─────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────

    def load_splits(
        self, force_reprocess: bool = False
    ) -> Tuple[List, List, List]:
        """
        Return (train_sessions, val_sessions, test_sessions).
        Uses cached pickle if available.
        """
        cache_path = self.proc_dir / "lsapp_splits.pkl"
        if cache_path.exists() and not force_reprocess:
            logger.info("Loading cached LSApp splits from %s", cache_path)
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        all_sessions = self._build_all_sessions()
        splits = self._split_by_user(all_sessions)

        with open(cache_path, "wb") as f:
            pickle.dump(splits, f)
        logger.info("Saved processed splits to %s", cache_path)

        return splits

    def get_statistics(self) -> Dict:
        """Return dataset statistics for the dashboard Dataset Explorer tab."""
        df = self._load_raw()
        return {
            "total_records": len(df),
            "n_users": df["userId"].nunique(),
            "n_unique_apps_raw": df["appName"].nunique(),
            "date_range": {
                "start": pd.to_datetime(df["timestamp"], unit="ms").min().strftime("%Y-%m-%d"),
                "end":   pd.to_datetime(df["timestamp"], unit="ms").max().strftime("%Y-%m-%d"),
            },
            "top_apps": df[df["eventType"] == "open"]["appName"]
                          .value_counts()
                          .head(15)
                          .to_dict(),
        }

    # ─────────────────────────────────────────────────────
    # Internal pipeline
    # ─────────────────────────────────────────────────────

    def _load_raw(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        if not os.path.exists(self.tsv_path):
            raise FileNotFoundError(
                f"LSApp dataset not found at {self.tsv_path}. "
                "Run `python setup_data.py` to download it."
            )

        logger.info("Loading LSApp TSV from %s ...", self.tsv_path)
        df = pd.read_csv(
            self.tsv_path,
            sep="\t",
            header=0,
            names=LSAPP_COLS,
            dtype={"userId": str, "sessionId": str, "appName": str, "eventType": str},
            low_memory=False,
        )

        # Timestamps: LSApp stores ms epoch
        df["timestamp_s"] = pd.to_numeric(df["timestamp"], errors="coerce") / 1000.0
        df = df.dropna(subset=["timestamp_s"])
        df["dt"] = pd.to_datetime(df["timestamp_s"], unit="s", utc=True)
        df["date"] = df["dt"].dt.date.astype(str)
        df["hour"] = df["dt"].dt.hour
        df["day_of_week"] = df["dt"].dt.dayofweek

        self._df = df
        logger.info("Loaded %d raw records", len(df))
        return df

    def _build_all_sessions(self) -> Dict[str, List[List[Dict]]]:
        """Returns {user_id: [session_event_list, ...]} after full pipeline."""
        df = self._load_raw()

        # Step 1: Keep only "open" events
        df = df[df["eventType"] == "open"].copy()
        logger.info("After open-filter: %d records", len(df))

        # Step 2: Build frequency map, normalize app names
        self._freq_map = df["appName"].value_counts().to_dict()
        df["app_canonical"] = df["appName"].apply(
            lambda x: normalize_app_name(x, self._freq_map)
        )

        # Step 3: Remove blacklisted / None
        df = df[df["app_canonical"].notna()].copy()
        logger.info("After normalization: %d records", len(df))

        # Step 4: Sort
        df = df.sort_values(["userId", "timestamp_s"])

        # Step 5: Group by (userId, date) → sessions
        user_sessions: Dict[str, List[List[Dict]]] = {}

        for (user_id, date), group in tqdm(
            df.groupby(["userId", "date"]), desc="Building sessions"
        ):
            events = self._build_events(group, user_id, date)
            if len(events) < 3:
                continue  # too short for meaningful training
            if user_id not in user_sessions:
                user_sessions[user_id] = []
            user_sessions[user_id].append(events)

        logger.info(
            "Built sessions for %d users, %d total episodes",
            len(user_sessions),
            sum(len(v) for v in user_sessions.values()),
        )
        return user_sessions

    def _build_events(self, group: pd.DataFrame, user_id: str, date: str) -> List[Dict]:
        """Convert a day's app opens into a list of event dicts."""
        events = []
        prev_ts = None

        for pos, (_, row) in enumerate(group.iterrows()):
            ts = float(row["timestamp_s"])
            inter = (ts - prev_ts) if prev_ts is not None else 0.0
            prev_ts = ts

            ctx = self.fe.build_context_dict(
                hour=int(row["hour"]),
                day_of_week=int(row["day_of_week"]),
                inter_arrival_s=inter,
                session_app_count=pos,
            )
            ctx["is_morning"] = 6 <= int(row["hour"]) <= 11
            ctx["is_evening"] = 18 <= int(row["hour"]) <= 23
            ctx["session_length_so_far"] = pos

            events.append({
                "user_id": user_id,
                "session_id": str(row.get("sessionId", f"{user_id}_{date}")),
                "app_name": row["app_canonical"],
                "app": row["app_canonical"],  # alias for env compatibility
                "timestamp_unix": int(ts),
                "hour_of_day": int(row["hour"]),
                "day_of_week": int(row["day_of_week"]),
                "inter_arrival_s": float(inter),
                "session_position": pos,
                "context": ctx,
            })

        # Run feature engineer to enrich context
        self.fe.process_events(events)
        return events

    def _split_by_user(
        self, user_sessions: Dict[str, List[List[Dict]]]
    ) -> Tuple[List, List, List]:
        """
        Split 70/15/15 by user ID (not by time) to prevent data leakage.
        """
        rng = np.random.default_rng(self.seed)
        user_ids = list(user_sessions.keys())
        rng.shuffle(user_ids)

        n = len(user_ids)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        train_users = user_ids[:train_end]
        val_users   = user_ids[train_end:val_end]
        test_users  = user_ids[val_end:]

        def flatten(users):
            sessions = []
            for u in users:
                sessions.extend(user_sessions[u])
            return sessions

        train = flatten(train_users)
        val   = flatten(val_users)
        test  = flatten(test_users)

        logger.info(
            "Split: train=%d, val=%d, test=%d sessions",
            len(train), len(val), len(test),
        )
        return train, val, test
