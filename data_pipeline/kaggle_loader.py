"""
data_pipeline/kaggle_loader.py — Kaggle Mobile Device Usage Dataset Parser

AX Memory | Samsung AX Hackathon 2026 | PS-03

Supplementary dataset (700 user daily aggregates).
Used for:
  - User archetype labeling (light / medium / heavy user segmentation)
  - Context vector enrichment (app usage time, battery drain baselines)

Source: https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KAGGLE_CSV_NAME = "mobile_device_usage.csv"
KAGGLE_SEARCH_DIRS = [
    "data/raw",
    "data/raw/mobile-device-usage-and-user-behavior-dataset",
]


class KaggleLoader:
    """
    Loads the Kaggle mobile device usage dataset to derive user-level
    archetype labels that enrich LSApp sessions.

    Usage:
        loader = KaggleLoader()
        archetypes = loader.get_archetype_map()
        # Returns {"light": [...features], "heavy": [...features]}
    """

    USAGE_COLUMNS = [
        "App Usage Time (min/day)",
        "Screen On Time (hours/day)",
        "Battery Drain (mAh/day)",
        "Number of Apps Installed",
        "Data Usage (MB/day)",
        "Age",
        "User Behavior Class",
    ]

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or self._find_csv()
        self._df: Optional[pd.DataFrame] = None

    def _find_csv(self) -> Optional[str]:
        for d in KAGGLE_SEARCH_DIRS:
            for fname in [KAGGLE_CSV_NAME, "user_behavior_dataset.csv"]:
                p = os.path.join(d, fname)
                if os.path.exists(p):
                    return p
        return None

    def is_available(self) -> bool:
        return self.csv_path is not None and os.path.exists(self.csv_path)

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        if not self.is_available():
            logger.warning(
                "Kaggle dataset not found. Using default archetype distributions. "
                "Download from https://www.kaggle.com/datasets/valakhorasani/"
                "mobile-device-usage-and-user-behavior-dataset"
            )
            return pd.DataFrame()

        df = pd.read_csv(self.csv_path)
        self._df = df
        logger.info("Loaded Kaggle dataset: %d records, columns: %s", len(df), list(df.columns))
        return df

    def get_user_segments(self) -> Dict[str, Dict]:
        """
        Returns average feature profiles per user behavior class.
        Classes: 1=Light, 2=Moderate, 3=Regular, 4=Heavy, 5=Extreme
        """
        df = self.load()
        if df.empty:
            return self._default_segments()

        behavior_col = "User Behavior Class"
        if behavior_col not in df.columns:
            return self._default_segments()

        segments = {}
        for cls, group in df.groupby(behavior_col):
            label = {1: "light", 2: "moderate", 3: "regular", 4: "heavy", 5: "extreme"}.get(cls, str(cls))
            segments[label] = {
                "avg_usage_min": float(group.get("App Usage Time (min/day)", pd.Series([60])).mean()),
                "avg_screen_h":  float(group.get("Screen On Time (hours/day)", pd.Series([3])).mean()),
                "avg_apps":      float(group.get("Number of Apps Installed", pd.Series([40])).mean()),
                "avg_data_mb":   float(group.get("Data Usage (MB/day)", pd.Series([500])).mean()),
                "n_users": len(group),
            }
        return segments

    def _default_segments(self) -> Dict[str, Dict]:
        """Fallback segments derived from typical Android usage statistics."""
        return {
            "light":    {"avg_usage_min": 60,  "avg_screen_h": 2.0, "avg_apps": 25, "avg_data_mb": 200,  "n_users": 0},
            "moderate": {"avg_usage_min": 150, "avg_screen_h": 4.0, "avg_apps": 40, "avg_data_mb": 600,  "n_users": 0},
            "regular":  {"avg_usage_min": 300, "avg_screen_h": 6.0, "avg_apps": 55, "avg_data_mb": 1200, "n_users": 0},
            "heavy":    {"avg_usage_min": 500, "avg_screen_h": 8.5, "avg_apps": 75, "avg_data_mb": 2500, "n_users": 0},
            "extreme":  {"avg_usage_min": 700, "avg_screen_h": 12.0,"avg_apps": 100,"avg_data_mb": 4000, "n_users": 0},
        }

    def archetype_from_usage(self, daily_app_min: float) -> str:
        """Map daily usage time to a synthetic archetype label."""
        if daily_app_min < 100:
            return "mixed"           # light user → random behaviour
        elif daily_app_min < 250:
            return "morning_commuter"
        elif daily_app_min < 400:
            return "work"
        elif daily_app_min < 600:
            return "social"
        else:
            return "night_owl"
