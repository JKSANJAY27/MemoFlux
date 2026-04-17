"""
data_pipeline/session_builder.py — Episode Constructor

AX Memory | Samsung AX Hackathon 2026 | PS-03

Converts flat event logs (from any source — LSApp, Kaggle, or synthetic)
into structured Episode objects ready for MemorySimEnv.

Session boundary rule: 5-minute inactivity gap = new sub-session.
Episode filter: remove episodes with < 3 app switches.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

SESSION_GAP_SECONDS = 300  # 5 minutes → new sub-session
MIN_EVENTS = 3             # discard shorter episodes


@dataclass
class Episode:
    """One user-day of app switches, ready for MemorySimEnv."""
    user_id: str
    date: str
    events: List[Dict] = field(default_factory=list)
    archetype: str = "mixed"      # inferred usage pattern
    stats: Dict = field(default_factory=dict)

    def to_session_list(self) -> List[Dict]:
        """Return flat list of event dicts (env-compatible)."""
        return self.events

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def n_unique_apps(self) -> int:
        return len(set(e["app"] for e in self.events))


class SessionBuilder:
    """
    Builds Episode objects from raw event lists.

    Usage:
        builder = SessionBuilder()
        episodes = builder.build(raw_events)   # list of flat event dicts
        sessions = [ep.to_session_list() for ep in episodes]
    """

    # Archetype inference rules (based on hour distribution)
    _ARCHETYPE_RULES = {
        "morning_commuter": lambda h: (7 <= h <= 9),
        "night_owl":        lambda h: (h >= 22 or h <= 2),
        "work":             lambda h: (9 <= h <= 17),
        "social":           lambda h: (12 <= h <= 14 or 19 <= h <= 22),
    }

    def __init__(self, seed: int = 42):
        self.fe = FeatureEngineer(seed=seed)

    # ─────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────

    def build(self, events: List[Dict]) -> List[Episode]:
        """
        Convert a flat list of app-open events into Episode objects.

        Each event dict must contain at minimum:
          {"user_id", "app", "timestamp_unix", "hour_of_day", "day_of_week"}
        """
        # Group by (user_id, date)
        by_user_day: Dict[Tuple[str, str], List[Dict]] = {}
        for ev in events:
            uid = ev.get("user_id", "u0")
            ts  = ev.get("timestamp_unix", 0)
            date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            key = (uid, date)
            by_user_day.setdefault(key, []).append(ev)

        episodes: List[Episode] = []
        for (uid, date), day_events in by_user_day.items():
            day_events.sort(key=lambda e: e.get("timestamp_unix", 0))
            sub_sessions = self._split_sub_sessions(day_events)
            for sub in sub_sessions:
                if len(sub) < MIN_EVENTS:
                    continue
                sub = self._enrich_events(sub)
                ep = Episode(
                    user_id=uid,
                    date=date,
                    events=sub,
                    archetype=self._infer_archetype(sub),
                    stats=self._compute_stats(sub),
                )
                episodes.append(ep)

        logger.info("Built %d episodes from %d events", len(episodes), len(events))
        return episodes

    def build_from_sessions(self, sessions: List[List[Dict]]) -> List[Episode]:
        """Build episodes from already-grouped session lists."""
        episodes = []
        for sess in sessions:
            if len(sess) < MIN_EVENTS:
                continue
            uid  = sess[0].get("user_id", "u0")
            ts   = sess[0].get("timestamp_unix", 0)
            date = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            ep = Episode(
                user_id=uid,
                date=date,
                events=sess,
                archetype=self._infer_archetype(sess),
                stats=self._compute_stats(sess),
            )
            episodes.append(ep)
        return episodes

    # ─────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────

    def _split_sub_sessions(self, events: List[Dict]) -> List[List[Dict]]:
        """Split day events into sub-sessions on 5-min inactivity gaps."""
        if not events:
            return []
        sub_sessions: List[List[Dict]] = [[events[0]]]
        for ev in events[1:]:
            prev_ts = sub_sessions[-1][-1].get("timestamp_unix", 0)
            curr_ts = ev.get("timestamp_unix", 0)
            if curr_ts - prev_ts > SESSION_GAP_SECONDS:
                sub_sessions.append([ev])
            else:
                sub_sessions[-1].append(ev)
        return sub_sessions

    def _enrich_events(self, events: List[Dict]) -> List[Dict]:
        """Compute inter-arrival times and add context vectors."""
        prev_ts = None
        for pos, ev in enumerate(events):
            ts = ev.get("timestamp_unix", 0)
            inter = (ts - prev_ts) if prev_ts is not None else 0.0
            prev_ts = ts
            ev["inter_arrival_s"] = float(inter)
            ev["session_position"] = pos
            if "context" not in ev:
                ctx = self.fe.build_context_dict(
                    hour=ev.get("hour_of_day", 12),
                    day_of_week=ev.get("day_of_week", 0),
                    inter_arrival_s=inter,
                    session_app_count=pos,
                )
                ctx["is_morning"] = 6 <= ev.get("hour_of_day", 12) <= 11
                ctx["is_evening"] = 18 <= ev.get("hour_of_day", 12) <= 23
                ctx["session_length_so_far"] = pos
                ev["context"] = ctx
        return events

    def _infer_archetype(self, events: List[Dict]) -> str:
        """Infer user archetype from peak activity hours."""
        if not events:
            return "mixed"
        hours = [e.get("hour_of_day", 12) for e in events]
        # Count votes from archetype rules
        votes: Dict[str, int] = {arch: 0 for arch in self._ARCHETYPE_RULES}
        for h in hours:
            for arch, rule in self._ARCHETYPE_RULES.items():
                if rule(h):
                    votes[arch] += 1
        best = max(votes, key=votes.get)
        return best if votes[best] > 0 else "mixed"

    def _compute_stats(self, events: List[Dict]) -> Dict:
        if not events:
            return {}
        apps = [e.get("app", "UNKNOWN") for e in events]
        timestamps = [e.get("timestamp_unix", 0) for e in events]
        total_dur = (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0
        return {
            "n_events": len(events),
            "n_unique_apps": len(set(apps)),
            "total_duration_s": total_dur,
            "most_used_app": max(set(apps), key=apps.count),
        }
