"""
evaluation/benchmark_runner.py — Automated Benchmark Runner

AX Memory | Samsung AX Hackathon 2026 | PS-03

Runs N episodes with a memory manager and collects full KPI statistics.
Produces the "before" numbers the ML system will beat.

Usage:
  python -m evaluation.benchmark_runner
  python -m evaluation.benchmark_runner --manager lfu --episodes 200
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.memory_sim_env import MemorySimEnv, DEVICE_PROFILES
from baselines.lru_manager import LRUMemoryManager
from baselines.lfu_manager import LFUMemoryManager
from baselines.static_priority import StaticPriorityManager
from evaluation.kpi_tracker import KPITracker
from data_pipeline.synthetic_generator import SyntheticGenerator

console = Console()
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


class BenchmarkRunner:
    """
    Runs N episodes of the MemorySimEnv with a given memory manager
    and returns full KPI statistics.

    Usage:
        runner = BenchmarkRunner(env, manager, kpi_tracker)
        results = runner.run(n_episodes=500, verbose=True)
        runner.save_report("results/lru_500ep.json")
    """

    def __init__(
        self,
        env: MemorySimEnv,
        manager,
        kpi_tracker: KPITracker,
    ):
        self.env = env
        self.manager = manager
        self.kpi_tracker = kpi_tracker
        self._results: Optional[Dict] = None

    def run(self, n_episodes: int = 500, verbose: bool = True) -> Dict:
        """Run benchmark and return results dict."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.kpi_tracker.reset()
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.fields[cache_rate]:.0%} hit"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"[blue]{self.manager.manager_name}",
                total=n_episodes,
                cache_rate=0.0,
            )

            for ep in range(n_episodes):
                obs, _ = self.env.reset()
                self.manager.reset() if hasattr(self.manager, "reset") else None
                self.kpi_tracker.start_episode()
                crashed = False
                ep_cache_hits = 0
                ep_steps = 0

                try:
                    terminated = truncated = False
                    while not (terminated or truncated):
                        action = self.manager.act(obs)
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        self.kpi_tracker.record_step(info)
                        if info.get("cache_hit"):
                            ep_cache_hits += 1
                        ep_steps += 1

                except Exception as e:
                    crashed = True
                    logger.warning("Episode %d crashed: %s", ep, e)

                self.kpi_tracker.end_episode(crashed=crashed)
                rolling_hit = ep_cache_hits / max(ep_steps, 1)
                progress.advance(task, 1)
                progress.update(task, cache_rate=rolling_hit)

        elapsed = time.time() - start_time
        kpis = self.kpi_tracker.compute_all_kpis()

        self._results = {
            "manager_name": self.manager.manager_name,
            "n_episodes": n_episodes,
            "timestamp": datetime.now().isoformat(),
            "device_profile": self.env.device_profile,
            "elapsed_s": round(elapsed, 2),
            "kpis": kpis,
            "per_episode_stats": self.kpi_tracker.compute_per_episode_stats(),
            "convergence_plot_data": self._compute_convergence(
                self.kpi_tracker.compute_per_episode_stats()
            ),
        }

        if verbose:
            self._print_report(kpis, elapsed)

        return self._results

    def save_report(self, path: Optional[str] = None) -> str:
        if self._results is None:
            raise RuntimeError("Call run() before save_report()")
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = self.manager.manager_name.replace(" ", "_").lower()
            path = str(RESULTS_DIR / f"{safe_name}_{ts}.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._results, f, indent=2, default=str)
        console.print(f"[green]✓[/green] Report saved → [cyan]{path}[/cyan]")
        return path

    # ─────────────────────────────────────────────────────
    # Pretty display
    # ─────────────────────────────────────────────────────

    def _print_report(self, kpis: Dict, elapsed: float):
        console.print()
        console.print(Panel.fit(
            f"[bold blue]AX Memory — Baseline KPI Report[/bold blue]\n"
            f"Manager: [yellow]{self.manager.manager_name}[/yellow]   "
            f"Device: [cyan]{self.env.device_profile}[/cyan]   "
            f"Time: {elapsed:.1f}s",
            border_style="blue",
        ))

        t = Table(show_header=True, header_style="bold magenta")
        t.add_column("KPI", style="cyan", width=30)
        t.add_column("Value", justify="right")
        t.add_column("Target", justify="right")
        t.add_column("Status")

        rows = [
            ("Avg Load Time",       f"{kpis['avg_load_time_ms']:.1f} ms",  "≤182.7 ms (↓20%)",  "⏳ baseline"),
            ("Avg Cold Start",      f"{kpis['avg_cold_start_ms']:.1f} ms", "—",                  "⏳ baseline"),
            ("Cache Hit Rate",      f"{kpis['cache_hit_rate']:.1%}",       "≥85%",               _status(kpis['cache_hit_rate'], 0.85, higher=True)),
            ("Thrash / 100 steps",  f"{kpis['thrash_rate_per_100']:.1f}",  "≤9.1 (↓50%)",        "⏳ baseline"),
            ("Stability Rate",      f"{kpis['stability_rate']:.1%}",       "100%",               _status(kpis['stability_rate'], 1.0, higher=True)),
            ("Next-App HR@3",       f"{kpis['next_app_hr3']:.1%}",         "≥75% (Week 2)",      "[dim]N/A for baselines[/dim]"),
            ("Memory Efficiency",   f"{kpis['memory_efficiency']:.1%}",    "↑30% from baseline", "⏳ baseline"),
        ]
        for name, val, target, status in rows:
            t.add_row(name, val, target, status)

        console.print(t)
        console.print()
        console.print(
            "[dim italic]These are your BASELINE numbers. "
            "Week 2+3 will beat every single one.[/dim italic]"
        )
        console.print()

    def _compute_convergence(self, ep_stats: List[Dict]) -> List[Dict]:
        """Rolling 50-episode window for convergence curve."""
        window = 50
        data = []
        for i, ep in enumerate(ep_stats):
            recent = ep_stats[max(0, i - window):i + 1]
            data.append({
                "episode": i,
                "rolling_cache_hit": np.mean([e["cache_hit_rate"] for e in recent]),
                "rolling_load_time": np.mean([e["avg_load_time_ms"] for e in recent]),
                "thrash": ep["thrash_events"],
            })
        return data


def _status(val: float, target: float, higher: bool = True) -> str:
    ok = (val >= target) if higher else (val <= target)
    return "[green]✓ Pass[/green]" if ok else "[red]✗ Fail[/red]"


# ─────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────

def _load_sessions(n_synthetic: int = 500) -> List:
    """Try LSApp first, fall back to synthetic."""
    try:
        from data_pipeline.lsapp_loader import LSAppLoader
        loader = LSAppLoader()
        train, val, _ = loader.load_splits()
        if len(train) > 0:
            console.print(f"[green]✓[/green] Using LSApp: {len(train)} train sessions")
            return train
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] LSApp unavailable ({e}), using synthetic data")

    gen = SyntheticGenerator(n_users=n_synthetic // 30, days_per_user=30, seed=42)
    sessions = gen.generate_all()
    console.print(f"[green]✓[/green] Generated {len(sessions)} synthetic sessions")
    return sessions


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="AX Memory Benchmark Runner")
    parser.add_argument("--manager",  choices=["lru", "lfu", "static", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--device",   default="galaxy_s24", choices=list(DEVICE_PROFILES.keys()))
    parser.add_argument("--synthetic", type=int, default=500,
                        help="Number of synthetic users if LSApp unavailable")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold blue]AX Memory — Benchmark Runner[/bold blue]\n"
        "[dim]Samsung AX Hackathon 2026 | PS-03[/dim]",
        border_style="blue",
    ))

    sessions = _load_sessions(args.synthetic)
    assert sessions, "No sessions available"

    managers = {
        "lru":    ("LRU Baseline",            LRUMemoryManager(n_slots=DEVICE_PROFILES[args.device]["n_slots"])),
        "lfu":    ("LFU Baseline",            LFUMemoryManager(n_slots=DEVICE_PROFILES[args.device]["n_slots"])),
        "static": ("Static Priority Baseline", StaticPriorityManager(n_slots=DEVICE_PROFILES[args.device]["n_slots"])),
    }

    selected = list(managers.keys()) if args.manager == "all" else [args.manager]
    all_results = {}

    for mgr_key in selected:
        _, mgr = managers[mgr_key]
        env = MemorySimEnv(sessions=sessions, device_profile=args.device)
        tracker = KPITracker()
        runner = BenchmarkRunner(env, mgr, tracker)
        console.print(f"\n[bold]Running {mgr.manager_name} — {args.episodes} episodes on {args.device}[/bold]")
        results = runner.run(n_episodes=args.episodes)
        report_path = runner.save_report()
        all_results[mgr_key] = results

    # Combined comparison
    if len(selected) > 1:
        console.print("\n[bold magenta]=== BASELINE COMPARISON ===[/bold magenta]")
        t = Table(header_style="bold")
        t.add_column("KPI")
        for k in selected:
            t.add_column(managers[k][0], justify="right")

        kpi_keys = ["avg_load_time_ms","cache_hit_rate","thrash_rate_per_100","stability_rate","memory_efficiency"]
        labels   = ["Avg Load (ms)","Cache Hit","Thrash/100","Stability","Mem Eff"]
        for kk, label in zip(kpi_keys, labels):
            vals = [str(all_results[k]["kpis"].get(kk, "—")) for k in selected]
            t.add_row(label, *vals)
        console.print(t)


if __name__ == "__main__":
    main()
