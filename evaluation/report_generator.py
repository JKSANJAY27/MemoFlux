"""
evaluation/report_generator.py — JSON + Markdown Report Exporter

AX Memory | Samsung AX Hackathon 2026 | PS-03
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ReportGenerator:
    """Generate JSON and Markdown reports from benchmark results."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, results: Dict, filename: Optional[str] = None) -> str:
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = results.get("manager_name", "baseline").replace(" ", "_").lower()
            filename = f"{name}_{ts}.json"
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return str(path)

    def save_markdown(self, results: Dict, filename: Optional[str] = None) -> str:
        md = self._render_markdown(results)
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = results.get("manager_name", "baseline").replace(" ", "_").lower()
            filename = f"{name}_{ts}_report.md"
        path = self.output_dir / filename
        with open(path, "w") as f:
            f.write(md)
        return str(path)

    def save_combined_comparison(self, results_list: List[Dict]) -> str:
        """Generate a multi-baseline comparison markdown report."""
        lines = [
            "# AX Memory — Baseline Comparison Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Samsung AX Hackathon 2026 | PS-03",
            "",
            "## Summary Table",
            "",
            "| KPI | " + " | ".join(r["manager_name"] for r in results_list) + " |",
            "|-----|" + "|".join(["---"] * len(results_list)) + "|",
        ]
        kpis = ["avg_load_time_ms","avg_cold_start_ms","cache_hit_rate",
                "thrash_rate_per_100","stability_rate","next_app_hr3","memory_efficiency"]
        labels = ["Avg Load (ms)","Avg Cold Start (ms)","Cache Hit Rate","Thrash/100 Steps",
                  "Stability Rate","Next-App HR@3","Memory Efficiency"]
        for kk, label in zip(kpis, labels):
            vals = [f"{r['kpis'].get(kk, 0):.3f}" for r in results_list]
            lines.append(f"| {label} | " + " | ".join(vals) + " |")

        lines += ["", "## Week 2+3 Targets", ""]
        lines += [
            "| KPI | Baseline (Best) | AX Target | Δ Required |",
            "|-----|----------------|-----------|----------|",
            "| Cache Hit Rate | ~65% | ≥85% | +20pp |",
            "| Thrash / 100 steps | ~18 | ≤9 | ↓50% |",
            "| Avg Load Time | ~228 ms | ≤183 ms | ↓20% |",
            "| Next-App HR@3 | 0% | ≥75% | +75pp |",
            "| Memory Efficiency | ~42% | ≥72% | +30pp |",
        ]

        path = self.output_dir / "baseline_comparison_report.md"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return str(path)

    def _render_markdown(self, results: Dict) -> str:
        kpis = results.get("kpis", {})
        return f"""# AX Memory — KPI Report
**Manager**: {results.get('manager_name', '—')}
**Device**: {results.get('device_profile', '—')}
**Episodes**: {results.get('n_episodes', '—')}
**Timestamp**: {results.get('timestamp', '—')}

## KPI Results

| KPI | Value | Target |
|-----|-------|--------|
| Avg Load Time | {kpis.get('avg_load_time_ms', 0):.1f} ms | ≤182.7 ms |
| Avg Cold Start | {kpis.get('avg_cold_start_ms', 0):.1f} ms | — |
| Cache Hit Rate | {kpis.get('cache_hit_rate', 0):.1%} | ≥85% |
| Thrash / 100 Steps | {kpis.get('thrash_rate_per_100', 0):.1f} | ≤9.1 |
| Stability Rate | {kpis.get('stability_rate', 0):.1%} | 100% |
| Next-App HR@3 | {kpis.get('next_app_hr3', 0):.1%} | ≥75% |
| Memory Efficiency | {kpis.get('memory_efficiency', 0):.1%} | ↑30% |

*These baseline numbers will be beaten by the Week 2+3 AX ML system.*
"""
