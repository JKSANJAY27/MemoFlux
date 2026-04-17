"""dashboard/components/kpi_cards.py — Live KPI Metric Cards"""
import plotly.graph_objects as go
from typing import Dict, List, Optional


def make_kpi_chart(
    values: List[float],
    color: str = "#1428A0",
    title: str = "",
) -> go.Figure:
    """Mini sparkline chart for KPI trend."""
    fig = go.Figure(go.Scatter(
        y=values, mode="lines",
        line=dict(color=color, width=2, shape="spline"),
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in color else f"{color}26",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        title=dict(text=title, font=dict(color="#8B9DC3", size=10), x=0),
    )
    return fig


def make_kpi_comparison_bar(baseline_kpis: Dict, ax_kpis: Dict) -> go.Figure:
    """Side-by-side KPI comparison bar chart."""
    labels = [
        "Avg Load (ms)", "Cache Hit %", "Thrash/100",
        "Stability", "HR@3", "Mem Eff"
    ]
    b_vals = [
        baseline_kpis.get("avg_load_time_ms", 228),
        baseline_kpis.get("cache_hit_rate", 0.65) * 100,
        baseline_kpis.get("thrash_rate_per_100", 18),
        baseline_kpis.get("stability_rate", 0.994) * 100,
        baseline_kpis.get("next_app_hr3", 0) * 100,
        baseline_kpis.get("memory_efficiency", 0.42) * 100,
    ]
    a_vals = [
        ax_kpis.get("avg_load_time_ms", 182),
        ax_kpis.get("cache_hit_rate", 0.87) * 100,
        ax_kpis.get("thrash_rate_per_100", 8),
        ax_kpis.get("stability_rate", 1.0) * 100,
        ax_kpis.get("next_app_hr3", 0.76) * 100,
        ax_kpis.get("memory_efficiency", 0.72) * 100,
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="LRU Baseline", x=labels, y=b_vals,
        marker_color="#2D4FD4", marker_line_width=0,
        text=[f"{v:.1f}" for v in b_vals], textposition="auto",
        textfont=dict(color="white", size=10),
    ))
    fig.add_trace(go.Bar(
        name="AX Memory (Target)", x=labels, y=a_vals,
        marker_color="#00C3FF", marker_line_width=0,
        text=[f"{v:.1f}" for v in a_vals], textposition="auto",
        textfont=dict(color="white", size=10),
    ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F0F4FF", family="Inter"),
        legend=dict(
            orientation="h", x=0, y=1.12,
            font=dict(color="#8B9DC3"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(tickfont=dict(color="#8B9DC3", size=11), gridcolor="#1a2235"),
        yaxis=dict(tickfont=dict(color="#8B9DC3"), gridcolor="#1a2235"),
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def make_prediction_bar(apps: List[str], probs: List[float]) -> go.Figure:
    """Animated horizontal probability bar for next-app predictions."""
    colors = ["#00C3FF", "#1428A0", "#6C63FF", "#2D6AE0", "#87CEEB"]

    fig = go.Figure(go.Bar(
        x=probs,
        y=apps,
        orientation="h",
        marker_color=colors[:len(apps)],
        marker_line_width=0,
        text=[f"{p:.0%}" for p in probs],
        textposition="inside",
        textfont=dict(color="white", size=12, family="Inter Bold"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(tickfont=dict(color="#F0F4FF", size=13), gridcolor="#1a2235"),
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig
