"""dashboard/components/memory_map.py — Visual Memory Slot Grid"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Optional, Dict


SLOT_COLORS = {
    "active":   "#1428A0",
    "background": "#1a3a5c",
    "preloaded": "#4a2d9e",
    "evicted":   "#3d0a0a",
    "empty":    "#111827",
}

SLOT_TEXT_COLORS = {
    "active":    "#00C3FF",
    "background": "#87CEEB",
    "preloaded": "#D8B4FE",
    "evicted":   "#FF4757",
    "empty":    "#4A5568",
}


def render_memory_map(slots: List[Dict], n_cols: int = 5) -> go.Figure:
    """
    Render interactive memory slot grid as a Plotly figure.

    slots: list of dicts with keys: app, state (active/background/preloaded/evicted/empty), ram_mb
    """
    n = len(slots)
    n_rows = (n + n_cols - 1) // n_cols

    fig = go.Figure()

    for idx, slot in enumerate(slots):
        row = idx // n_cols
        col = idx % n_cols
        state = slot.get("state", "empty")
        app   = slot.get("app", "—")
        ram   = slot.get("ram_mb", 0)

        color  = SLOT_COLORS.get(state, SLOT_COLORS["empty"])
        tcolor = SLOT_TEXT_COLORS.get(state, SLOT_TEXT_COLORS["empty"])

        x0, x1 = col * 1.1, col * 1.1 + 1.0
        y0, y1 = -row * 1.3, -row * 1.3 + 1.1

        # Slot rectangle
        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=color,
            line=dict(
                color="#00C3FF" if state == "active" else
                      "#6C63FF" if state == "preloaded" else
                      "#FF4757" if state == "evicted" else
                      "#2D4FD4" if state == "background" else "#1e2a3a",
                width=2 if state != "empty" else 1,
            ),
        )

        # Slot label
        state_icon = {"active": "●", "background": "○", "preloaded": "◆", "evicted": "✕", "empty": "·"}
        label_text = f"{state_icon.get(state,'·')} Slot {idx}<br><b>{app[:10] if app != '—' else '—'}</b>"
        if ram > 0:
            label_text += f"<br><span style='font-size:10px;color:#8B9DC3'>{ram}MB</span>"

        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2,
            text=label_text,
            showarrow=False,
            font=dict(size=11, color=tcolor, family="Inter"),
            align="center",
        )

    # Legend
    for state_name, color in SLOT_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=state_name.capitalize(),
            showlegend=True,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-0.1, n_cols * 1.1]),
        yaxis=dict(visible=False, range=[-(n_rows) * 1.3, 1.2]),
        height=max(180, n_rows * 140),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", x=0, y=1.05, font=dict(color="#8B9DC3", size=10)),
        showlegend=True,
    )
    return fig


def render_ram_gauge(used_mb: int, total_mb: int) -> go.Figure:
    """Gauge chart showing RAM utilisation."""
    pct = used_mb / max(total_mb, 1) * 100
    color = "#00D68F" if pct < 60 else "#FFD700" if pct < 85 else "#FF4757"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"color": color, "size": 28, "family": "Inter"}},
        title={"text": f"RAM Used<br><span style='font-size:12px;color:#8B9DC3'>{used_mb}/{total_mb} MB</span>",
               "font": {"color": "#8B9DC3", "size": 13}},
        gauge={
            "axis":   {"range": [0, 100], "tickcolor": "#4A5568"},
            "bar":    {"color": color, "thickness": 0.3},
            "bgcolor": "#111827",
            "steps": [
                {"range": [0, 60],  "color": "#0a1a14"},
                {"range": [60, 85], "color": "#1a1a0a"},
                {"range": [85, 100],"color": "#1a0a0a"},
            ],
            "threshold": {"line": {"color": "#FF4757", "width": 3}, "value": 90},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(color="#F0F4FF"),
    )
    return fig
