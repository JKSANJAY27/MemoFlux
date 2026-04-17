"""dashboard/components/decision_log.py — Human-Readable Action Log"""
import streamlit as st
from typing import List, Dict


ACTION_ICONS  = {"preload": "◆", "evict": "✕", "keep": "●", "promote": "▲"}
ACTION_COLORS = {"preload": "#6C63FF", "evict": "#FF4757", "keep": "#00D68F", "promote": "#FFD700"}


def render_decision_log(log_entries: List[Dict], max_entries: int = 10):
    """Render a scrollable human-readable decision log."""
    entries = log_entries[-max_entries:][::-1]  # newest first

    if not entries:
        st.markdown(
            "<div style='color:#4A5568;font-size:0.8rem;font-style:italic;'>"
            "No actions yet — press 'Step' to start</div>",
            unsafe_allow_html=True,
        )
        return

    log_html = "<div style='max-height:280px;overflow-y:auto;'>"
    for entry in entries:
        action = entry.get("action_type", "keep")
        icon   = ACTION_ICONS.get(action, "·")
        color  = ACTION_COLORS.get(action, "#8B9DC3")
        app    = entry.get("app", "?")
        reason = entry.get("reason", "")
        step   = entry.get("step", "?")
        ts     = entry.get("timestamp", "")

        log_html += f"""
        <div style='
          background:#111827;
          border-left:3px solid {color};
          border-radius:0 8px 8px 0;
          padding:8px 12px;
          margin:4px 0;
          font-family:"JetBrains Mono",monospace;
          font-size:0.72rem;
        '>
          <span style='color:{color};font-weight:bold;'>{icon} [{ts}] STEP {step}</span>
          <span style='color:#F0F4FF;font-weight:600;'> {action.upper()} → {app}</span><br/>
          <span style='color:#8B9DC3;'>{reason[:100]}</span>
        </div>"""

    log_html += "</div>"
    st.markdown(log_html, unsafe_allow_html=True)
