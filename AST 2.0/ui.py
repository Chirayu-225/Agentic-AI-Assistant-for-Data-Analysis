"""
ui.py — Reusable Streamlit rendering helpers.

render_insight_box()  — styled ranked insight panel (green)
render_explanation()  — styled deep explanation panel (blue)
render_auto_chart()   — auto bar/line chart for tabular results
compute_confidence()  — confidence scoring for analysis results
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# INSIGHT BOX
# ─────────────────────────────────────────────────────────────────────────────

def render_insight_box(raw_insight: str, title: str = "// AI INSIGHTS") -> None:
    """Renders the styled insight panel with consistent font size."""
    st.markdown(f"""
    <div style="margin-top:8px; padding:22px 26px 18px; background:#11151F;
                border:1px solid #242B3D; border-top:2px solid #2DD4BF; border-radius:8px;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                    color:#2DD4BF; letter-spacing:0.28em; margin-bottom:18px;
                    opacity:0.9; text-transform:uppercase;">{title}</div>
    """, unsafe_allow_html=True)

    lines_raw = [l.strip() for l in raw_insight.splitlines() if l.strip()]
    for i, line in enumerate(lines_raw):
        # Strip markdown artifacts
        clean = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
        clean = re.sub(r'\*(.*?)\*',     r'\1', clean)
        clean = re.sub(r'`(.*?)`',       r'\1', clean)
        clean = clean.strip()
        if not clean:
            continue

        # Detect rank prefix: "1•", "2•", "1.", "2." etc.
        rank_match = re.match(r'^([1-9][•.:\-])\s*(.*)', clean)
        if rank_match:
            rank = rank_match.group(1).rstrip('.:- ')
            body = rank_match.group(2).strip()
        else:
            rank = "•"
            body = clean.lstrip('•–-● ').strip()

        # Rank 1 slightly more prominent — all others identical and large
        num_color   = "#2DD4BF"
        body_color  = "#F5F6F8" if i == 0 else "#DDE1E8"
        body_weight = "600"     if i == 0 else "500"

        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:14px;
                    padding:14px 0; border-bottom:1px solid #242B3D;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem;
                        color:{num_color}; font-weight:700; min-width:30px;
                        flex-shrink:0; padding-top:2px;">{rank}</div>
            <div style="font-family:'Inter',sans-serif; font-size:1.05rem;
                        color:{body_color}; font-weight:{body_weight};
                        line-height:1.6; letter-spacing:0.01em;">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEEP EXPLANATION PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_explanation(raw_explanation: str) -> None:
    """Renders the styled deep explanation panel (accent border, dark theme)."""
    st.markdown("""
    <div style="margin-top:8px; padding:22px 26px 18px; background:#11151F;
                border:1px solid #242B3D; border-top:2px solid #2DD4BF; border-radius:8px;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                    color:#2DD4BF; letter-spacing:0.28em; margin-bottom:18px;
                    opacity:0.9; text-transform:uppercase;">// DEEP EXPLANATION</div>
    """, unsafe_allow_html=True)

    exp_lines = [l.strip() for l in raw_explanation.splitlines() if l.strip()]
    for line in exp_lines:
        clean = re.sub(r'^#{1,4}\s*', '', line)
        clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean)
        clean = re.sub(r'\*(.*?)\*',     r'\1', clean)
        clean = re.sub(r'`(.*?)`',       r'\1', clean)
        clean = clean.strip()
        if not clean:
            continue

        is_numbered_heading = bool(re.match(r'^[1-4][.:].{2,}', clean))
        # Short colon-terminated intro lines with no numbering, e.g. "Insight Explained:"
        # — style as a small muted label instead of letting them blend into body text.
        is_label_line = (
            not is_numbered_heading
            and clean.endswith(':')
            and len(clean) <= 40
        )

        if is_numbered_heading:
            st.markdown(
                f"<div style='font-family:Inter,sans-serif; font-size:1.12rem;"
                f"color:#2DD4BF; font-weight:600; padding:14px 0 4px;"
                f"letter-spacing:0.01em; border-bottom:1px solid #242B3D;"
                f"margin-bottom:6px;'>{clean}</div>",
                unsafe_allow_html=True
            )
        elif is_label_line:
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace; font-size:0.78rem;"
                f"color:#7C8698; font-weight:500; letter-spacing:0.15em;"
                f"text-transform:uppercase; padding:0 0 10px;'>{clean}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='font-family:Inter,sans-serif; font-size:1.02rem;"
                f"color:#DDE1E8; font-weight:400; padding:4px 0;"
                f"line-height:1.65;'>{clean}</div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Shared dark theme applied to every Plotly figure
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FAFBFC",
    font=dict(family="Inter, sans-serif", color="#334155", size=12),
    xaxis=dict(
        gridcolor="#E2E8F0", linecolor="#E2E8F0",
        tickfont=dict(color="#64748B", size=10),
    ),
    yaxis=dict(
        gridcolor="#E2E8F0", linecolor="#E2E8F0",
        tickfont=dict(color="#64748B", size=10),
    ),
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(
        bgcolor="#FAFBFC", bordercolor="#2563EB",
        font=dict(family="JetBrains Mono", color="#2563EB"),
    ),
    legend=dict(
        bgcolor="#FAFBFC", bordercolor="#E2E8F0",
        font=dict(color="#334155"),
    ),
)

# Cyberpunk colour sequence for multi-series charts
_PLOTLY_COLOURS = [
    "#2563EB", "#2563EB", "#D97706", "#FF4488",
    "#AA44FF", "#44FFFF", "#FFFF44", "#FF8844",
]


def _apply_theme(fig):
    """Apply the dark cyberpunk theme to any Plotly figure."""
    fig.update_layout(**_PLOTLY_LAYOUT)
    fig.update_traces(marker_line_width=0)
    return fig


def _chart_label():
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;"
        "color:#94A3B8;margin-top:12px;letter-spacing:0.1em;'>AUTO CHART</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AUTO CHART  (Plotly-first, matplotlib fallback)
# ─────────────────────────────────────────────────────────────────────────────

def render_auto_chart(result) -> None:
    """
    Smart auto-chart using Plotly for interactivity.
    - DataFrame with 1 numeric col  → horizontal bar chart
    - DataFrame with 2+ numeric cols → grouped bar chart
    - Series, datetime index         → line + area chart
    - Series, categorical index      → bar chart
    Falls back to matplotlib if Plotly is not installed.
    All failures are silent — chart is always best-effort.
    """
    try:
        if not PLOTLY_AVAILABLE:
            _render_auto_chart_mpl(result)
            return

        if isinstance(result, pd.DataFrame):
            num_cols = result.select_dtypes(include="number").columns.tolist()
            if not num_cols or len(result) > 50:
                return

            df_plot = result.copy()
            df_plot.index = df_plot.index.astype(str)
            x_label = df_plot.index.name or "index"
            df_plot = df_plot.reset_index()
            df_plot.columns = [str(c) for c in df_plot.columns]
            x_col = df_plot.columns[0]

            if len(num_cols) == 1:
                # Single numeric — horizontal bar, sorted descending
                df_sorted = df_plot.sort_values(num_cols[0], ascending=True)
                fig = px.bar(
                    df_sorted, x=num_cols[0], y=x_col,
                    orientation="h",
                    color_discrete_sequence=_PLOTLY_COLOURS,
                    labels={num_cols[0]: num_cols[0], x_col: x_label},
                )
            else:
                # Multiple numeric — grouped vertical bar
                fig = px.bar(
                    df_plot, x=x_col, y=num_cols[:6],  # cap at 6 series
                    barmode="group",
                    color_discrete_sequence=_PLOTLY_COLOURS,
                    labels={x_col: x_label},
                )

            _apply_theme(fig)
            _chart_label()
            st.plotly_chart(fig, width='stretch')

        elif isinstance(result, pd.Series):
            if not pd.api.types.is_numeric_dtype(result) or len(result) > 50:
                return

            s = result.copy()

            if pd.api.types.is_datetime64_any_dtype(s.index):
                # Time series → line + area
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=s.index, y=s.values,
                    mode="lines",
                    line=dict(color="#2563EB", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(37,99,235,0.08)",
                    name=s.name or "value",
                    hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                ))
            else:
                # Categorical index → bar
                df_s = s.reset_index()
                df_s.columns = ["category", "value"]
                df_s = df_s.sort_values("value", ascending=True)
                fig = px.bar(
                    df_s, x="value", y="category",
                    orientation="h",
                    color_discrete_sequence=_PLOTLY_COLOURS,
                    labels={"value": s.name or "value", "category": ""},
                    hover_data={"value": ":.4f"},
                )

            _apply_theme(fig)
            _chart_label()
            st.plotly_chart(fig, width='stretch')

    except Exception:
        pass  # always silent


def _render_auto_chart_mpl(result) -> None:
    """Matplotlib fallback when Plotly is not installed."""
    try:
        if isinstance(result, pd.DataFrame):
            num_cols = result.select_dtypes(include="number").columns.tolist()
            if not num_cols or len(result) > 25:
                return
            fig, ax = plt.subplots(figsize=(8, 3.5))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FAFBFC")
            ax.bar(result.index.astype(str), result[num_cols[0]],
                   color="#2563EB", alpha=0.85)
            ax.set_ylabel(num_cols[0], color="#64748B", fontsize=9)
        elif isinstance(result, pd.Series):
            if not pd.api.types.is_numeric_dtype(result) or len(result) > 25:
                return
            fig, ax = plt.subplots(figsize=(8, 3.5))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FAFBFC")
            if pd.api.types.is_datetime64_any_dtype(result.index):
                ax.plot(result.index, result.values, color="#2563EB", linewidth=2)
                ax.fill_between(result.index, result.values, alpha=0.15, color="#2563EB")
            else:
                ax.bar(result.index.astype(str), result.values, color="#2563EB", alpha=0.85)
                plt.xticks(rotation=35, ha="right")
        else:
            return
        ax.tick_params(colors="#64748B", labelsize=8)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax.spines[sp].set_color("#E2E8F0")
        plt.tight_layout()
        _chart_label()
        st.pyplot(fig)
        plt.clf()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(result, success: bool, shortcut_used: bool) -> tuple[str, str]:
    """
    Score analysis confidence from execution factors.
    Returns (label, hex_colour).
    """
    factors = []

    if result is not None:
        factors.append("non_null_result")
    if success:
        factors.append("clean_execution")
    if shortcut_used:
        factors.append("shortcut_path")  # rule-based = highest confidence

    if isinstance(result, pd.DataFrame) and len(result) > 0:
        factors.append("non_empty_dataframe")
    elif isinstance(result, pd.Series) and len(result) > 0:
        factors.append("non_empty_series")
    elif isinstance(result, (int, float)) and result == result:  # NaN check
        factors.append("valid_scalar")

    n = len(factors)
    if n >= 4:
        return "HIGH",   "#2563EB"
    elif n >= 2:
        return "MEDIUM", "#D97706"
    else:
        return "LOW",    "#DC2626"
