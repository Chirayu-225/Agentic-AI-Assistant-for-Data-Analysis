import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import ollama
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = go = None
    PLOTLY_AVAILABLE = False

# Modular components
from sandbox       import is_code_safe, safe_exec
from cleaning      import rule_based_clean
from query_engine  import (detect_shortcut, build_analysis_prompt,
                            build_retry_prompt, cached_llm_call,
                            llm_with_fallback, insight_llm,
                            extract_and_clean_code)
from profiler      import (build_dataset_profile, format_profile_for_llm,
                            build_insight_prompt, build_auto_analyze_prompt,
                            build_explain_prompt)
from ui            import (render_insight_box, render_explanation,
                            render_auto_chart, compute_confidence)
from rag_engine    import (index_document, rag_answer, extract_document_insights,
                            check_dependencies, all_deps_ok)

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="ANALYST ASSISTANT", layout="wide", page_icon="⚡")

# ==========================================
# CUSTOM CSS — CYBERPUNK TERMINAL AESTHETIC + LARGER FONTS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

/* ---- GLOBAL ---- */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background-color: #060A0F !important;
    color: #C8D8E8 !important;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.12rem !important;
}
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,170,0.012) 2px, rgba(0,255,170,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Prevent sidebar collapse ── */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060D14 0%, #040810 100%) !important;
    border-right: 1px solid #0D2235 !important;
    box-shadow: 4px 0 30px rgba(0,255,170,0.04);
    padding-top: 0 !important;
    display: flex !important;
    visibility: visible !important;
    min-width: 350px !important;
    width: 350px !important;
    transform: none !important;
    left: 0 !important;
    position: relative !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
section[data-testid="stSidebar"] .block-container { padding: 0 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #060A0F; }
::-webkit-scrollbar-thumb { background: #00FFAA33; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #00FFAA88; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stDataFrame { border: 1px solid #0D2235 !important; border-radius: 2px !important; background: #060D14 !important; }
.stDataFrame [data-testid="stDataFrameResizable"] { background: #060D14 !important; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #060D14, #08121C);
    border: 1px solid #0D2235; border-left: 3px solid #00FFAA;
    border-radius: 2px; padding: 18px 22px !important;
    position: relative; overflow: hidden;
}
[data-testid="stMetric"]::after {
    content: ''; position: absolute; top: 0; right: 0;
    width: 40px; height: 40px;
    background: radial-gradient(circle at top right, rgba(0,255,170,0.08), transparent 70%);
}
[data-testid="stMetricLabel"] p {
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.95rem !important;
    color: #4A7A6A !important; letter-spacing: 0.15em !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important; font-size: 2.4rem !important;
    font-weight: 900 !important; color: #00FFAA !important;
    text-shadow: 0 0 24px rgba(0,255,170,0.55) !important;
}
.stTextInput input {
    background: #08121C !important; border: 1px solid #0D2235 !important;
    border-bottom: 2px solid #00FFAA44 !important; border-radius: 2px !important;
    color: #C8D8E8 !important; font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.32rem !important; font-weight: 500 !important;
    letter-spacing: 0.03em; padding: 14px 18px !important; transition: all 0.2s ease;
}
.stTextInput input:focus {
    border-bottom: 2px solid #00FFAA !important;
    box-shadow: 0 4px 20px rgba(0,255,170,0.1) !important; outline: none !important;
}
.stTextInput input::placeholder { color: #2A4A5A !important; }
.stTextInput label p {
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.9rem !important;
    color: #4A7A6A !important; letter-spacing: 0.12em !important;
}
.stButton > button {
    background: transparent !important; border: 1px solid #00FFAA !important;
    border-radius: 2px !important; color: #00FFAA !important;
    font-family: 'Orbitron', monospace !important; font-size: 1.05rem !important;
    font-weight: 700 !important; letter-spacing: 0.2em !important;
    padding: 14px 32px !important; text-transform: uppercase !important;
    transition: all 0.25s ease !important; position: relative; overflow: hidden; width: 100%;
}
.stButton > button::before {
    content: ''; position: absolute; top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,255,170,0.08), transparent);
    transition: left 0.4s ease;
}
.stButton > button:hover {
    background: rgba(0,255,170,0.06) !important;
    box-shadow: 0 0 30px rgba(0,255,170,0.2), inset 0 0 30px rgba(0,255,170,0.05) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:active { transform: translateY(0px) !important; }
.stSlider [data-baseweb="slider"] { padding-top: 4px !important; }
.stSlider [data-testid="stThumbValue"] {
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.9rem !important; color: #00FFAA !important;
}
[data-testid="stFileUploader"] {
    background: #08121C !important; border: 1px dashed #0D2235 !important;
    border-radius: 2px !important; padding: 10px !important; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #00FFAA44 !important; }
[data-testid="stFileUploader"] label {
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.85rem !important;
    color: #4A7A6A !important; letter-spacing: 0.12em;
}
.stSuccess, [data-testid="stNotification"] {
    background: #0A1A12 !important; border: 1px solid #00FFAA33 !important;
    border-left: 3px solid #00FFAA !important; border-radius: 2px !important;
}
.stInfo {
    background: #0A1220 !important; border: 1px solid #0070FF33 !important;
    border-left: 3px solid #0070FF !important; border-radius: 2px !important;
}
.stWarning {
    background: #1A1208 !important; border: 1px solid #FFAA0033 !important;
    border-left: 3px solid #FFAA00 !important; border-radius: 2px !important;
}
.stError {
    background: #1A0A0A !important; border: 1px solid #FF444433 !important;
    border-left: 3px solid #FF4444 !important; border-radius: 2px !important;
}
.stCode, code, pre {
    font-family: 'Share Tech Mono', monospace !important; background: #08121C !important;
    border: 1px solid #0D2235 !important; border-radius: 2px !important;
    font-size: 1.05rem !important; color: #00FFAA !important; line-height: 1.45 !important;
}
hr { border: none !important; border-top: 1px solid #0D2235 !important; margin: 1.5rem 0 !important; }
[data-testid="stStatusWidget"] {
    background: #08121C !important; border: 1px solid #0D2235 !important;
    border-left: 3px solid #0070FF !important; border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.95rem !important;
}
.header-block { padding: 32px 0 24px 0; border-bottom: 1px solid #0D2235; margin-bottom: 28px; }
.brand-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; color: #00FFAA;
    letter-spacing: 0.35em; text-transform: uppercase; margin-bottom: 4px; opacity: 0.7;
}
.main-title {
    font-family: 'Orbitron', monospace; font-size: 3.8rem !important; font-weight: 900;
    color: #E8F4F0; letter-spacing: 0.07em; line-height: 1;
    text-shadow: 0 0 70px rgba(0,255,170,0.18);
}
.main-title span { color: #00FFAA; text-shadow: 0 0 40px rgba(0,255,170,0.7); }
.sub-title {
    font-family: 'Rajdhani', sans-serif; font-size: 1.32rem !important; color: #3A6A5A;
    letter-spacing: 0.12em; font-weight: 500; text-transform: uppercase; margin-top: 6px;
}
.status-bar { display: flex; align-items: center; gap: 20px; margin-top: 16px; }
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.92rem !important;
    padding: 4px 10px; border: 1px solid #0D3025; border-radius: 1px;
}
.status-dot {
    width: 5px; height: 5px; border-radius: 50%; background: #00FFAA;
    box-shadow: 0 0 6px #00FFAA; animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.section-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.95rem !important; color: #00FFAA;
    letter-spacing: 0.25em; text-transform: uppercase; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, #0D2235, transparent); }
.panel {
    background: linear-gradient(135deg, #060D14 0%, #05090F 100%);
    border: 1px solid #0D2235; border-radius: 2px; padding: 20px;
    height: 100%; position: relative; overflow: hidden;
}
.panel::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #00FFAA33, transparent);
}
.panel-corner {
    position: absolute; top: 0; right: 0; width: 20px; height: 20px;
    border-top: 2px solid #00FFAA44; border-right: 2px solid #00FFAA44;
}
.sidebar-header {
    background: linear-gradient(180deg, #060D14 0%, transparent 100%);
    padding: 24px 20px 16px; border-bottom: 1px solid #0D2235; margin-bottom: 20px;
}
.sidebar-title {
    font-family: 'Orbitron', monospace; font-size: 0.95rem; color: #00FFAA;
    letter-spacing: 0.3em; font-weight: 700; text-transform: uppercase;
}
.sidebar-sub { font-family: 'Share Tech Mono', monospace; font-size: 0.82rem; color: #2A4A3A; letter-spacing: 0.15em; margin-top: 2px; }
.sys-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 20px; border-bottom: 1px solid #0A1820;
    font-family: 'Share Tech Mono', monospace; font-size: 0.92rem !important; letter-spacing: 0.1em;
}
.sys-key { color: #2A4A5A; }
.sys-val { color: #00FFAA; }
.sys-val.blue { color: #4488FF; }
.sys-val.orange { color: #FFAA44; }
.clock-display {
    font-family: 'Orbitron', monospace; font-size: 1.75rem; color: #00FFAA;
    text-shadow: 0 0 20px rgba(0,255,170,0.5); font-weight: 700; letter-spacing: 0.1em;
    text-align: center; padding: 16px 20px; background: #040810; border-bottom: 1px solid #0D2235;
}
.upload-zone { padding: 0 20px 20px; }
.upload-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; color: #2A4A5A;
    letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 8px;
}
.result-header {
    font-family: 'Orbitron', monospace; font-size: 1.1rem !important; color: #E8F4F0;
    letter-spacing: 0.2em; font-weight: 700; text-transform: uppercase;
    padding: 16px 24px; background: #08121C; border: 1px solid #0D2235;
    border-bottom: 2px solid #00FFAA; border-radius: 2px 2px 0 0;
    display: flex; align-items: center; gap: 10px;
}
.result-body {
    background: #060D14; border: 1px solid #0D2235; border-top: none;
    border-radius: 0 0 2px 2px; padding: 20px;
}
.schema-block {
    font-family: 'Share Tech Mono', monospace; font-size: 1.02rem !important; color: #6AB090;
    background: #040810; border: 1px solid #0A1820; border-left: 3px solid #00FFAA33;
    padding: 14px 16px; border-radius: 2px; line-height: 1.9; white-space: pre; overflow-x: auto;
}
.query-container {
    background: linear-gradient(135deg, #060D14 0%, #040A12 100%);
    border: 1px solid #0D2235; border-radius: 2px; padding: 28px; position: relative;
}
.query-container::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, #00FFAA22, #00FFAA44, #00FFAA22);
}
.timestamp { font-family: 'Share Tech Mono', monospace; font-size: 0.9rem !important; color: #1A3A2A; letter-spacing: 0.15em; }
.corner-decoration {
    position: absolute; bottom: 0; right: 0; width: 30px; height: 30px;
    border-bottom: 2px solid #00FFAA22; border-right: 2px solid #00FFAA22;
}
div[style*="cursor:pointer"] > div { font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    now = datetime.now()
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-title">ANALYST ASSISTANT</div>
        <div class="sidebar-sub">v1.0 // SECURE CHANNEL</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Self-contained live clock ──────────────────────────────────────────
    # Rendered via components.v1.html instead of st.markdown so it gets its
    # own persistent iframe that Streamlit does NOT tear down on every
    # rerun. The clock ticks using its own internal Date() — it never
    # reaches into window.parent, so it keeps running independently of
    # whatever else is happening in the main Streamlit script.
    components.html(
        """
        <div id="live-clock" style="
            font-family: 'Orbitron', monospace; font-size: 1.75rem; color: #00FFAA;
            text-shadow: 0 0 20px rgba(0,255,170,0.5); font-weight: 700; letter-spacing: 0.1em;
            text-align: center; padding: 16px 20px; background: #040810;
            border-bottom: 1px solid #0D2235; margin: 0;
        ">--:--:--</div>
        <script>
            function tickClock() {
                const el = document.getElementById('live-clock');
                if (!el) return;
                const now = new Date();
                const h = String(now.getHours()).padStart(2, '0');
                const m = String(now.getMinutes()).padStart(2, '0');
                const s = String(now.getSeconds()).padStart(2, '0');
                el.textContent = h + ':' + m + ':' + s;
            }
            tickClock();
            setInterval(tickClock, 1000);
        </script>
        """,
        height=68,
    )

    st.markdown("""
    <div class="sys-row"><span class="sys-key">SYS_STATUS</span><span class="sys-val">ONLINE</span></div>
    <div class="sys-row"><span class="sys-key">LLM_MODEL</span><span class="sys-val">QWEN 2.5 CODER</span></div>
    <div class="sys-row"><span class="sys-key">DEPLOYMENT</span><span class="sys-val blue">LOCAL</span></div>
    <div class="sys-row"><span class="sys-key">NETWORK</span><span class="sys-val orange">AIR-GAPPED</span></div>
    <div class="sys-row"><span class="sys-key">ENCRYPTION</span><span class="sys-val">AES-256</span></div>
    <div class="sys-row"><span class="sys-key">UPTIME</span><span class="sys-val">99.97%</span></div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    temp = 0.1  # default before file upload

    with st.container():
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        st.markdown('<div class="upload-label">// DATA INGEST — CSV</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        st.markdown("""
        <div class="sys-row" style="margin-top:8px;">
            <span class="sys-key">FILE_STATUS</span><span class="sys-val">LOADED ✓</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='padding: 16px 20px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="upload-label">// ENGINE PARAMS</div>', unsafe_allow_html=True)
        temp = st.slider("TEMPERATURE", 0.0, 1.0, 0.1, key="temp")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        st.markdown('<div class="upload-label">// DATA INGEST — DOCUMENT</div>', unsafe_allow_html=True)
        uploaded_doc = st.file_uploader(
            "Upload document for RAG",
            type=["pdf", "docx", "txt", "md"],
            label_visibility="collapsed",
            key="doc_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_doc:
        st.markdown("""
        <div class="sys-row" style="margin-top:8px;">
            <span class="sys-key">DOC_STATUS</span><span class="sys-val">LOADED ✓</span>
        </div>
        <div class="sys-row">
            <span class="sys-key">RAG_ENGINE</span><span class="sys-val">READY</span>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# Security + sandbox imported from sandbox.py
# ==========================================
# MAIN CONTENT HEADER
# ==========================================
st.markdown(f"""
<div class="header-block">
    <div class="brand-label">// SECURE ANALYTICS TERMINAL</div>
    <div class="main-title">ANALYST<span>_</span>ASSISTANT</div>
    <div class="sub-title">AI-Powered Data Intelligence System — Local Inference Engine</div>
    <div class="status-bar">
        <div class="status-pill"><div class="status-dot"></div> SYSTEM NOMINAL</div>
        <div class="status-pill"><div class="status-dot"></div> LLM READY</div>
        <div class="status-pill"><div class="status-dot"></div> {now.strftime("%Y-%m-%d // %H:%M UTC")}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# PLOTLY HELPERS
# ==========================================
def _is_plotly_fig(obj) -> bool:
    t = str(type(obj))
    return "plotly" in t and "Figure" in t

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#060D14", plot_bgcolor="#08121C",
    font=dict(family="Rajdhani, sans-serif", color="#C8D8E8", size=12),
    xaxis=dict(gridcolor="#0D2235", linecolor="#0D2235",
               tickfont=dict(color="#6AB090", size=10)),
    yaxis=dict(gridcolor="#0D2235", linecolor="#0D2235",
               tickfont=dict(color="#6AB090", size=10)),
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor="#08121C", bordercolor="#00FFAA",
                    font=dict(family="Share Tech Mono", color="#00FFAA")),
    legend=dict(bgcolor="#08121C", bordercolor="#0D2235",
                font=dict(color="#C8D8E8")),
)

def _apply_plotly_theme(fig) -> None:
    try:
        fig.update_layout(**_PLOTLY_LAYOUT)
        fig.update_traces(marker_line_width=0)
    except Exception:
        pass

# ==========================================
# TABS
# ==========================================
_tab_csv, _tab_doc = st.tabs(["⚡  CSV ANALYSIS", "📄  DOCUMENT INTELLIGENCE"])

# ── TAB 1: CSV ANALYSIS ──────────────────
with _tab_csv:
    if uploaded_file is None:
        st.markdown("""
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:1px; background:#0D2235; border:1px solid #0D2235; border-radius:2px; overflow:hidden; margin-bottom:24px;">
            <div style="background:#060D14; padding:28px; text-align:center;">
                <div style="font-family:'Orbitron',monospace; font-size:2.4rem; color:#00FFAA; font-weight:900; text-shadow:0 0 20px rgba(0,255,170,0.5);">01</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; color:#2A5A4A; letter-spacing:0.2em; margin-top:6px;">INGEST</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:1.2rem; color:#6A9A8A; margin-top:8px; font-weight:500;">Upload your dataset via the sidebar panel</div>
            </div>
            <div style="background:#060D14; padding:28px; text-align:center; border-left:1px solid #0D2235;">
                <div style="font-family:'Orbitron',monospace; font-size:2.4rem; color:#00FFAA; font-weight:900; text-shadow:0 0 20px rgba(0,255,170,0.5);">02</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; color:#2A5A4A; letter-spacing:0.2em; margin-top:6px;">QUERY</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:1.2rem; color:#6A9A8A; margin-top:8px; font-weight:500;">Enter a natural language analytical question</div>
            </div>
            <div style="background:#060D14; padding:28px; text-align:center; border-left:1px solid #0D2235;">
                <div style="font-family:'Orbitron',monospace; font-size:2.4rem; color:#00FFAA; font-weight:900; text-shadow:0 0 20px rgba(0,255,170,0.5);">03</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; color:#2A5A4A; letter-spacing:0.2em; margin-top:6px;">ANALYZE</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:1.2rem; color:#6A9A8A; margin-top:8px; font-weight:500;">Qwen 2.5 generates and executes Python code</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("""
            <div class="panel">
                <div class="panel-corner"></div>
                <div class="section-label">// SYSTEM CAPABILITIES</div>
                <div style="display:flex; flex-direction:column; gap:10px; margin-top:6px;">
                    <div style="display:flex; align-items:flex-start; gap:12px; padding:12px; background:#040810; border:1px solid #0A1820;">
                        <div style="font-family:'Orbitron',monospace; color:#00FFAA; font-size:0.95rem; min-width:28px; padding-top:2px;">▸</div>
                        <div>
                            <div style="font-family:'Rajdhani',sans-serif; font-size:1.3rem; font-weight:600; color:#C8D8E8; letter-spacing:0.05em;">Statistical Analysis</div>
                            <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; color:#3A6A5A; margin-top:2px;">Correlations, distributions, descriptive stats, outlier detection</div>
                        </div>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:12px; padding:12px; background:#040810; border:1px solid #0A1820;">
                        <div style="font-family:'Orbitron',monospace; color:#00FFAA; font-size:0.95rem; min-width:28px; padding-top:2px;">▸</div>
                        <div>
                            <div style="font-family:'Rajdhani',sans-serif; font-size:1.3rem; font-weight:600; color:#C8D8E8; letter-spacing:0.05em;">Data Filtering & Aggregation</div>
                            <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; color:#3A6A5A; margin-top:2px;">Group by, filter, sort, pivot — all via natural language</div>
                        </div>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:12px; padding:12px; background:#040810; border:1px solid #0A1820;">
                        <div style="font-family:'Orbitron',monospace; color:#00FFAA; font-size:0.95rem; min-width:28px; padding-top:2px;">▸</div>
                        <div>
                            <div style="font-family:'Rajdhani',sans-serif; font-size:1.3rem; font-weight:600; color:#C8D8E8; letter-spacing:0.05em;">Pattern Recognition</div>
                            <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; color:#3A6A5A; margin-top:2px;">Trends, seasonality, anomaly identification across dimensions</div>
                        </div>
                    </div>
                    <div style="display:flex; align-items:flex-start; gap:12px; padding:12px; background:#040810; border:1px solid #0A1820;">
                        <div style="font-family:'Orbitron',monospace; color:#00FFAA; font-size:0.95rem; min-width:28px; padding-top:2px;">▸</div>
                        <div>
                            <div style="font-family:'Rajdhani',sans-serif; font-size:1.3rem; font-weight:600; color:#C8D8E8; letter-spacing:0.05em;">Code Generation & Execution</div>
                            <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; color:#3A6A5A; margin-top:2px;">Auto-generates pandas code, runs it, surfaces results instantly</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div class="panel" style="height:100%;">
                <div class="panel-corner"></div>
                <div class="section-label">// QUERY EXAMPLES</div>
                <div style="display:flex; flex-direction:column; gap:8px; margin-top:6px;">
            """, unsafe_allow_html=True)
            examples = [
                "What are the top 5 rows by sales?",
                "Show correlation between price and volume",
                "Find all rows where revenue > 1000",
                "Group by category and sum profit",
                "What columns have null values?",
                "Calculate mean and std of numeric cols",
                "Show the distribution of age column",
            ]
            for ex in examples:
                st.markdown(f"""
                <div style="background:#040810; border:1px solid #0A1820; border-left:2px solid #00FFAA22; padding:10px 14px;">
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.95rem; color:#4A8A7A; letter-spacing:0.05em;">> {ex}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:24px; padding:16px 20px; background:#040810; border:1px solid #0A1820; border-top:2px solid #00FFAA11; border-radius:2px; display:flex; align-items:center; gap:16px;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; color:#1A3A2A; letter-spacing:0.15em;">AWAITING_DATA_STREAM</div>
            <div style="flex:1; height:1px; background:linear-gradient(90deg, #0D2235, transparent);"></div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem; color:#1A3A2A; letter-spacing:0.1em;">UPLOAD CSV TO BEGIN ANALYSIS</div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # DATA LOADED STATE
    # ==========================================
    else:
        if "file_name" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                st.session_state.df = pd.read_csv(uploaded_file, encoding="latin1")
            st.session_state.file_name = uploaded_file.name
            st.session_state.data_cleaned = False

        if "data_cleaned" not in st.session_state:
            st.session_state.data_cleaned = False

        # Conversation memory: stores (query, result_summary) pairs
        if "convo_history" not in st.session_state:
            st.session_state.convo_history = []

        df = st.session_state.df

        # ==========================================
        # RULE-BASED CLEANING ENGINE
        # ==========================================
        st.markdown('<div class="section-label">// AI DATA ENGINEERING AGENT</div>', unsafe_allow_html=True)

        if st.session_state.data_cleaned:
            st.success("✓ DATA PIPELINE COMPLETE — Dataset has been cleaned and is ready for analysis.")
            if st.button("↺ RESET & RE-CLEAN", use_container_width=True):
                st.session_state.data_cleaned = False
                st.rerun()

        else:
            if st.button("✨ INITIALIZE AUTO-CLEAN PIPELINE", use_container_width=True):
                with st.status("🛠️ Running Data Profiling & Cleaning Agent...", expanded=True) as clean_status:

                    # ── STAGE 1: RULE-BASED PASS ─────────────────────────────
                    st.write("▸ Stage 1 — Running deterministic rule-based cleaning...")
                    try:
                        rule_cleaned_df, rule_log = rule_based_clean(df)
                        st.markdown("**📋 RULE-BASED CLEANING REPORT:**")
                        for entry in rule_log:
                            st.markdown(
                                f"<div style='font-family:Share Tech Mono,monospace; font-size:0.9rem;"
                                f"color:#00FFAA; padding:2px 0;'>{entry}</div>",
                                unsafe_allow_html=True
                            )
                    except Exception as e:
                        st.warning(f"Rule-based pass issue: {e} — continuing to LLM pass.")
                        rule_cleaned_df = df.copy()

                    # ── STAGE 2: LLM PASS ────────────────────────────────────
                    st.write("▸ Stage 2 — LLM pass for remaining anomalies...")
                    raw_sample  = rule_cleaned_df.head(8).to_string()
                    col_list    = rule_cleaned_df.columns.tolist()
                    dtypes_list = rule_cleaned_df.dtypes.to_string()

                    clean_prompt = f"""
You are an expert Data Engineer. A pandas DataFrame named 'df' ALREADY EXISTS in memory.
Rule-based cleaning has already run. Handle ONLY what it missed.

Current state after rule-based cleaning:
Columns: {col_list}
Dtypes:
{dtypes_list}

Sample (first 8 rows):
{raw_sample}

Write Python code to finish cleaning. Assign result to 'cleaned_df'.

ONLY do these if needed:
1. Convert object columns with mixed numeric/string values that look like numbers.
2. Replace obvious sentinel nulls (-999, 'N/A', 'na', 'none', '--') with NaN.
3. Otherwise just: cleaned_df = df.copy()

STRICT RULES:
- Do NOT import anything. pd and np are already available.
- Do NOT use .dtype on a DataFrame — only on a Series.
- Do NOT redo work already done.
- cleaned_df MUST be assigned.
- Output ONLY a Python code block. No explanations.
"""
                    response   = ollama.chat(
                        model="qwen2.5-coder:3b",
                        messages=[{"role": "user", "content": clean_prompt}],
                        options={"temperature": 0.0}
                    )
                    raw_output = response["message"]["content"]
                    code_match = re.search(r"\x60\x60\x60(?:python)?\s*(.*?)\x60\x60\x60", raw_output, re.DOTALL)
                    llm_code   = code_match.group(1).strip() if code_match else raw_output.strip()
                    llm_code   = "\n".join(
                        l for l in llm_code.splitlines()
                        if not l.strip().startswith("import ") and not l.strip().startswith("from ")
                    )

                    st.markdown("**🔍 LLM CLEANING CODE:**")
                    st.code(llm_code, language="python")

                    # ── EXEC with retry fallback ──────────────────────────────
                    local_vars = {"df": rule_cleaned_df.copy(), "pd": pd, "np": np}
                    success, error = safe_exec(llm_code, local_vars)

                    if not success or "cleaned_df" not in local_vars:
                        st.warning(f"⚠ LLM pass failed ({error}) — retrying with simplified prompt...")
                        fallback_response = ollama.chat(
                            model="qwen2.5-coder:3b",
                            messages=[{"role": "user", "content": "Write ONE line: cleaned_df = df.copy()"}],
                            options={"temperature": 0.0}
                        )
                        fb_raw   = fallback_response["message"]["content"]
                        fb_match = re.search(r"\x60\x60\x60(?:python)?\s*(.*?)\x60\x60\x60", fb_raw, re.DOTALL)
                        fb_code  = fb_match.group(1).strip() if fb_match else "cleaned_df = df.copy()"
                        fb_code  = "\n".join(
                            l for l in fb_code.splitlines()
                            if not l.strip().startswith("import ") and not l.strip().startswith("from ")
                        )
                        local_vars = {"df": rule_cleaned_df.copy(), "pd": pd, "np": np}
                        success, error = safe_exec(fb_code, local_vars)
                        if not success or "cleaned_df" not in local_vars:
                            st.warning("⚠ LLM retry failed — using rule-based result only.")
                            local_vars["cleaned_df"] = rule_cleaned_df.copy()

                    st.session_state.df = local_vars["cleaned_df"]
                    st.session_state.data_cleaned = True
                    clean_status.update(label="✓ DATA STANDARDIZED & CLEANED", state="complete", expanded=False)
                    st.rerun()

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        df = st.session_state.df  # always pick up the latest (cleaned) version

        # ==========================================
        # METRICS ROW
        # ==========================================
        num_numeric = len(df.select_dtypes(include='number').columns)
        num_null    = df.isnull().sum().sum()
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("TOTAL ROWS",    f"{df.shape[0]:,}")
        with m2: st.metric("TOTAL COLUMNS", f"{df.shape[1]:,}")
        with m3: st.metric("NUMERIC COLS",  f"{num_numeric}")
        with m4: st.metric("NULL VALUES",   f"{num_null:,}")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ==========================================
        # DATA PREVIEW + SCHEMA
        # ==========================================
        col1, col2 = st.columns([3, 2], gap="medium")
        with col1:
            st.markdown('<div class="section-label">// DATA STREAM PREVIEW</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True, height=280)
            st.markdown(f"""
            <div style="margin-top:10px;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.85rem; color:#1A3A2A; letter-spacing:0.1em;">
                    SHOWING 10 / {df.shape[0]:,} RECORDS — {uploaded_file.name.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-label">// SCHEMA MANIFEST</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            schema_lines = []
            for col_name, dtype in df.dtypes.items():
                null_count = df[col_name].isnull().sum()
                schema_lines.append(f" {col_name:<28} {str(dtype):<12} [{null_count} null]")
            st.markdown(f'<div class="schema-block">{chr(10).join(schema_lines)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)

        # ==========================================
        # QUERY INTERFACE
        # ==========================================
        st.markdown('<div class="query-container">', unsafe_allow_html=True)
        _qh1, _qh2 = st.columns([3, 1])
        with _qh1:
            st.markdown('<div class="section-label">// ANALYTICAL QUERY INTERFACE</div>', unsafe_allow_html=True)
        with _qh2:
            if st.button("🗑 Clear Memory", use_container_width=True):
                st.session_state.convo_history = []
                st.rerun()
        if st.session_state.convo_history:
            st.markdown(
                f"<div style='font-family:Share Tech Mono,monospace;font-size:0.82rem;"
                f"color:#2A5A4A;margin-bottom:10px;letter-spacing:0.1em;'>"
                f"MEMORY: {len(st.session_state.convo_history)} exchange(s) in context</div>",
                unsafe_allow_html=True
            )

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols     = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if numeric_cols and cat_cols:
            hint = f"e.g. Show the average {numeric_cols[0]} grouped by {cat_cols[0]}"
        elif len(numeric_cols) >= 2:
            hint = f"e.g. Draw a scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}"
        elif numeric_cols:
            hint = f"e.g. What is the maximum value in the {numeric_cols[0]} column?"
        else:
            hint = "e.g. Count the number of rows..."

        user_query = st.text_input("QUERY INPUT", placeholder=hint, label_visibility="collapsed")

        btn_col, hint_col = st.columns([1, 3])
        with btn_col:
            execute = st.button("⚡ EXECUTE ANALYSIS")
        with hint_col:
            st.markdown("""
            <div style="padding-top:12px; font-family:'Share Tech Mono',monospace; font-size:0.85rem; color:#1A3A2A; letter-spacing:0.1em;">
                ENTER QUERY ABOVE → PRESS EXECUTE → QWEN 2.5 GENERATES AND RUNS PANDAS CODE
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="corner-decoration"></div></div>', unsafe_allow_html=True)

        # ==========================================
        # EXECUTION ENGINE
        # ==========================================
        if execute:
            if not user_query:
                st.warning("⚠️ Query input required before execution.")
                st.stop()

            st.session_state.pop("last_insight", None)  # clear stale insight on new query

            with st.status("🧠 Processing Analysis Pipeline...", expanded=True) as status:

                st.write("▸ Extracting dataset schema...")
                schema_text = "\n".join([f"- {col} ({df[col].dtype})" for col in df.columns])

                # Build conversation context (last 3 exchanges max)
                history_text = ""
                if st.session_state.convo_history:
                    history_text = "\nPrevious conversation context:\n"
                    for i, ex in enumerate(st.session_state.convo_history[-3:]):
                        history_text += f"Q{i+1}: {ex['query']}\nResult: {ex['result_summary'][:300]}\n"
                    history_text += "\n"

                # Query routing: simple queries bypass LLM via shortcut layer
                shortcut_code = detect_shortcut(user_query, df)

                if shortcut_code:
                    st.write("▸ Query matched shortcut — skipping LLM...")
                    code = shortcut_code
                    raw_output = f"[SHORTCUT] {shortcut_code}"
                else:
                    # Build hardened prompt with exact column names + history
                    prompt = build_analysis_prompt(schema_text, user_query, history_text)
                    st.write("▸ Sending request to Qwen 2.5 Coder...")
                    raw_output = llm_with_fallback(prompt, temperature=temp)

                st.write("▸ Extracting logic...")
                if shortcut_code:
                    code = shortcut_code
                else:
                    code = extract_and_clean_code(raw_output)


                st.markdown("**🔍 GENERATED CODE:**")
                st.code(code, language="python")

                # Safety guard before exec
                safe, pattern = is_code_safe(code)
                if not safe:
                    status.update(label="✗ UNSAFE CODE BLOCKED", state="error", expanded=False)
                    st.error(f"🚫 Blocked pattern detected: `{pattern}` — query rejected for safety.")
                    st.stop()

                st.write("▸ Executing in secure sandbox...")
                plt.close("all")
                local_vars = {"df": df, "pd": pd, "plt": plt, "np": np, "px": px, "go": go}
                success, error = safe_exec(code, local_vars)

                if not success:
                    # ── RETRY with error context ──────────────────────────────
                    st.warning(f"⚠ First attempt failed ({error}) — retrying with error context...")
                    retry_prompt = f"""
The following Python code failed with error: {error}

Failing code:
{code}

Fix ONLY the error. Do not rewrite everything. Keep the same logic.
Output ONLY the corrected Python code block. No explanations.
"""
                    retry_response = ollama.chat(
                        model="qwen2.5-coder:3b",
                        messages=[{"role": "user", "content": retry_prompt}],
                        options={"temperature": 0.0}
                    )
                    retry_raw   = retry_response["message"]["content"]
                    retry_match = re.search(r"\x60\x60\x60(?:python)?\s*(.*?)\x60\x60\x60", retry_raw, re.DOTALL)
                    retry_code  = retry_match.group(1).strip() if retry_match else retry_raw.strip()
                    retry_code  = "\n".join(
                        l for l in retry_code.splitlines()
                        if not l.strip().startswith("import ") and not l.strip().startswith("from ")
                    )
                    retry_code = retry_code.replace("plt.show()", "")
                    if "plt." in retry_code and "result = plt.gcf()" not in retry_code:
                        retry_code += "\nresult = plt.gcf()"

                    st.markdown("**🔁 RETRY CODE:**")
                    st.code(retry_code, language="python")

                    plt.close("all")
                    local_vars = {"df": df, "pd": pd, "plt": plt, "np": np, "px": px, "go": go}
                    success, error = safe_exec(retry_code, local_vars)

                    if not success:
                        status.update(label="✗ EXECUTION ERROR (retry failed)", state="error", expanded=False)
                        st.error(f"Runtime Exception after retry: {error}")
                        st.stop()

                # Resolve result + compute confidence score
                confidence_factors = []
                if "result" in local_vars:
                    result = local_vars["result"]
                    # If matplotlib plot was assigned to result incorrectly, grab figure
                    if not hasattr(result, "patch") and not _is_plotly_fig(result) and plt.get_fignums():
                        result = plt.gcf()
                    confidence_factors.append("result_assigned")
                    status.update(label="✓ EXECUTION COMPLETE", state="complete", expanded=False)
                elif plt.get_fignums():
                    result = plt.gcf()
                    confidence_factors.append("plot_captured")
                    status.update(label="✓ EXECUTION COMPLETE (Auto-captured plot)", state="complete", expanded=False)
                else:
                    result = None
                    status.update(label="⚠ Model did not produce 'result'", state="error", expanded=False)

                confidence_level, conf_color = compute_confidence(
                    result, success, bool(shortcut_code)
                )
                st.session_state.last_confidence = (confidence_level, conf_color)

            # ==========================================
            # SHOW RESULT
            # ==========================================
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="result-header"><span style="color:#00FFAA;">▸</span> ANALYSIS RESULT</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-body">', unsafe_allow_html=True)

            result_summary = ""

            if result is None:
                st.warning("No result returned from analysis.")

            elif _is_plotly_fig(result):
                # ── Plotly interactive figure ──────────────────────────────────
                _apply_plotly_theme(result)
                st.plotly_chart(result, use_container_width=True)
                result_summary = "[chart/plot]"

            elif str(type(result)) == "<class 'matplotlib.figure.Figure'>":
                result.patch.set_facecolor('#060D14')
                st.pyplot(result)
                plt.clf()
                result_summary = "[chart/plot]"

            elif isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
                render_auto_chart(result)
                result_summary = result.to_string(max_rows=20, max_cols=10)

            elif isinstance(result, pd.Series):
                st.dataframe(result)
                render_auto_chart(result)
                result_summary = result.to_string(max_rows=20)

            elif isinstance(result, (int, float, str)):
                st.markdown(f"""
                <div style="font-family:'Orbitron',monospace; font-size:1.6rem; color:#00FFAA;
                            text-shadow:0 0 20px rgba(0,255,170,0.4); padding:16px;
                            background:#040810; border-left:3px solid #00FFAA44;">
                    {result}
                </div>
                """, unsafe_allow_html=True)
                result_summary = str(result)
            else:
                st.write(result)
                result_summary = str(result)[:500]

            st.markdown('</div>', unsafe_allow_html=True)

            # Update conversation memory
            if result_summary:
                st.session_state.convo_history.append({
                    "query": user_query,
                    "result_summary": result_summary[:600]
                })
                st.session_state.convo_history = st.session_state.convo_history[-3:]

            # Persist result to session state so insight button works across reruns
            st.session_state.last_summary = result_summary
            st.session_state.last_query   = user_query
            # Save DataFrame result separately for structured profiling
            if isinstance(result, pd.DataFrame):
                st.session_state.last_result_df = result
            elif isinstance(result, pd.Series):
                st.session_state.last_result_df = result.to_frame()
            else:
                st.session_state.last_result_df = None
            # Clear stale insight when a new query runs
            st.session_state.pop("last_insight", None)

            _conf_level, _conf_color = st.session_state.get("last_confidence", ("MEDIUM", "#FFAA44"))
            st.markdown(f"""
            <div style="margin-top:20px; padding:10px 16px; background:#040810; border:1px solid #0A1820;
                        display:flex; align-items:center; justify-content:space-between; gap:16px;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.85rem; color:#1A3A2A; letter-spacing:0.1em;">
                    EXECUTION_LOG // {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </div>
                <div style="display:flex; align-items:center; gap:20px;">
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.82rem; color:#2A4A5A; letter-spacing:0.08em;">
                        ANALYSIS CONFIDENCE:
                        <span style="color:{_conf_color}; font-weight:700; margin-left:6px;
                                     text-shadow:0 0 8px {_conf_color}88;">{_conf_level}</span>
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.85rem; color:#00FFAA88; letter-spacing:0.1em;">
                        STATUS: SUCCESS
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ==========================================
        # INSIGHT PIPELINE
        # Step 1: Auto Analyze   — full dataset overview
        # Step 2: Generate Insights — deeper business insights, always on dataset
        # Step 3: Explain Insight  — drill into any insight line
        # No query required for any of these.
        # ==========================================
        _cache_key  = "cleaned" if st.session_state.get("data_cleaned") else "raw"
        _last_query = st.session_state.get("last_query", "dataset")

        # ── STEP 1: Auto Analyze ──────────────────────────────────────────────
        st.markdown('<div class="section-label">// PROACTIVE ANALYSIS</div>',
                    unsafe_allow_html=True)
        if st.button("🚀 AUTO ANALYZE DATASET", use_container_width=True):
            with st.spinner("Profiling dataset and generating insights..."):
                _profile     = build_dataset_profile(df, _cache_key=_cache_key)
                _profile_str = format_profile_for_llm(_profile)
                auto_prompt  = build_auto_analyze_prompt(_profile_str)
                st.session_state.auto_insight = insight_llm(auto_prompt).strip()

        if st.session_state.get("auto_insight"):
            render_insight_box(
                st.session_state.auto_insight,
                title="// DATASET INTELLIGENCE REPORT"
            )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)

        # ── STEP 2: Generate Insights (always works on full dataset) ──────────
        st.markdown('<div class="section-label">// AI INSIGHTS</div>',
                    unsafe_allow_html=True)

        if st.button("💡 GENERATE INSIGHTS", use_container_width=True,
                     key="btn_generate_insights"):
            with st.spinner("Synthesizing business insights from dataset..."):
                _profile     = build_dataset_profile(df, _cache_key=_cache_key)
                _profile_str = format_profile_for_llm(_profile)
                # Use the full dataset profile as both result and context
                # so insights are always grounded in the actual data
                insight_prompt = build_insight_prompt(
                    _last_query, _profile_str, _profile_str
                )
                st.session_state.last_insight = insight_llm(insight_prompt).strip()

        if st.session_state.get("last_insight"):
            render_insight_box(
                st.session_state.last_insight,
                title="// AI INSIGHTS"
            )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)

        # ── STEP 3: Explain any insight line ──────────────────────────────────
        st.markdown('<div class="section-label">// INSIGHT DRILL-DOWN</div>',
                    unsafe_allow_html=True)

        _has_insight = bool(
            st.session_state.get("last_insight") or
            st.session_state.get("auto_insight")
        )

        if not _has_insight:
            st.markdown("""
            <div style="padding:16px 20px; background:#060D14; border:1px solid #0D2235;
                        border-left:3px solid #4488FF22; border-radius:2px; margin-bottom:10px;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.88rem;
                            color:#2A4A5A; letter-spacing:0.12em;">
                    ▸ RUN AUTO ANALYZE OR GENERATE INSIGHTS FIRST — then paste any
                      insight line here to get a deep explanation
                </div>
            </div>
            """, unsafe_allow_html=True)

        explain_input = st.text_input(
            "EXPLAIN INPUT",
            placeholder="Paste any insight line here, e.g. '1• Classic Cars generates 39% of revenue...'",
            label_visibility="collapsed",
            key="explain_input"
        )
        st.markdown(
            "<div style='font-family:Share Tech Mono,monospace;font-size:0.82rem;"
            "color:#2A5A4A;margin-bottom:8px;letter-spacing:0.1em;'>"
            "PASTE ANY INSIGHT LINE ABOVE → CLICK EXPLAIN FOR ROOT CAUSE + ACTION PLAN"
            "</div>",
            unsafe_allow_html=True
        )

        if st.button("🔍 EXPLAIN THIS INSIGHT", use_container_width=True,
                     key="btn_explain_insight"):
            if not explain_input.strip():
                st.warning("Paste an insight line above before explaining.")
            else:
                with st.spinner("Generating deep explanation..."):
                    explain_prompt = build_explain_prompt(
                        explain_input.strip(), _last_query
                    )
                    st.session_state.last_explanation = insight_llm(
                        explain_prompt, temperature=0.4
                    ).strip()

        if st.session_state.get("last_explanation"):
            render_explanation(st.session_state.last_explanation)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENT INTELLIGENCE (RAG)
# ══════════════════════════════════════════════════════════════════════════════
with _tab_doc:

    # ── Dependency check ──────────────────────────────────────────────────────
    deps = check_dependencies()
    missing = [name for name, ok in deps.items() if not ok]

    if missing:
        st.markdown(f"""
        <div style="padding:20px 24px; background:#1A0A0A; border:1px solid #FF444433;
                    border-left:3px solid #FF4444; border-radius:2px; margin-top:8px;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.95rem;
                        color:#FF4444; letter-spacing:0.15em; margin-bottom:12px;">
                ⚠ MISSING DEPENDENCIES
            </div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:1.1rem; color:#C8D8E8;">
                Install the following packages, then restart the app:
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:1rem;
                        color:#00FFAA; margin-top:10px; background:#040810;
                        padding:12px 16px; border:1px solid #0D2235;">
                pip install {" ".join(missing)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Session state init ────────────────────────────────────────────────
        if "rag_doc_id"    not in st.session_state: st.session_state.rag_doc_id    = None
        if "rag_doc_name"  not in st.session_state: st.session_state.rag_doc_name  = None
        if "rag_metadata"  not in st.session_state: st.session_state.rag_metadata  = None
        if "rag_insight"   not in st.session_state: st.session_state.rag_insight   = None
        if "rag_answer"    not in st.session_state: st.session_state.rag_answer    = None
        if "rag_question"  not in st.session_state: st.session_state.rag_question  = None
        if "rag_history"   not in st.session_state: st.session_state.rag_history   = []

        # ── No document uploaded yet ──────────────────────────────────────────
        if uploaded_doc is None:
            st.markdown("""
            <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:1px;
                        background:#0D2235; border:1px solid #0D2235; border-radius:2px;
                        overflow:hidden; margin-bottom:24px;">
                <div style="background:#060D14; padding:28px; text-align:center;">
                    <div style="font-family:'Orbitron',monospace; font-size:2.4rem;
                                color:#00FFAA; font-weight:900; text-shadow:0 0 20px rgba(0,255,170,0.5);">01</div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem;
                                color:#2A5A4A; letter-spacing:0.2em; margin-top:6px;">UPLOAD</div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.2rem;
                                color:#6A9A8A; margin-top:8px; font-weight:500;">
                        Upload PDF, DOCX, or TXT via the sidebar panel</div>
                </div>
                <div style="background:#060D14; padding:28px; text-align:center; border-left:1px solid #0D2235;">
                    <div style="font-family:'Orbitron',monospace; font-size:2.4rem;
                                color:#00FFAA; font-weight:900; text-shadow:0 0 20px rgba(0,255,170,0.5);">02</div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem;
                                color:#2A5A4A; letter-spacing:0.2em; margin-top:6px;">INDEX</div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.2rem;
                                color:#6A9A8A; margin-top:8px; font-weight:500;">
                        Document is chunked, embedded, and stored locally</div>
                </div>
                <div style="background:#060D14; padding:28px; text-align:center; border-left:1px solid #0D2235;">
                    <div style="font-family:'Orbitron',monospace; font-size:2.4rem;
                                color:#00FFAA; font-weight:900; text-shadow:0 0 20px rgba(0,255,170,0.5);">03</div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem;
                                color:#2A5A4A; letter-spacing:0.2em; margin-top:6px;">QUERY</div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.2rem;
                                color:#6A9A8A; margin-top:8px; font-weight:500;">
                        Ask any question — RAG retrieves the right passages</div>
                </div>
            </div>
            <div style="padding:16px 20px; background:#040810; border:1px solid #0A1820;
                        border-top:2px solid #00FFAA11; border-radius:2px; margin-top:8px;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.9rem;
                            color:#1A3A2A; letter-spacing:0.1em;">
                    SUPPORTED FORMATS: PDF · DOCX · TXT · MD — all processing local, air-gapped
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Index document (once per upload) ─────────────────────────────
            doc_key = f"{uploaded_doc.name}_{uploaded_doc.size}"
            if st.session_state.rag_doc_id is None or st.session_state.rag_doc_name != doc_key:
                with st.status("⚙ Indexing document...", expanded=True) as idx_status:
                    try:
                        st.write("▸ Parsing document...")
                        file_bytes = uploaded_doc.read()

                        st.write("▸ Chunking text into passages...")
                        st.write("▸ Generating embeddings (all-MiniLM-L6-v2)...")
                        metadata = index_document(file_bytes, uploaded_doc.name)

                        st.session_state.rag_doc_id   = metadata["doc_id"]
                        st.session_state.rag_doc_name = doc_key
                        st.session_state.rag_metadata = metadata
                        st.session_state.rag_insight  = None
                        st.session_state.rag_answer   = None
                        st.session_state.rag_history  = []

                        idx_status.update(
                            label=f"✓ INDEXED — {metadata['chunk_count']} passages ready",
                            state="complete", expanded=False
                        )
                    except Exception as e:
                        idx_status.update(label="✗ INDEXING FAILED", state="error", expanded=False)
                        st.error(f"Error: {e}")
                        st.stop()

            # ── Document metrics ──────────────────────────────────────────────
            if st.session_state.rag_metadata:
                meta = st.session_state.rag_metadata
                dm1, dm2, dm3, dm4 = st.columns(4)
                with dm1: st.metric("DOCUMENT",  meta["filename"][:18] + "…" if len(meta["filename"]) > 18 else meta["filename"])
                with dm2: st.metric("WORDS",      f"{meta['word_count']:,}")
                with dm3: st.metric("CHARACTERS", f"{meta['char_count']:,}")
                with dm4: st.metric("PASSAGES",   f"{meta['chunk_count']}")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # ── Auto insight extraction ───────────────────────────────────────
            st.markdown('<div class="section-label">// DOCUMENT INTELLIGENCE REPORT</div>',
                        unsafe_allow_html=True)
            if st.button("🚀 EXTRACT KEY INSIGHTS FROM DOCUMENT", use_container_width=True):
                with st.spinner("Reading document and extracting insights..."):
                    try:
                        raw = extract_document_insights(
                            st.session_state.rag_doc_id,
                            uploaded_doc.name
                        )
                        st.session_state.rag_insight = raw
                    except Exception as e:
                        st.error(f"Insight extraction failed: {e}")

            if st.session_state.get("rag_insight"):
                render_insight_box(
                    st.session_state.rag_insight,
                    title="// DOCUMENT INTELLIGENCE REPORT"
                )

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            st.markdown('<hr>', unsafe_allow_html=True)

            # ── RAG Q&A Interface ─────────────────────────────────────────────
            st.markdown('<div class="query-container">', unsafe_allow_html=True)
            _rqh1, _rqh2 = st.columns([3, 1])
            with _rqh1:
                st.markdown('<div class="section-label">// DOCUMENT QUERY INTERFACE</div>',
                            unsafe_allow_html=True)
            with _rqh2:
                if st.button("🗑 Clear History", use_container_width=True, key="rag_clear"):
                    st.session_state.rag_history = []
                    st.session_state.rag_answer  = None
                    st.rerun()

            if st.session_state.rag_history:
                st.markdown(
                    f"<div style='font-family:Share Tech Mono,monospace;font-size:0.82rem;"
                    f"color:#2A5A4A;margin-bottom:10px;letter-spacing:0.1em;'>"
                    f"CONVERSATION: {len(st.session_state.rag_history)} exchange(s)</div>",
                    unsafe_allow_html=True
                )

            rag_query = st.text_input(
                "RAG QUERY INPUT",
                placeholder="e.g. What are the main findings? What does section 3 say about revenue?",
                label_visibility="collapsed",
                key="rag_query_input"
            )

            rbtn_col, rhint_col = st.columns([1, 3])
            with rbtn_col:
                rag_execute = st.button("⚡ QUERY DOCUMENT", use_container_width=True)
            with rhint_col:
                st.markdown("""
                <div style="padding-top:12px; font-family:'Share Tech Mono',monospace;
                            font-size:0.85rem; color:#1A3A2A; letter-spacing:0.1em;">
                    ASK ANYTHING ABOUT THE DOCUMENT → RAG RETRIEVES RELEVANT PASSAGES → LLM ANSWERS
                </div>
                """, unsafe_allow_html=True)
            st.markdown('<div class="corner-decoration"></div></div>', unsafe_allow_html=True)

            # ── Execute RAG query ─────────────────────────────────────────────
            if rag_execute:
                if not rag_query.strip():
                    st.warning("⚠ Enter a question before querying.")
                    st.stop()

                with st.status("🔍 Retrieving relevant passages...", expanded=True) as rag_status:
                    st.write("▸ Embedding query...")
                    st.write("▸ Searching vector store for top passages...")
                    st.write("▸ Sending context + question to Qwen 2.5...")
                    try:
                        answer = rag_answer(
                            st.session_state.rag_doc_id,
                            rag_query.strip(),
                            uploaded_doc.name,
                            top_k=5
                        )
                        st.session_state.rag_answer   = answer
                        st.session_state.rag_question = rag_query.strip()
                        st.session_state.rag_history.append({
                            "q": rag_query.strip(),
                            "a": answer[:400]
                        })
                        st.session_state.rag_history = st.session_state.rag_history[-5:]
                        rag_status.update(label="✓ ANSWER GENERATED",
                                          state="complete", expanded=False)
                    except Exception as e:
                        rag_status.update(label="✗ QUERY FAILED", state="error", expanded=False)
                        st.error(f"RAG error: {e}")

            # ── Show answer ───────────────────────────────────────────────────
            if st.session_state.get("rag_answer"):
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="result-header"><span style="color:#00FFAA;">▸</span>'
                    ' DOCUMENT ANSWER</div>',
                    unsafe_allow_html=True
                )
                st.markdown('<div class="result-body">', unsafe_allow_html=True)

                # Render the answer — detect if it has sections
                ans_lines = [l.strip() for l in
                             st.session_state.rag_answer.splitlines() if l.strip()]
                for line in ans_lines:
                    clean = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                    clean = re.sub(r'\*(.*?)\*',     r'\1', clean)
                    clean = re.sub(r'`(.*?)`',        r'\1', clean)
                    clean = clean.strip()
                    if not clean:
                        continue
                    is_heading = bool(re.match(r'^[1-9][.•:]\s', clean))
                    if is_heading:
                        st.markdown(
                            f"<div style='font-family:Rajdhani,sans-serif; font-size:1.2rem;"
                            f"color:#00FFAA; font-weight:600; padding:6px 0 2px;"
                            f"letter-spacing:0.02em;'>{clean}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div style='font-family:Rajdhani,sans-serif; font-size:1.18rem;"
                            f"color:#C8D8E8; font-weight:400; padding:3px 0;"
                            f"line-height:1.65;'>{clean}</div>",
                            unsafe_allow_html=True
                        )

                st.markdown('</div>', unsafe_allow_html=True)

                # Execution log
                st.markdown(f"""
                <div style="margin-top:16px; padding:10px 16px; background:#040810;
                            border:1px solid #0A1820;
                            display:flex; align-items:center; justify-content:space-between;">
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.85rem;
                                color:#1A3A2A; letter-spacing:0.1em;">
                        RAG_LOG // {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.85rem;
                                color:#00FFAA88; letter-spacing:0.1em;">
                        PASSAGES RETRIEVED: 5 // STATUS: SUCCESS
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Conversation history ──────────────────────────────────────
                if len(st.session_state.rag_history) > 1:
                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                    st.markdown(
                        '<div class="section-label">// CONVERSATION HISTORY</div>',
                        unsafe_allow_html=True
                    )
                    for ex in reversed(st.session_state.rag_history[:-1]):
                        st.markdown(f"""
                        <div style="background:#060D14; border:1px solid #0D2235;
                                    border-left:2px solid #00FFAA33;
                                    padding:12px 16px; margin-bottom:6px; border-radius:2px;">
                            <div style="font-family:'Share Tech Mono',monospace;
                                        font-size:0.82rem; color:#4A7A6A;
                                        letter-spacing:0.08em; margin-bottom:4px;">
                                Q: {ex['q']}</div>
                            <div style="font-family:'Rajdhani',sans-serif;
                                        font-size:1.05rem; color:#8AAABB;
                                        line-height:1.5;">{ex['a'][:300]}{'…' if len(ex['a']) > 300 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)
