"""
RAIT — Streamlit Application Entry Point
Unified interface for RAG Chatbot & Drift Monitoring Dashboard.
"""

import streamlit as st

# ── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="RAIT · Drift Monitor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for sleek dark theme ──────────────────────────────────────────
st.markdown("""
<style>
    /* Global overrides */
    .stApp { background-color: #0f172a; }
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0;
    }

    /* Header styling */
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0;
        padding-top: 0;
    }

    /* Metric cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.6rem;
    }
    .metric-card h3 {
        margin: 0 0 0.3rem 0;
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1;
    }

    /* Status badges */
    .badge { display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em; }
    .badge-green { background: #064e3b; color: #6ee7b7; border: 1px solid #065f46; }
    .badge-amber { background: #78350f; color: #fbbf24; border: 1px solid #92400e; }
    .badge-red { background: #7f1d1d; color: #fca5a5; border: 1px solid #991b1b; }

    /* Chat messages */
    .user-msg {
        background: #1e3a5f;
        border-radius: 12px 12px 4px 12px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        color: #e2e8f0;
    }
    .bot-msg {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px 12px 12px 4px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        color: #e2e8f0;
    }

    /* Test output */
    .test-pass { color: #6ee7b7; font-weight: 600; }
    .test-fail { color: #fca5a5; font-weight: 600; }
    .test-output {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
        color: #cbd5e1;
        white-space: pre-wrap;
        max-height: 500px;
        overflow-y: auto;
    }

    /* Buttons */
    .stButton > button {
        background: #2563eb;
        color: #f8fafc;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: background 0.15s;
    }
    .stButton > button:hover {
        background: #1d4ed8;
        color: #f8fafc;
    }

    /* Dividers */
    hr { border-color: #334155; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header">RAIT</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Data & Model Drift Monitor<br>UK Public Sector RAG System</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["💬  Chatbot", "📊  Monitoring", "🧪  Tests", "⚙️  System"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("RAIT Intern Assessment · 2026")
    st.caption("Ethical Dimension: Data & Model Drift")


# ── Page Router ──────────────────────────────────────────────────────────────
if "💬" in page:
    from src.dashboard.page_chatbot import render
    render()
elif "📊" in page:
    from src.dashboard.page_monitoring import render
    render()
elif "🧪" in page:
    from src.dashboard.page_tests import render
    render()
elif "⚙️" in page:
    from src.dashboard.page_system import render
    render()
