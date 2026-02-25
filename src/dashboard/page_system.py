"""
System configuration page — environment status, Pinecone health, ingestion info.
"""

import json
import streamlit as st
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def render():
    st.markdown("### ⚙️ System Status")
    st.markdown(
        '<p class="sub-header">Environment, connections & data pipeline status</p>',
        unsafe_allow_html=True,
    )

    # ── Environment Check ────────────────────────────────────────────────
    st.markdown("#### Environment")

    from src.utils.config import PINECONE_API_KEY, GROQ_API_KEY, GOVUK_BASE_URL

    env_items = [
        ("Pinecone API Key", bool(PINECONE_API_KEY)),
        ("Groq API Key", bool(GROQ_API_KEY)),
        ("GOV.UK Base URL", bool(GOVUK_BASE_URL)),
    ]

    for label, ok in env_items:
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} **{label}** — {'Configured' if ok else 'Missing'}")

    st.markdown("---")

    # ── Pinecone Connection ──────────────────────────────────────────────
    st.markdown("#### Pinecone Index")

    if st.button("🔗 Test Connection", key="test_pinecone"):
        with st.spinner("Connecting to Pinecone…"):
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index("rait-chatbot")
                stats = index.describe_index_stats()

                st.success("Connected to Pinecone ✓")

                cols = st.columns(3)
                total = stats.get("total_vector_count", 0)
                namespaces = stats.get("namespaces", {})
                cols[0].metric("Total Vectors", f"{total:,}")
                cols[1].metric("Namespaces", len(namespaces))
                cols[2].metric("Dimension", stats.get("dimension", "?"))

                if namespaces:
                    with st.expander("Namespace details"):
                        for ns, info in namespaces.items():
                            vc = info.get("vector_count", 0)
                            st.markdown(f"- **{ns}**: {vc:,} vectors")

            except Exception as e:
                st.error(f"Connection failed: {e}")

    st.markdown("---")

    # ── Ingestion Summary ────────────────────────────────────────────────
    st.markdown("#### Ingestion Status")

    summary_path = DATA_DIR / "processed" / "ingestion_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        cols = st.columns(4)
        cols[0].metric("Pages Attempted", summary.get("pages_attempted", 0))
        cols[1].metric("Pages Succeeded", summary.get("pages_succeeded", 0))
        cols[2].metric("Total Chunks", summary.get("total_chunks", 0))
        ts = summary.get("timestamp", "")
        cols[3].metric("Last Ingestion", ts[:16] if ts else "Never")

        per_page = summary.get("per_page", [])
        if per_page:
            with st.expander("Per-page details"):
                import pandas as pd
                df = pd.DataFrame(per_page)
                st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No ingestion has been run yet. Go to Tests → Integration → Run Ingestion.")

    st.markdown("---")

    # ── Database Stats ───────────────────────────────────────────────────
    st.markdown("#### Production Log Database")

    db_path = DATA_DIR / "logs" / "production.db"
    if db_path.exists():
        try:
            from src.production.logging import LogStore
            store = LogStore(db_path)
            count = store.count()
            st.metric("Total Logged Queries", f"{count:,}")

            if count > 0:
                logs = store.read_all()
                latest = logs[-1]
                st.caption(f"Latest query: _{latest.raw_query[:80]}_ ({latest.timestamp[:16]})")

                # Topic distribution
                from collections import Counter
                topics = Counter(l.query_topic for l in logs)
                if topics:
                    with st.expander("Topic distribution"):
                        import pandas as pd
                        df = pd.DataFrame(topics.most_common(), columns=["Topic", "Count"])
                        st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not read database: {e}")
    else:
        st.info("No production logs yet. Use the Chatbot to generate queries.")

    st.markdown("---")

    # ── Monitoring Reports ───────────────────────────────────────────────
    st.markdown("#### Monitoring Reports")
    results_dir = DATA_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if reports:
        st.caption(f"{len(reports)} report(s) found")
        for rp in reports[:10]:
            st.markdown(f"📄 `{rp.name}`")
    else:
        st.info("No monitoring reports yet. Run monitoring from the dashboard.")
