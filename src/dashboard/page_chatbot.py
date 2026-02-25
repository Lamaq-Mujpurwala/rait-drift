"""
Chatbot page — RAG-powered conversational interface for UK government services.
"""

import streamlit as st
from datetime import datetime


def render():
    st.markdown("### 💬 GOV.UK Assistant")
    st.markdown(
        '<p class="sub-header">Ask questions about UK housing, benefits & council services</p>',
        unsafe_allow_html=True,
    )

    # ── Session State Init ───────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

    # ── Lazy Pipeline Init ───────────────────────────────────────────────
    def _get_pipeline():
        if st.session_state.pipeline is None:
            try:
                from src.production.pipeline import QueryPipeline
                st.session_state.pipeline = QueryPipeline(session_id="streamlit")
            except Exception as e:
                st.error(f"Failed to initialise pipeline: {e}")
                return None
        return st.session_state.pipeline

    # ── Chat History ─────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if msg.get("meta"):
                    meta = msg["meta"]
                    cols = st.columns(5)
                    cols[0].caption(f"⏱ {meta.get('latency', '?')} ms")
                    cols[1].caption(f"📎 {meta.get('citations', '?')} citations")
                    cols[2].caption(f"📖 {meta.get('readability', '?')}")
                    cols[3].caption(f"🏷️ {meta.get('topic', '?')}")
                    cols[4].caption(f"🔗 {meta.get('chunks', '?')} chunks")

    # ── Chat Input ───────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about UK benefits, housing, council tax…"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        pipeline = _get_pipeline()
        if pipeline is None:
            with st.chat_message("assistant"):
                st.error("Pipeline not available. Please check system configuration.")
            return

        with st.chat_message("assistant"):
            with st.spinner("Searching GOV.UK knowledge base…"):
                try:
                    log = pipeline.process(prompt)
                    response = log.final_response
                    meta = {
                        "latency": log.response_latency_ms,
                        "citations": log.citation_count,
                        "readability": f"{log.response_readability:.0f}",
                        "topic": log.query_topic.replace("_", " ").title(),
                        "chunks": len(log.selected_context_ids),
                        "query_id": log.query_id,
                        "sentiment": f"{log.response_sentiment:.2f}",
                        "hedges": log.hedge_word_count,
                        "refusal": log.refusal_flag,
                    }
                    st.markdown(response)
                    cols = st.columns(5)
                    cols[0].caption(f"⏱ {meta['latency']} ms")
                    cols[1].caption(f"📎 {meta['citations']} citations")
                    cols[2].caption(f"📖 {meta['readability']}")
                    cols[3].caption(f"🏷️ {meta['topic']}")
                    cols[4].caption(f"🔗 {meta['chunks']} chunks")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "meta": meta,
                    })
                    st.session_state.total_queries += 1
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Error generating response: {e}",
                    })

    # ── Sidebar Stats ────────────────────────────────────────────────────
    with st.sidebar:
        if st.session_state.total_queries > 0:
            st.markdown("---")
            st.caption(f"Session queries: {st.session_state.total_queries}")

        if st.button("🗑 Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.rerun()
