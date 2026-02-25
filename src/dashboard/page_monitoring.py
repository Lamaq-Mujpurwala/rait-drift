"""
Monitoring dashboard — drift metrics visualisation & alerting.
"""

import json
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timezone, timedelta


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"


def _status_badge(status: str) -> str:
    s = status.upper()
    if s == "GREEN":
        return '<span class="badge badge-green">GREEN</span>'
    elif s == "AMBER":
        return '<span class="badge badge-amber">AMBER</span>'
    elif s == "RED":
        return '<span class="badge badge-red">RED</span>'
    return f'<span class="badge">{s}</span>'


def _metric_card(label: str, value: str, status: str = "") -> str:
    badge = _status_badge(status) if status else ""
    return f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <div class="value">{value} {badge}</div>
    </div>
    """


def _load_latest_report() -> dict | None:
    """Load the most recent monitoring report JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        return None
    with open(reports[0], "r", encoding="utf-8") as f:
        return json.load(f)


def _load_all_reports() -> list[dict]:
    """Load all monitoring reports for history plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for p in sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        with open(p, "r", encoding="utf-8") as f:
            reports.append(json.load(f))
    return reports


def render():
    st.markdown("### 📊 Drift Monitoring Dashboard")
    st.markdown(
        '<p class="sub-header">Real-time data & model drift detection across 4 metrics</p>',
        unsafe_allow_html=True,
    )

    # ── Controls ─────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([1, 1, 2])

    with col_a:
        if st.button("▶ Run Daily", key="run_daily", use_container_width=True):
            _run_monitoring("daily")

    with col_b:
        if st.button("▶ Run Weekly", key="run_weekly", use_container_width=True):
            _run_monitoring("weekly")

    with col_c:
        metric_choice = st.selectbox(
            "Single metric",
            ["TRCI", "CDS", "FDS", "DDI"],
            label_visibility="collapsed",
            key="single_metric",
        )
        if st.button("▶ Run Single", key="run_single"):
            _run_single_metric(metric_choice)

    st.markdown("---")

    # ── Latest Report ────────────────────────────────────────────────────
    report = _load_latest_report()

    if report is None:
        st.info("No monitoring reports yet. Run a daily or weekly check to get started.")
        return

    # ── Metric Overview Cards ────────────────────────────────────────────
    results = report.get("results", [])
    if results:
        cols = st.columns(len(results))
        for i, r in enumerate(results):
            with cols[i]:
                st.markdown(
                    _metric_card(r["metric_name"], f"{r['value']:.4f}", r["status"]),
                    unsafe_allow_html=True,
                )

    # ── Alerts ───────────────────────────────────────────────────────────
    alerts = report.get("alerts", [])
    if alerts:
        st.markdown("#### Alerts")
        for a in alerts:
            severity = a.get("severity", "INFO")
            icon = "🔴" if severity == "CRITICAL" else "🟡"
            st.warning(f"{icon} **{a['metric']}** — {a['explanation'][:200]}")
    else:
        st.success("No active alerts.")

    st.markdown("---")

    # ── Detailed Metric Panels ───────────────────────────────────────────
    tabs = st.tabs([r["metric_name"] for r in results]) if results else []

    for tab, r in zip(tabs, results):
        with tab:
            st.markdown(f"**Status:** {r['status']} &nbsp; **Value:** `{r['value']:.6f}`")
            st.markdown(f"_{r['explanation']}_")

            details = r.get("details", {})

            # CDS: per-descriptor breakdown
            if r["metric_name"] == "CDS" and "per_descriptor" in details:
                _render_cds_details(details)

            # TRCI: per-canary table
            elif r["metric_name"] == "TRCI" and "per_canary" in details:
                _render_trci_details(details)

            # DDI: per-segment breakdown
            elif r["metric_name"] == "DDI" and "per_segment" in details:
                _render_ddi_details(details)

            # FDS: faithfulness details
            elif r["metric_name"] == "FDS":
                _render_fds_details(details)

            # Raw JSON fallback
            with st.expander("Raw details"):
                st.json(details)

    # ── History Timeline ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Metric History")
    _render_history()

    # ── Report metadata ──────────────────────────────────────────────────
    st.caption(f"Report type: {report.get('run_type', '?')} · {report.get('timestamp', '?')}")


def _run_monitoring(run_type: str):
    """Execute monitoring run and display results."""
    with st.spinner(f"Running {run_type} monitoring…"):
        try:
            from src.monitoring.orchestrator import MetricOrchestrator
            from src.production.pipeline import QueryPipeline

            pipeline = QueryPipeline(session_id="monitoring")
            orch = MetricOrchestrator(pipeline=pipeline)

            if run_type == "daily":
                report = orch.run_daily()
            else:
                report = orch.run_weekly()

            st.success(f"{run_type.title()} monitoring complete — {len(report.results)} metrics evaluated.")
            st.rerun()
        except Exception as e:
            st.error(f"Monitoring failed: {e}")


def _run_single_metric(metric_name: str):
    """Run a single metric on demand."""
    with st.spinner(f"Running {metric_name}…"):
        try:
            from src.monitoring.orchestrator import MetricOrchestrator
            from src.production.pipeline import QueryPipeline

            pipeline = QueryPipeline(session_id="monitoring")
            orch = MetricOrchestrator(pipeline=pipeline)
            result = orch.run_single_metric(metric_name)

            st.success(f"{metric_name}: {result.value:.4f} [{result.status}]")
            st.markdown(f"_{result.explanation}_")
        except Exception as e:
            st.error(f"Metric execution failed: {e}")


def _render_cds_details(details: dict):
    """Render CDS per-descriptor drift breakdown."""
    per_desc = details.get("per_descriptor", {})
    if not per_desc:
        return

    names = list(per_desc.keys())
    jsds = [per_desc[n].get("jsd", 0) for n in names]
    weights = [per_desc[n].get("weight", 0) for n in names]
    contribs = [per_desc[n].get("weighted_contribution", 0) for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=jsds,
        marker_color=["#ef4444" if j > 0.15 else "#f59e0b" if j > 0.05 else "#22c55e" for j in jsds],
        text=[f"{j:.3f}" for j in jsds],
        textposition="outside",
    ))
    fig.add_hline(y=0.05, line_dash="dot", line_color="#22c55e", annotation_text="GREEN")
    fig.add_hline(y=0.15, line_dash="dot", line_color="#f59e0b", annotation_text="AMBER")
    fig.update_layout(
        title="Per-Descriptor JSD",
        xaxis_title="Descriptor",
        yaxis_title="JSD",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_trci_details(details: dict):
    """Render TRCI canary probe results."""
    canaries = details.get("per_canary", [])
    if not canaries:
        return

    import pandas as pd
    df = pd.DataFrame(canaries)
    if "similarity" in df.columns:
        df["similarity"] = df["similarity"].round(4)
    st.dataframe(
        df[["canary_id", "topic", "similarity", "status"]].head(20),
        use_container_width=True,
        hide_index=True,
    )


def _render_ddi_details(details: dict):
    """Render DDI per-segment fairness breakdown."""
    per_seg = details.get("per_segment", {})
    if not per_seg:
        return

    names = list(per_seg.keys())
    drifts = [per_seg[n].get("drift_score", 0) for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=drifts,
        marker_color=["#ef4444" if d > 0.15 else "#f59e0b" if d > 0.05 else "#22c55e" for d in drifts],
        text=[f"{d:.3f}" for d in drifts],
        textposition="outside",
    ))
    fig.update_layout(
        title="Per-Segment Drift (DDI)",
        xaxis_title="Topic Segment",
        yaxis_title="Drift Score",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_fds_details(details: dict):
    """Render FDS faithfulness details."""
    st.markdown(f"**FDS Value:** `{details.get('fds_value', 0):.4f}`")
    if "per_query" in details:
        import pandas as pd
        queries = details["per_query"][:20]
        if queries:
            df = pd.DataFrame(queries)
            st.dataframe(df, use_container_width=True, hide_index=True)


def _render_history():
    """Render historical metric trends."""
    reports = _load_all_reports()
    if len(reports) < 2:
        st.caption("Insufficient history for trend plot. Run monitoring multiple times.")
        return

    # Collect time series
    from collections import defaultdict
    series = defaultdict(lambda: {"timestamps": [], "values": [], "statuses": []})

    for report in reports:
        ts = report.get("timestamp", "")
        for r in report.get("results", []):
            name = r["metric_name"]
            series[name]["timestamps"].append(ts)
            series[name]["values"].append(r["value"])
            series[name]["statuses"].append(r["status"])

    fig = go.Figure()
    colors = {"TRCI": "#3b82f6", "CDS": "#f59e0b", "FDS": "#8b5cf6", "DDI": "#ef4444"}

    for name, data in series.items():
        fig.add_trace(go.Scatter(
            x=data["timestamps"],
            y=data["values"],
            mode="lines+markers",
            name=name,
            line=dict(color=colors.get(name, "#94a3b8"), width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Metric Value",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
