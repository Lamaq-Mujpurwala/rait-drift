"""
Test runner page — execute pytest and CLI tests directly from the GUI.
"""

import subprocess
import sys
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable


def _run_command(cmd: list[str], label: str) -> tuple[bool, str]:
    """Run a subprocess and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"{label} timed out after 120 seconds."
    except Exception as e:
        return False, f"Error running {label}: {e}"


def render():
    st.markdown("### 🧪 Test Suite")
    st.markdown(
        '<p class="sub-header">Run and verify all system tests from here</p>',
        unsafe_allow_html=True,
    )

    # ── Full Pytest Suite ────────────────────────────────────────────────
    st.markdown("#### Pytest Suite")
    st.caption("Runs all unit tests: statistics, logging, text processing, CDS, DDI")

    col1, col2 = st.columns([1, 3])
    with col1:
        run_pytest = st.button("▶ Run All Tests", key="run_pytest", use_container_width=True)

    if run_pytest:
        with st.spinner("Running pytest…"):
            success, output = _run_command(
                [PYTHON, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header"],
                "pytest",
            )

        if success:
            st.success("All tests passed ✓")
        else:
            st.error("Some tests failed ✗")

        # Parse and display results
        _display_pytest_output(output)

    st.markdown("---")

    # ── CLI Smoke Tests ──────────────────────────────────────────────────
    st.markdown("#### CLI Smoke Tests")
    st.caption("Offline tests that verify core functionality without API calls")

    tests = [
        ("test-stats", "Statistical Utilities", "JSD, KS test, cosine similarity"),
        ("test-logging", "Production Logging", "SQLite write / read / round-trip"),
        ("test-descriptors", "Descriptor Extraction", "All 9 CDS descriptors"),
        ("test-cds", "CDS Metric Engine", "Drift detection with synthetic data"),
    ]

    cols = st.columns(len(tests))
    for i, (cmd, label, desc) in enumerate(tests):
        with cols[i]:
            st.markdown(f"**{label}**")
            st.caption(desc)
            if st.button(f"▶ Run", key=f"cli_{cmd}", use_container_width=True):
                with st.spinner(f"Running {label}…"):
                    success, output = _run_command(
                        [PYTHON, "main.py", cmd],
                        label,
                    )
                if success:
                    st.success("Passed ✓")
                else:
                    st.error("Failed ✗")
                st.markdown(f'<div class="test-output">{_escape_html(output)}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Integration Tests ────────────────────────────────────────────────
    st.markdown("#### Integration Tests")
    st.caption("Tests that require API keys (Pinecone + Groq)")

    icol1, icol2, icol3 = st.columns(3)

    with icol1:
        st.markdown("**Data Ingestion**")
        st.caption("Fetch GOV.UK → chunk → upsert to Pinecone")
        if st.button("▶ Run Ingestion", key="run_ingest", use_container_width=True):
            with st.spinner("Ingesting GOV.UK content (this may take a few minutes)…"):
                success, output = _run_command(
                    [PYTHON, "main.py", "ingest"],
                    "Ingestion",
                )
            if success:
                st.success("Ingestion complete ✓")
            else:
                st.error("Ingestion failed ✗")
            st.markdown(f'<div class="test-output">{_escape_html(output)}</div>', unsafe_allow_html=True)

    with icol2:
        st.markdown("**Query Pipeline**")
        st.caption("Full RAG: retrieve → generate → log")
        test_query = st.text_input("Test query", "What is Universal Credit?", key="test_query")
        if st.button("▶ Run Query", key="run_query", use_container_width=True):
            with st.spinner("Processing query…"):
                success, output = _run_command(
                    [PYTHON, "main.py", "query", test_query],
                    "Query Pipeline",
                )
            if success:
                st.success("Query processed ✓")
            else:
                st.error("Query failed ✗")
            st.markdown(f'<div class="test-output">{_escape_html(output)}</div>', unsafe_allow_html=True)

    with icol3:
        st.markdown("**Daily Monitoring**")
        st.caption("TRCI canary probe + CDS drift check")
        if st.button("▶ Run Monitor", key="run_monitor", use_container_width=True):
            with st.spinner("Running daily monitoring…"):
                success, output = _run_command(
                    [PYTHON, "main.py", "monitor", "daily"],
                    "Monitoring",
                )
            if success:
                st.success("Monitoring complete ✓")
            else:
                st.error("Monitoring failed ✗")
            st.markdown(f'<div class="test-output">{_escape_html(output)}</div>', unsafe_allow_html=True)


def _display_pytest_output(output: str):
    """Parse and display pytest output with colour coding."""
    lines = output.strip().split("\n")

    # Count results
    passed = sum(1 for l in lines if "PASSED" in l)
    failed = sum(1 for l in lines if "FAILED" in l)
    errors = sum(1 for l in lines if "ERROR" in l)

    # Summary bar
    cols = st.columns(4)
    cols[0].metric("Total", passed + failed + errors)
    cols[1].metric("Passed", passed)
    cols[2].metric("Failed", failed)
    cols[3].metric("Errors", errors)

    # Detailed output
    with st.expander("Full output", expanded=failed > 0):
        formatted = []
        for line in lines:
            if "PASSED" in line:
                formatted.append(f'<span class="test-pass">✓ {_escape_html(line)}</span>')
            elif "FAILED" in line:
                formatted.append(f'<span class="test-fail">✗ {_escape_html(line)}</span>')
            elif "ERROR" in line:
                formatted.append(f'<span class="test-fail">⚠ {_escape_html(line)}</span>')
            else:
                formatted.append(_escape_html(line))

        st.markdown(
            f'<div class="test-output">{"<br>".join(formatted)}</div>',
            unsafe_allow_html=True,
        )


def _escape_html(text: str) -> str:
    """Escape HTML entities."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
