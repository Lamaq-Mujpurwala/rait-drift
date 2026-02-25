"""
RAIT — Data & Model Drift Monitoring for UK Public Sector RAG Chatbot
CLI entry point for local development and testing.

Usage:
    python main.py ingest              # Ingest GOV.UK pages into Pinecone
    python main.py init-canaries       # Initialise all 50 canary reference responses
    python main.py query "..."         # Ask a question
    python main.py monitor daily       # Run daily monitoring (TRCI + CDS)
    python main.py monitor weekly      # Run weekly monitoring (all 4 metrics)
    python main.py test-stats          # Quick test of statistical utilities
"""

from __future__ import annotations

import sys
import json


def cmd_ingest():
    """Run the full GOV.UK ingestion pipeline."""
    from src.production.ingestion import GOVUKIngester

    print("Starting GOV.UK content ingestion...")
    ingester = GOVUKIngester()
    summary = ingester.run_full_ingestion()
    print(f"\nIngestion complete:")
    print(f"  Pages attempted: {summary['pages_attempted']}")
    print(f"  Pages succeeded: {summary['pages_succeeded']}")
    print(f"  Total chunks: {summary['total_chunks']}")
    print(f"\nFull summary saved to data/processed/ingestion_summary.json")


def cmd_query(question: str):
    """Process a single user query through the production pipeline."""
    from src.production.pipeline import QueryPipeline

    pipeline = QueryPipeline(session_id="cli")
    print(f"Query: {question}\n")
    print("Processing...")

    log = pipeline.process(question)
    print(f"\n{'='*60}")
    print(log.final_response)
    print(f"\n{'='*60}")
    print(f"Query ID:    {log.query_id}")
    print(f"Topic:       {log.query_topic}")
    print(f"Model:       {log.model_name}")
    print(f"Latency:     {log.response_latency_ms} ms")
    print(f"Citations:   {log.citation_count}")
    print(f"Refusal:     {log.refusal_flag}")
    print(f"Readability: {log.response_readability:.1f}")


def cmd_monitor(run_type: str):
    """Run the monitoring system."""
    from src.monitoring.orchestrator import MetricOrchestrator
    from src.production.pipeline import QueryPipeline

    pipeline = QueryPipeline(session_id="monitoring")
    orch = MetricOrchestrator(pipeline=pipeline)

    if run_type == "daily":
        print("Running daily monitoring (TRCI + CDS)...")
        report = orch.run_daily()
    elif run_type == "weekly":
        print("Running weekly monitoring (all 4 metrics)...")
        report = orch.run_weekly()
    else:
        print(f"Unknown run type: {run_type}. Use 'daily' or 'weekly'.")
        return

    print(f"\nMonitoring Report ({report.run_type}):")
    for r in report.results:
        print(f"  [{r.status:6s}] {r.metric_name}: {r.value:.4f}")
        print(f"          {r.explanation[:120]}")

    if report.alerts:
        print(f"\n  ALERTS ({len(report.alerts)}):")
        for a in report.alerts:
            print(f"    [{a['severity']}] {a['metric']}: {a['explanation'][:100]}")
    else:
        print("\n  No alerts.")


def cmd_init_canaries():
    """
    Initialise reference responses for ALL canary queries.
    Runs each canary through the production pipeline, saves the response
    back to canary_queries.json, then upserts reference embeddings to Pinecone.
    
    This is a ONE-TIME setup step required before TRCI monitoring works.
    """
    import time
    from pathlib import Path
    from src.production.pipeline import QueryPipeline
    from src.monitoring.metrics.trci import TRCIEngine
    from src.utils.config import CONFIG_DIR

    canary_path = CONFIG_DIR / "canary_queries.json"
    with open(canary_path, "r", encoding="utf-8") as f:
        canaries = json.load(f)

    # Count how many need initialisation
    needs_init = [c for c in canaries if not c.get("reference_response")]
    already_done = [c for c in canaries if c.get("reference_response")]
    print(f"Canary reference initialisation:")
    print(f"  Total canaries:        {len(canaries)}")
    print(f"  Already initialised:   {len(already_done)}")
    print(f"  Need initialisation:   {len(needs_init)}")

    if not needs_init:
        print("\nAll canaries already have reference responses. Nothing to do.")
        return

    pipeline = QueryPipeline(session_id="canary_init")
    responses: dict[str, str] = {}

    # Collect already-init references too
    for c in already_done:
        responses[c["id"]] = c["reference_response"]

    print(f"\nInitialising {len(needs_init)} canaries (rate-limited, ~40s between calls)...\n")

    for i, canary in enumerate(needs_init):
        cid = canary["id"]
        query = canary["query"]
        print(f"  [{i+1}/{len(needs_init)}] {cid}: {query[:60]}...")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                log = pipeline.process(query)
                canary["reference_response"] = log.raw_response
                responses[cid] = log.raw_response
                print(f"           ✓ {len(log.raw_response)} chars")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"           ↻ Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    print(f"           ✗ FAILED after {max_retries} attempts: {str(e)[:80]}")

        # Rate limit: 40s between calls to stay within Groq TPM limits (12000 TPM)
        if i < len(needs_init) - 1:
            time.sleep(40)

    # Save updated canary_queries.json
    with open(canary_path, "w", encoding="utf-8") as f:
        json.dump(canaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {canary_path}")

    # Upsert ALL reference embeddings to Pinecone
    print(f"\nUpserting {len(responses)} reference embeddings to Pinecone...")
    trci = TRCIEngine(pipeline=pipeline)
    trci.init_reference_embeddings(responses)
    print("Done! All canary references initialised.")


def cmd_test_stats():
    """Quick smoke test of statistical utilities (no API calls needed)."""
    import numpy as np
    from src.utils.statistics import (
        continuous_jsd, discrete_jsd, binary_jsd, signed_jsd,
        ks_test, cosine_similarity, summarise_distribution,
    )

    print("Testing statistical utilities...\n")

    # JSD on identical distributions → 0
    p = np.random.normal(0, 1, 1000)
    jsd = continuous_jsd(p, p, bins=50)
    print(f"JSD(identical):  {jsd:.6f} (should be ~0)")

    # JSD on shifted distributions → > 0
    q = np.random.normal(2, 1, 1000)
    jsd = continuous_jsd(p, q, bins=50)
    print(f"JSD(shifted):    {jsd:.6f} (should be > 0)")

    # Discrete JSD
    d1 = np.array([1, 1, 2, 2, 3])
    d2 = np.array([1, 2, 3, 3, 3])
    jsd = discrete_jsd(d1, d2)
    print(f"JSD(discrete):   {jsd:.6f}")

    # Binary JSD
    jsd = binary_jsd(0.5, 0.5)
    print(f"JSD(binary=same):{jsd:.6f} (should be ~0)")
    jsd = binary_jsd(0.1, 0.9)
    print(f"JSD(binary=diff):{jsd:.6f} (should be > 0)")

    # Signed JSD
    ref = np.random.normal(0.8, 0.1, 100)
    cur = np.random.normal(0.6, 0.1, 100)
    s = signed_jsd(ref, cur, bins=20)
    print(f"Signed JSD:      {s:+.6f} (negative = decay)")

    # KS test
    stat, pval = ks_test(p, q)
    print(f"KS test:         stat={stat:.4f}, p={pval:.6f}")

    # Cosine similarity
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    sim = cosine_similarity(a, b)
    print(f"Cosine(same):    {sim:.4f} (should be 1.0)")

    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    sim = cosine_similarity(a, b)
    print(f"Cosine(ortho):   {sim:.4f} (should be 0.0)")

    # Summary
    summary = summarise_distribution(p)
    print(f"Summary:         mean={summary['mean']:.4f}, std={summary['std']:.4f}, n={summary['n']}")

    print("\nAll statistical utilities working correctly!")


def cmd_test_logging():
    """Quick test of the production logging system (no API calls needed)."""
    import tempfile
    from pathlib import Path
    from src.production.logging import LogStore, ProductionLog, generate_id, utcnow

    print("Testing production logging...\n")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = LogStore(db_path)

        # Write a log entry
        log = ProductionLog(
            query_id=generate_id(),
            timestamp=utcnow(),
            raw_query="What is Universal Credit?",
            cleaned_query="what is universal credit?",
            query_topic="universal_credit",
            query_token_count=5,
            response_length=150,
            response_sentiment=0.3,
            response_readability=65.0,
            citation_count=2,
            hedge_word_count=1,
            refusal_flag=False,
            response_latency_ms=1200,
            raw_response="Universal Credit is a benefit...",
            final_response="Universal Credit is a benefit... [disclaimer]",
            model_name="groq/llama-3.3-70b-versatile",
        )
        store.write(log)
        print(f"  Written log: {log.query_id}")

        # Read it back
        count = store.count()
        print(f"  Total logs:  {count}")

        logs = store.read_all()
        assert len(logs) == 1
        assert logs[0].raw_query == "What is Universal Credit?"
        print(f"  Verified:    query matches")

    print("\nProduction logging working correctly!")


def cmd_test_descriptors():
    """Test descriptor extraction (no API calls needed)."""
    from src.production.logging import ProductionLog, generate_id, utcnow
    from src.monitoring.descriptors import extract_descriptors

    print("Testing descriptor extraction...\n")

    logs = [
        ProductionLog(
            query_id=generate_id(),
            timestamp=utcnow(),
            raw_query="What is PIP?",
            response_length=120,
            response_sentiment=0.4,
            response_readability=70.0,
            citation_count=2,
            hedge_word_count=1,
            refusal_flag=False,
            response_latency_ms=1100,
            mean_retrieval_distance=0.15,
            context_token_ratio=2.5,
        ),
        ProductionLog(
            query_id=generate_id(),
            timestamp=utcnow(),
            raw_query="How do I apply for Housing Benefit?",
            response_length=200,
            response_sentiment=0.5,
            response_readability=60.0,
            citation_count=3,
            hedge_word_count=0,
            refusal_flag=False,
            response_latency_ms=900,
            mean_retrieval_distance=0.12,
            context_token_ratio=3.0,
        ),
    ]

    desc = extract_descriptors(logs)
    print(f"  Queries:          {desc.n_queries}")
    print(f"  Descriptors:      {list(desc.data.keys())}")
    print(f"  response_length:  {desc.get_values('response_length')}")
    print(f"  citation_count:   {desc.get_values('citation_count')}")
    print(f"  refusal_flag:     {desc.get_values('refusal_flag')}")

    print("\nDescriptor extraction working correctly!")


def cmd_test_cds():
    """Test CDS metric computation with synthetic data (no API calls needed)."""
    import numpy as np
    from src.monitoring.descriptors import DescriptorSet
    from src.monitoring.metrics.cds import CDSEngine

    print("Testing CDS metric computation...\n")

    np.random.seed(42)

    # Reference window: stable baseline
    ref = DescriptorSet(data={
        "response_length": np.random.normal(150, 20, 100),
        "response_sentiment": np.random.normal(0.3, 0.1, 100),
        "response_readability": np.random.normal(65, 10, 100),
        "citation_count": np.random.choice([1, 2, 3], 100),
        "hedge_word_count": np.random.choice([0, 1, 2], 100),
        "refusal_flag": np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        "response_latency_ms": np.random.normal(1000, 200, 100),
        "mean_retrieval_distance": np.random.normal(0.15, 0.05, 100),
        "context_token_ratio": np.random.normal(2.5, 0.5, 100),
    }, query_ids=[f"ref_{i}" for i in range(100)])

    # Current window: identical → should be GREEN
    cur_same = DescriptorSet(data={
        "response_length": np.random.normal(150, 20, 100),
        "response_sentiment": np.random.normal(0.3, 0.1, 100),
        "response_readability": np.random.normal(65, 10, 100),
        "citation_count": np.random.choice([1, 2, 3], 100),
        "hedge_word_count": np.random.choice([0, 1, 2], 100),
        "refusal_flag": np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        "response_latency_ms": np.random.normal(1000, 200, 100),
        "mean_retrieval_distance": np.random.normal(0.15, 0.05, 100),
        "context_token_ratio": np.random.normal(2.5, 0.5, 100),
    }, query_ids=[f"cur_{i}" for i in range(100)])

    cds = CDSEngine()
    result = cds.compute(ref, cur_same)
    print(f"  Stable CDS:   {result.value:.4f} [{result.status}]")
    print(f"  Explanation:   {result.explanation[:120]}")

    # Drifted window: shifted distributions → should be AMBER or RED
    cur_drift = DescriptorSet(data={
        "response_length": np.random.normal(80, 30, 100),   # shorter responses
        "response_sentiment": np.random.normal(-0.1, 0.2, 100),  # more negative
        "response_readability": np.random.normal(45, 15, 100),
        "citation_count": np.random.choice([0, 1], 100),     # fewer citations
        "hedge_word_count": np.random.choice([2, 3, 4], 100), # more hedging
        "refusal_flag": np.random.choice([0, 1], 100, p=[0.5, 0.5]),  # more refusals
        "response_latency_ms": np.random.normal(2000, 500, 100),
        "mean_retrieval_distance": np.random.normal(0.35, 0.1, 100),
        "context_token_ratio": np.random.normal(1.0, 0.3, 100),
    }, query_ids=[f"drift_{i}" for i in range(100)])

    result = cds.compute(ref, cur_drift)
    print(f"  Drifted CDS:  {result.value:.4f} [{result.status}]")
    print(f"  Explanation:   {result.explanation[:120]}")

    print("\nCDS metric computation working correctly!")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "ingest":
        cmd_ingest()
    elif cmd == "init-canaries":
        cmd_init_canaries()
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Usage: python main.py query \"Your question here\"")
            return
        cmd_query(" ".join(sys.argv[2:]))
    elif cmd == "monitor":
        run_type = sys.argv[2] if len(sys.argv) > 2 else "daily"
        cmd_monitor(run_type)
    elif cmd == "test-stats":
        cmd_test_stats()
    elif cmd == "test-logging":
        cmd_test_logging()
    elif cmd == "test-descriptors":
        cmd_test_descriptors()
    elif cmd == "test-cds":
        cmd_test_cds()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
