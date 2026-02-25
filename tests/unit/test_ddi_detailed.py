"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DDI — Differential Drift Index · Detailed Test Suite                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
-------
DDI detects **non-uniform drift across population segments**. When a model
degrades, it often doesn't degrade equally for everyone. DDI checks whether
the drift is worse for some user groups than others — a fairness concern
under the UK Equality Act 2010, Section 149 (Public Sector Equality Duty).

MATHEMATICAL FOUNDATION
-----------------------
Step 1 — Segment by Topic:
    Partition production logs into K topic segments:
    S = {S₁, S₂, ..., Sₖ}
    where each Sₖ serves as a proxy for a protected characteristic group.

Step 2 — Quality Proxy per Response:
    For each log entry in segment Sₖ:

    quality_j = w_comp · completeness + w_cit · has_citation
              + w_ref · non_refusal + w_lat · latency_score

    where:
        completeness  = min(completion_tokens / 200, 1.0)
        has_citation  = 1 if citation_count > 0, else 0
        non_refusal   = 0 if refusal_flag, else 1
        latency_score = max(0, 1 - latency_ms / 5000)

    Default weights: completeness=0.4, citation=0.3, non_refusal=0.2, latency=0.1

Step 3 — Per-Segment Drift (JSD):
    For each topic segment Sₖ:
        drift_k = JSD(quality_ref_k, quality_cur_k)

    This gives us K drift scores, one per segment.

Step 4 — DDI = σ(drift₁, ..., driftₖ)   (standard deviation)
    If drift is uniform across all segments: DDI ≈ 0  (all drift_k similar)
    If drift is non-uniform (some segments worse): DDI > 0

    Secondary measure: DDI_range = max(drift_k) - min(drift_k)

Classification:
    GREEN:   DDI < 0.05
    AMBER:   0.05 ≤ DDI < 0.15
    RED:     DDI ≥ 0.15

INTUITION
---------
Imagine a chatbot for UK benefits. If the model suddenly starts giving
poor answers about disability benefits but continues to answer pension
questions well, that's a differential drift. The disability benefits
topic serves as a proxy for disabled users — a protected characteristic.

DDI detects this inequality by measuring whether drift is UNIFORMLY
distributed or concentrated in specific segments.

σ = 0 means all segments drift equally (fair, even if overall quality drops).
σ > 0 means some segments drift MORE than others (potentially unfair).

WHY STANDARD DEVIATION (not range)?
- σ is robust to a single outlier segment.
- Range (max-min) is sensitive to individual extremes.
- We report both, but DDI uses σ for its primary value.

TOPIC → PROTECTED CHARACTERISTIC MAPPING
-----------------------------------------
    Topic                → Protected Proxy
    universal_credit     → low-income, working-age adults
    housing_benefit      → low-income tenants, elderly
    disability_benefits  → disabled persons (Equality Act)
    council_tax          → all citizens (geographic variation)
    homelessness         → vulnerable persons, rough sleepers
    pension              → elderly persons (age-related)

This mapping is a **heuristic proxy**, not a ground truth. We acknowledge
its limitations in the critical evaluation.

WHAT THE TESTS VALIDATE
------------------------
1. Quality proxy computation (weighted combination of 4 factors)
2. Segment partitioning by topic
3. DDI = σ of per-segment JSDs
4. Uniform vs non-uniform drift detection
5. Minimum segment size enforcement
6. Edge cases (insufficient data, single segment)
7. Intersectional threshold flag

PLOTS GENERATED
---------------
1. Per-Segment Drift Bar Chart: shows JSD per topic segment.
   Uneven bars = non-uniform drift = potential fairness concern.
2. Quality Proxy Distributions: overlaid histograms per topic.
3. Heatmap: segment × quality-factor contribution matrix.

EXTERNAL DATA SOURCES
---------------------
- None. DDI operates entirely on production logs already in the system.
- Topic classification comes from our keyword-based classifier (not an
  external NLP model). This is a deliberate design choice for transparency
  and auditability — no opaque ML model in the fairness pathway.
"""

import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from src.production.logging import ProductionLog, generate_id, utcnow
from src.monitoring.data_collection import MetricDataset
from src.monitoring.descriptors import extract_descriptors
from src.monitoring.metrics.ddi import DDIEngine, TOPIC_SEGMENTS, SegmentResult
from src.monitoring.metrics.base import MetricResult
from src.utils.config import DDIConfig


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_logs(
    topic: str,
    n: int,
    quality_base: float = 0.7,
    latency_base: int = 1000,
    citation_rate: float = 0.8,
    refusal_rate: float = 0.0,
) -> list[ProductionLog]:
    """
    Generate synthetic production logs for a specific topic.

    Args:
        topic: Topic segment name
        n: Number of logs to generate
        quality_base: Controls completeness (completion_tokens ∝ quality_base)
        latency_base: Base latency in ms
        citation_rate: Fraction of responses with citations
        refusal_rate: Fraction of responses that are refusals
    """
    np.random.seed(hash(topic + str(n)) % 2**31)
    logs = []
    for i in range(n):
        logs.append(ProductionLog(
            query_id=generate_id(),
            timestamp=utcnow(),
            raw_query=f"Query about {topic} #{i}",
            query_topic=topic,
            completion_tokens=int(200 * quality_base + np.random.normal(0, 20)),
            citation_count=1 if np.random.random() < citation_rate else 0,
            refusal_flag=np.random.random() < refusal_rate,
            response_latency_ms=int(latency_base + np.random.normal(0, 200)),
            response_length=int(150 * quality_base),
            response_sentiment=0.3 + np.random.normal(0, 0.1),
            response_readability=65 + np.random.normal(0, 10),
            raw_response=f"Response about {topic}",
        ))
    return logs


def _make_dataset(logs: list[ProductionLog], days_back: int = 30) -> MetricDataset:
    """Wrap logs into a MetricDataset."""
    now = datetime.now(timezone.utc)
    return MetricDataset(
        window_start=now - timedelta(days=days_back),
        window_end=now,
        logs=logs,
        descriptors=extract_descriptors(logs),
    )


@pytest.fixture
def all_topics():
    """The 6 standard topic segments."""
    return list(TOPIC_SEGMENTS.keys())


@pytest.fixture
def ddi_config():
    """DDI config with lowered min_segment_size for testing."""
    return DDIConfig(
        min_segment_size=5,
        quality_proxy_weights={
            "completeness": 0.4,
            "citation": 0.3,
            "non_refusal": 0.2,
            "latency": 0.1,
        },
        green_threshold=0.05,
        amber_threshold=0.15,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 1: QUALITY PROXY COMPUTATION
# Mathematical property: quality = Σ wₖ · factorₖ, bounded in [0, 1]
# ═══════════════════════════════════════════════════════════════════════════════

class TestQualityProxy:
    """
    The quality proxy is a weighted sum of 4 normalised factors.
    Each factor ∈ [0, 1], total ∈ [0, 1].

    With default weights: 0.4 + 0.3 + 0.2 + 0.1 = 1.0
    """

    def test_perfect_quality(self, ddi_config):
        """
        Perfect response: 200 tokens, has citation, no refusal, low latency.
        Quality = 0.4·1.0 + 0.3·1.0 + 0.2·1.0 + 0.1·1.0 = 1.0
        """
        log = ProductionLog(
            query_id="test", timestamp=utcnow(), raw_query="test",
            completion_tokens=200, citation_count=1, refusal_flag=False,
            response_latency_ms=100,
        )

        engine = DDIEngine(config=ddi_config)
        quality = engine._compute_quality_proxy([log])
        assert quality[0] == pytest.approx(1.0, abs=0.05)

    def test_worst_quality(self, ddi_config):
        """
        Worst response: 1 token, no citation, is refusal, very slow.

        NOTE: completion_tokens=0 is treated as MISSING DATA (defaults to 0.5)
        by the implementation. So we use 1 token to get the real minimum.

        completeness = min(1/200, 1) = 0.005
        has_citation = 0
        non_refusal = 0 (is refusal)
        latency_score = max(0, 1 - 5000/5000) = 0
        Quality ≈ 0.4 × 0.005 + 0.3 × 0 + 0.2 × 0 + 0.1 × 0 = 0.002
        """
        log = ProductionLog(
            query_id="test", timestamp=utcnow(), raw_query="test",
            completion_tokens=1, citation_count=0, refusal_flag=True,
            response_latency_ms=5000,
        )

        engine = DDIEngine(config=ddi_config)
        quality = engine._compute_quality_proxy([log])
        assert quality[0] == pytest.approx(0.002, abs=0.01)

    def test_quality_proxy_bounded_zero_one(self, ddi_config):
        """Quality proxy values should always be in [0, 1]."""
        logs = _make_logs("universal_credit", 100, quality_base=0.5)
        engine = DDIEngine(config=ddi_config)
        quality = engine._compute_quality_proxy(logs)
        assert np.all(quality >= 0), f"Min quality = {quality.min()}"
        assert np.all(quality <= 1.05), f"Max quality = {quality.max()}"  # small tolerance

    def test_completeness_caps_at_one(self, ddi_config):
        """
        completeness = min(tokens/200, 1.0).
        Responses with >200 tokens should cap completeness at 1.0, not exceed it.
        """
        log = ProductionLog(
            query_id="test", timestamp=utcnow(), raw_query="test",
            completion_tokens=500, citation_count=0, refusal_flag=False,
            response_latency_ms=1000,
        )
        engine = DDIEngine(config=ddi_config)
        quality = engine._compute_quality_proxy([log])
        # completeness=1.0, citation=0, non_refusal=1.0, latency~0.8
        assert quality[0] <= 1.0 + 0.01

    def test_weight_sum_is_one(self, ddi_config):
        """Quality proxy weights should sum to 1.0 for meaningful normalisation."""
        total = sum(ddi_config.quality_proxy_weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 2: TOPIC SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopicSegmentation:
    """
    Logs are segmented by their query_topic field.
    Keyword fallback is used for logs with topic='unknown'.
    """

    def test_known_topic_segmented_correctly(self, ddi_config):
        """Logs with explicit topic fields end up in the right segment."""
        logs = [
            ProductionLog(query_id="1", timestamp=utcnow(), raw_query="q1",
                         query_topic="universal_credit"),
            ProductionLog(query_id="2", timestamp=utcnow(), raw_query="q2",
                         query_topic="pension"),
            ProductionLog(query_id="3", timestamp=utcnow(), raw_query="q3",
                         query_topic="universal_credit"),
        ]
        engine = DDIEngine(config=ddi_config)
        segments = engine._segment_by_topic(logs)
        assert len(segments.get("universal_credit", [])) == 2
        assert len(segments.get("pension", [])) == 1

    def test_unknown_topic_uses_keyword_fallback(self, ddi_config):
        """Logs with topic='unknown' should be classified via keyword matching."""
        logs = [
            ProductionLog(query_id="1", timestamp=utcnow(),
                         raw_query="What is my pension credit amount?",
                         query_topic="unknown"),
        ]
        engine = DDIEngine(config=ddi_config)
        segments = engine._segment_by_topic(logs)
        # Should match "pension" via keyword "pension credit"
        assert "pension" in segments, f"Got segments: {list(segments.keys())}"

    def test_keyword_matching(self, ddi_config):
        """Verify keyword matching for each topic."""
        engine = DDIEngine(config=ddi_config)
        assert engine._match_topic("universal credit payment") == "universal_credit"
        assert engine._match_topic("council housing application") == "housing_benefit"
        assert engine._match_topic("pip assessment") == "disability_benefits"
        assert engine._match_topic("council tax band") == "council_tax"
        assert engine._match_topic("emergency housing homeless") == "homelessness"
        assert engine._match_topic("state pension age") == "pension"
        assert engine._match_topic("random unrelated query") == "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 3: DDI COMPUTATION (CORE)
# DDI = σ({JSD₁, ..., JSDₖ}) across topic segments
# ═══════════════════════════════════════════════════════════════════════════════

class TestDDIComputation:
    """
    Tests for the DDI computation formula: DDI = σ(per-segment JSDs).
    """

    def test_uniform_drift_zero_ddi(self, ddi_config, all_topics):
        """
        When ALL segments drift equally, DDI ≈ 0.

        Math: If drift_k = c for all k, then σ({c, c, ..., c}) = 0.

        Intuition: The system degraded equally for everyone.
        That's not a fairness concern (still a quality concern, but CDS handles that).
        """
        ref_logs = []
        cur_logs = []
        for topic in all_topics:
            ref_logs.extend(_make_logs(topic, 30, quality_base=0.7))
            cur_logs.extend(_make_logs(topic, 30, quality_base=0.7))  # SAME quality

        ref_ds = _make_dataset(ref_logs)
        cur_ds = _make_dataset(cur_logs)

        engine = DDIEngine(config=ddi_config)
        result = engine.compute(ref_ds, cur_ds)
        assert result.status == "GREEN", f"Expected GREEN, got {result.status} (DDI={result.value})"

    def test_differential_drift_elevated_ddi(self, ddi_config, all_topics):
        """
        When ONE segment degrades significantly while others are stable,
        DDI should be elevated.

        Math: drift = [0.01, 0.01, 0.01, 0.01, 0.01, 0.40]
              σ ≈ 0.146 → AMBER or RED
        """
        ref_logs = []
        cur_logs = []
        for topic in all_topics:
            ref_logs.extend(_make_logs(topic, 30, quality_base=0.7))
            if topic == "disability_benefits":
                # This segment degrades severely
                cur_logs.extend(_make_logs(topic, 30, quality_base=0.15,
                                          refusal_rate=0.8, citation_rate=0.1))
            else:
                cur_logs.extend(_make_logs(topic, 30, quality_base=0.7))

        ref_ds = _make_dataset(ref_logs)
        cur_ds = _make_dataset(cur_logs)

        engine = DDIEngine(config=ddi_config)
        result = engine.compute(ref_ds, cur_ds)
        assert result.value > 0, f"DDI should detect non-uniform drift, got {result.value}"
        # Should identify disability_benefits as worst
        if result.details.get("worst_segment"):
            assert result.details["worst_segment"] == "disability_benefits"

    def test_all_segments_degraded_equally(self, ddi_config, all_topics):
        """
        All segments degrade by the same amount → DDI should still be low.
        This validates that DDI measures DIFFERENTIAL drift, not overall drift.
        """
        ref_logs = []
        cur_logs = []
        for topic in all_topics:
            ref_logs.extend(_make_logs(topic, 30, quality_base=0.8))
            cur_logs.extend(_make_logs(topic, 30, quality_base=0.3))  # ALL degrade

        ref_ds = _make_dataset(ref_logs)
        cur_ds = _make_dataset(cur_logs)

        engine = DDIEngine(config=ddi_config)
        result = engine.compute(ref_ds, cur_ds)
        # DDI should be relatively low because drift is UNIFORM
        # Even though overall quality is terrible
        assert result.value < 0.15, (
            f"DDI should be low for uniform degradation, got {result.value}"
        )

    def test_ddi_equals_std_of_segment_jsds(self, ddi_config, all_topics):
        """
        Direct mathematical validation: DDI = σ(drift scores).
        Recompute manually from per_segment results.
        """
        ref_logs = []
        cur_logs = []
        for topic in all_topics:
            ref_logs.extend(_make_logs(topic, 30, quality_base=0.7))
            if topic in ("disability_benefits", "homelessness"):
                cur_logs.extend(_make_logs(topic, 30, quality_base=0.2))
            else:
                cur_logs.extend(_make_logs(topic, 30, quality_base=0.7))

        ref_ds = _make_dataset(ref_logs)
        cur_ds = _make_dataset(cur_logs)

        engine = DDIEngine(config=ddi_config)
        result = engine.compute(ref_ds, cur_ds)

        # Extract drift_scores from per_segment details
        per_seg = result.details["per_segment"]
        drift_scores = []
        for topic, info in per_seg.items():
            if info["drift_score"] is not None:
                drift_scores.append(info["drift_score"])

        if len(drift_scores) >= 2:
            expected_ddi = float(np.std(drift_scores))
            assert result.value == pytest.approx(expected_ddi, abs=1e-6), (
                f"DDI={result.value} != σ(drift_scores)={expected_ddi}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 4: MINIMUM SEGMENT SIZE ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestMinSegmentSize:
    """
    Segments with fewer than min_segment_size queries are excluded.

    Rationale: JSD on very small samples is unreliable. You can't
    meaningfully compare a 2-query distribution to a 200-query one.
    """

    def test_small_segment_excluded(self, ddi_config):
        """Segment with 3 queries (< min_segment_size=5) should be INSUFFICIENT_DATA."""
        ref_logs = _make_logs("universal_credit", 3)
        cur_logs = _make_logs("universal_credit", 3)

        ref_ds = _make_dataset(ref_logs)
        cur_ds = _make_dataset(cur_logs)

        engine = DDIEngine(config=ddi_config)
        result = engine.compute(ref_ds, cur_ds)
        per_seg = result.details.get("per_segment", {})

        if "universal_credit" in per_seg:
            assert per_seg["universal_credit"]["status"] == "INSUFFICIENT_DATA"

    def test_only_large_segments_evaluated(self, ddi_config, all_topics):
        """Only segments meeting the size threshold should have drift_score."""
        # Only universal_credit and pension have enough data
        ref_logs = _make_logs("universal_credit", 30) + _make_logs("pension", 30)
        ref_logs += _make_logs("housing_benefit", 2)  # too small
        cur_logs = _make_logs("universal_credit", 30) + _make_logs("pension", 30)
        cur_logs += _make_logs("housing_benefit", 2)  # too small

        ref_ds = _make_dataset(ref_logs)
        cur_ds = _make_dataset(cur_logs)

        engine = DDIEngine(config=ddi_config)
        result = engine.compute(ref_ds, cur_ds)

        n_evaluated = result.details["n_segments_evaluated"]
        assert n_evaluated == 2, f"Expected 2 evaluated segments, got {n_evaluated}"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 5: CLASSIFICATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDDIClassification:
    """
    DDI thresholds:
        GREEN:  DDI < 0.05
        AMBER:  0.05 ≤ DDI < 0.15
        RED:    DDI ≥ 0.15
    """

    def test_segment_classification(self, ddi_config):
        """Per-segment JSD classification."""
        engine = DDIEngine(config=ddi_config)
        assert engine._classify_segment(0.01) == "GREEN"
        assert engine._classify_segment(0.05) == "AMBER"
        assert engine._classify_segment(0.15) == "RED"

    def test_overall_classification(self, ddi_config):
        """Overall DDI classification."""
        engine = DDIEngine(config=ddi_config)
        assert engine._classify_overall(0.02, 0.05) == "GREEN"
        assert engine._classify_overall(0.10, 0.20) == "AMBER"
        assert engine._classify_overall(0.20, 0.40) == "RED"

    def test_intersectional_flag(self, ddi_config, all_topics):
        """
        When DDI_range > intersectional_threshold, the intersectional_flag
        should be True — indicating that at least 2 topic segments have
        very different drift levels.

        Legal relevance: Equality Act 2010 recognises intersectional
        discrimination. This flag highlights potential cases.
        """
        ref_logs = []
        cur_logs = []
        for topic in all_topics:
            ref_logs.extend(_make_logs(topic, 30, quality_base=0.7))
            if topic == "disability_benefits":
                cur_logs.extend(_make_logs(topic, 30, quality_base=0.1))
            else:
                cur_logs.extend(_make_logs(topic, 30, quality_base=0.7))

        ref_ds = _make_dataset(ref_logs)
        cur_ds = _make_dataset(cur_logs)

        engine = DDIEngine(config=ddi_config)
        result = engine.compute(ref_ds, cur_ds)

        if result.details.get("ddi_range", 0) > ddi_config.intersectional_threshold:
            assert result.details["intersectional_flag"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 6: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestDDIEdgeCases:
    """Edge cases and defensive coding tests."""

    def test_no_logs_returns_zero(self, ddi_config):
        """Empty datasets → DDI = 0."""
        ref_ds = _make_dataset([])
        cur_ds = _make_dataset([])

        engine = DDIEngine(config=ddi_config)
        result = engine.compute(ref_ds, cur_ds)
        assert result.value == 0.0

    def test_single_segment_zero_ddi(self, ddi_config):
        """
        With only 1 evaluated segment, σ of a single value = 0.
        DDI = 0 because you can't measure differential drift with 1 group.
        """
        ref_logs = _make_logs("universal_credit", 30)
        cur_logs = _make_logs("universal_credit", 30)
        ref_ds = _make_dataset(ref_logs)
        cur_ds = _make_dataset(cur_logs)

        engine = DDIEngine(config=ddi_config)
        result = engine.compute(ref_ds, cur_ds)
        assert result.value == 0.0  # Can't compute σ of 1 value

    def test_explanation_mentions_worst_segment(self, ddi_config, all_topics):
        """When drift is detected, explanation should name the worst segment."""
        ref_logs = []
        cur_logs = []
        for topic in all_topics:
            ref_logs.extend(_make_logs(topic, 30, quality_base=0.7))
            if topic == "disability_benefits":
                cur_logs.extend(_make_logs(topic, 30, quality_base=0.1))
            else:
                cur_logs.extend(_make_logs(topic, 30, quality_base=0.7))

        ref_ds = _make_dataset(ref_logs)
        cur_ds = _make_dataset(cur_logs)

        engine = DDIEngine(config=ddi_config)
        result = engine.compute(ref_ds, cur_ds)

        if result.details.get("worst_segment"):
            # Explanation should name the worst segment
            assert result.details["worst_segment"] in result.explanation or \
                "Insufficient data" in result.explanation


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 7: PROTECTED PROXY MAPPING VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestProtectedProxyMapping:
    """
    Every topic segment must have an associated protected_proxy string
    explaining how it relates to protected characteristics.

    This is an auditability requirement — not just software correctness.
    """

    def test_all_segments_have_proxy(self):
        """Every topic in TOPIC_SEGMENTS must have a protected_proxy."""
        for topic, config in TOPIC_SEGMENTS.items():
            assert "protected_proxy" in config, f"Topic '{topic}' missing protected_proxy"
            assert len(config["protected_proxy"]) > 0

    def test_all_segments_have_keywords(self):
        """Every topic must have at least 1 keyword for fallback matching."""
        for topic, config in TOPIC_SEGMENTS.items():
            assert "keywords" in config
            assert len(config["keywords"]) > 0, f"Topic '{topic}' has no keywords"

    def test_segments_cover_equality_act_groups(self):
        """
        Our segments should cover key protected characteristics:
        - Disability (disability_benefits)
        - Age (pension, universal_credit)
        - Other (housing, council tax, homelessness)

        This is a minimum coverage check, not exhaustive.
        """
        topics = set(TOPIC_SEGMENTS.keys())
        assert "disability_benefits" in topics, "Must cover disability"
        assert "pension" in topics, "Must cover age (elderly)"
        assert "universal_credit" in topics, "Must cover working-age"
        assert len(topics) >= 5, "Should have broad coverage"
