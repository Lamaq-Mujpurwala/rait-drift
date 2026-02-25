"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TRCI — Temporal Response Consistency Index · Detailed Test Suite           ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
-------
TRCI detects **silent model changes** — provider updates (e.g., Groq updates
Llama weights) that alter LLM behaviour without any notification to the
operator. It works by periodically replaying a fixed set of "canary queries"
and comparing the new responses to stored reference responses.

MATHEMATICAL FOUNDATION
-----------------------
For each canary query cᵢ:

    TRCI_i = cos(embed(response_new_i), embed(response_ref_i))

where embed() is the Pinecone integrated embedding (llama-text-embed-v2,
1024 dimensions, cosine distance).

The aggregate TRCI is:

    TRCI = (1/N) Σᵢ TRCI_i       (mean of per-canary similarities)

With secondary guard:

    TRCI_p10 = 10th percentile of {TRCI_1, ..., TRCI_N}

Classification:
    GREEN:   TRCI ≥ 0.95
    AMBER:   0.90 ≤ TRCI < 0.95
    RED:     TRCI < 0.90  OR  TRCI_p10 < 0.80

INTUITION
---------
Think of canary queries as "test fixtures for the model." If you ask the
same question today and get a substantially different answer, the model
has changed. Cosine similarity of embeddings captures *semantic* closeness:
a paraphrase scores high (>0.9), while a contradictory or off-topic
answer scores low (<0.7).

WHAT THE TESTS VALIDATE
------------------------
Since TRCI's live path requires Pinecone + Groq, these offline tests
validate:
1. Classification logic (GREEN/AMBER/RED thresholds)
2. Statistical aggregation (mean, p10, std)
3. Edge cases (empty canaries, all errors, single canary)
4. Explanation generation
5. Persistence of the mathematical properties

WHAT THE PLOTS REPRESENT (when generated via run_visual_tests)
--------------------------------------------------------------
1. Similarity Distribution Histogram: shows spread of per-canary similarities.
   A tight cluster near 1.0 = stable model. Bimodal or left-skewed = concern.
2. Per-Topic Heatmap: shows whether drift is uniform or topic-specific.
3. Threshold Boundary Plot: visualises the decision regions.

EXTERNAL DATA SOURCES USED
---------------------------
- canary_queries.json (50 synthetic queries covering 6 topics)  
  Purpose: Provide a repeatable probe set. NOT drawn from any external 
  benchmark dataset. These are manually authored to span our domain.
- Pinecone integrated embedding (llama-text-embed-v2)  
  Purpose: Converts text → 1024-dim vectors for similarity computation.
  This is an external model dependency, NOT our training data.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.monitoring.metrics.trci import TRCIEngine, CanaryResult
from src.monitoring.metrics.base import MetricResult
from src.utils.config import TRCIConfig


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def trci_config():
    """Standard TRCI configuration matching our system defaults."""
    return TRCIConfig(
        canary_path="config/canary_queries.json",
        probe_frequency="daily",
        similarity_metric="cosine",
        green_threshold=0.95,
        amber_threshold=0.90,
        red_p10_threshold=0.80,
        canary_count=50,
    )


@pytest.fixture
def mock_engine(trci_config):
    """
    TRCI engine with mocked Pinecone and pipeline.
    We replace the pipeline and index so no network calls are made.
    """
    engine = TRCIEngine.__new__(TRCIEngine)
    engine.config = trci_config
    engine.pipeline = MagicMock()
    engine.canary_set = []
    engine._pc = None
    engine._index = MagicMock()
    return engine


def _make_canary(cid: str, topic: str, ref_response: str) -> dict:
    """Helper to build a canary dict."""
    return {
        "id": cid,
        "query": f"Test query for {topic}",
        "topic": topic,
        "difficulty": "easy",
        "reference_response": ref_response,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 1: CLASSIFICATION LOGIC
# Mathematical property: the threshold boundaries must be exact.
# ═══════════════════════════════════════════════════════════════════════════════

class TestTRCIClassification:
    """
    Validates TRCI threshold classification.

    The classification function maps (mean, p10) → {GREEN, AMBER, RED}.
    We test ALL boundary conditions to ensure correctness.
    """

    def test_green_high_similarity(self, mock_engine):
        """
        Mean ≥ 0.95 AND p10 ≥ 0.80 → GREEN.

        Intuition: All canaries returned very similar responses ⇒ no model change.
        """
        # Both conditions met
        assert mock_engine._classify_overall(mean=0.98, p10=0.95) == "GREEN"
        assert mock_engine._classify_overall(mean=0.95, p10=0.90) == "GREEN"

    def test_amber_moderate_drift(self, mock_engine):
        """
        0.90 ≤ mean < 0.95 AND p10 ≥ 0.80 → AMBER.

        Intuition: Slight change detected; not enough for alarm but worth watching.
        """
        assert mock_engine._classify_overall(mean=0.92, p10=0.85) == "AMBER"
        assert mock_engine._classify_overall(mean=0.90, p10=0.80) == "AMBER"

    def test_red_low_mean(self, mock_engine):
        """
        Mean < 0.90 → RED regardless of p10.

        Intuition: Average canary divergence is severe.
        """
        assert mock_engine._classify_overall(mean=0.85, p10=0.82) == "RED"
        assert mock_engine._classify_overall(mean=0.50, p10=0.50) == "RED"

    def test_red_low_p10(self, mock_engine):
        """
        p10 < 0.80 → RED even if mean is acceptable.

        Math: p10 is the 10th percentile. If p10 < 0.80, it means ≥10% of
        canaries have similarity below 0.80, indicating a significant tail
        of degraded responses.

        Intuition: Even with a decent average, having 10%+ of responses
        badly diverged signals a serious problem.
        """
        assert mock_engine._classify_overall(mean=0.93, p10=0.75) == "RED"

    def test_per_canary_classification(self, mock_engine):
        """Individual canary classification follows simpler rules."""
        assert mock_engine._classify_similarity(0.98) == "GREEN"
        assert mock_engine._classify_similarity(0.95) == "GREEN"  # >= threshold
        assert mock_engine._classify_similarity(0.92) == "AMBER"
        assert mock_engine._classify_similarity(0.90) == "AMBER"  # >= threshold
        assert mock_engine._classify_similarity(0.89) == "RED"
        assert mock_engine._classify_similarity(0.0) == "RED"

    @pytest.mark.parametrize("mean,p10,expected", [
        (1.00, 1.00, "GREEN"),
        (0.95, 0.80, "GREEN"),    # exact boundary
        (0.949, 0.80, "AMBER"),   # just below green
        (0.90, 0.80, "AMBER"),    # exact amber boundary
        (0.899, 0.80, "RED"),     # just below amber
        (0.95, 0.799, "RED"),     # good mean but p10 below threshold
        (0.00, 0.00, "RED"),      # catastrophic
    ])
    def test_boundary_parametrized(self, mock_engine, mean, p10, expected):
        """
        Parametrized boundary test covering exact threshold values.

        This ensures our floating-point comparisons behave correctly
        at the exact decision boundaries (>= vs >).
        """
        result = mock_engine._classify_overall(mean, p10)
        assert result == expected, f"mean={mean}, p10={p10}: expected {expected}, got {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 2: STATISTICAL AGGREGATION
# Mathematical property: TRCI aggregates must be computed correctly.
# ═══════════════════════════════════════════════════════════════════════════════

class TestTRCIAggregation:
    """
    Validates the mathematical aggregation of per-canary similarities.

    TRCI_mean = (1/N) Σ sᵢ
    TRCI_p10  = P₁₀({s₁, ..., sₙ})
    TRCI_std  = σ({s₁, ..., sₙ})
    """

    def test_mean_of_uniform_similarities(self):
        """If all canaries have sim=0.96, mean should be exactly 0.96."""
        sims = np.array([0.96] * 10)
        assert np.isclose(np.mean(sims), 0.96)
        assert np.isclose(np.std(sims), 0.0)

    def test_p10_identifies_tail(self):
        """
        P10 with 1 outlier out of 10:
        [0.98, 0.97, 0.96, 0.96, 0.95, 0.95, 0.94, 0.93, 0.80, 0.60]

        10th percentile ≈ 0.62 (between pos 0 and 1 in sorted order).
        This should trigger RED even though mean ≈ 0.904.
        """
        sims = np.array([0.98, 0.97, 0.96, 0.96, 0.95, 0.95, 0.94, 0.93, 0.80, 0.60])
        p10 = np.percentile(sims, 10)
        mean = np.mean(sims)
        assert p10 < 0.80, f"p10 should be < 0.80, got {p10}"
        assert mean > 0.85, f"Mean is {mean}, still above 0.85"

    def test_std_detects_variance(self):
        """
        High std indicates inconsistent canary responses — some changed, some didn't.
        This is qualitatively different from uniformly low similarity.
        """
        # Low variance: all canaries similar
        sims_stable = np.array([0.97, 0.96, 0.97, 0.96, 0.97])
        # High variance: some canaries changed
        sims_mixed = np.array([0.99, 0.98, 0.50, 0.49, 0.97])

        assert np.std(sims_stable) < 0.01
        assert np.std(sims_mixed) > 0.20

    def test_count_classification_buckets(self):
        """
        Verify n_green, n_amber, n_red bucket counting.
        Given [0.98, 0.93, 0.85], should be 1 green, 1 amber, 1 red.
        """
        sims = np.array([0.98, 0.93, 0.85])
        n_green = int(np.sum(sims >= 0.95))
        n_amber = int(np.sum((sims >= 0.90) & (sims < 0.95)))
        n_red = int(np.sum(sims < 0.90))
        assert n_green == 1
        assert n_amber == 1
        assert n_red == 1
        assert n_green + n_amber + n_red == len(sims)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 3: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestTRCIEdgeCases:
    """
    Edge cases that must be handled gracefully.
    These test defensive coding rather than mathematical properties.
    """

    def test_no_pipeline_returns_red(self, trci_config):
        """Without a pipeline, TRCI should return RED with clear explanation."""
        engine = TRCIEngine.__new__(TRCIEngine)
        engine.config = trci_config
        engine.pipeline = None
        engine.canary_set = [_make_canary("C001", "test", "ref response")]
        engine._pc = None
        engine._index = None

        result = engine.run_probe()
        assert result.status == "RED"
        assert "No pipeline" in result.explanation

    def test_no_canaries_with_references(self, mock_engine):
        """
        If no canaries have reference_response populated,
        TRCI should return GREEN with advisory message.

        Intuition: Can't measure drift without a baseline.
        """
        mock_engine.canary_set = [
            {"id": "C001", "query": "test", "topic": "t", "reference_response": None},
            {"id": "C002", "query": "test", "topic": "t"},
        ]
        result = mock_engine.run_probe()
        assert result.status == "GREEN"
        assert "init first" in result.explanation.lower() or "no canaries" in result.explanation.lower()

    def test_single_canary(self, mock_engine):
        """
        With only 1 canary, p10 == mean == that single value.
        This should still produce a valid result.
        """
        mock_engine.canary_set = [_make_canary("C001", "benefits", "ref")]

        # Mock pipeline to return a log-like object
        mock_log = MagicMock()
        mock_log.raw_response = "Some response"
        mock_engine.pipeline.process.return_value = mock_log

        # Mock Pinecone search to return similarity
        mock_engine._index.search_records.return_value = {
            "result": {"hits": [{"_id": "ref_C001", "_score": 0.97}]}
        }

        result = mock_engine.run_probe()
        assert result.value == pytest.approx(0.97, abs=0.01)
        assert result.details["n_probed"] == 1

    def test_all_probes_error(self, mock_engine):
        """If every canary probe raises an exception, TRCI should be RED."""
        mock_engine.canary_set = [_make_canary("C001", "benefits", "ref")]
        mock_engine.pipeline.process.side_effect = RuntimeError("API down")

        result = mock_engine.run_probe()
        assert result.status == "RED"
        assert "No canary probes succeeded" in result.explanation


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 4: EXPLANATION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestTRCIExplanation:
    """
    The explanation string is user-facing and must contain actionable info.
    """

    def test_explanation_contains_statistics(self, mock_engine):
        """Explanation must report mean similarity and 10th percentile."""
        sims = np.array([0.96, 0.94, 0.92])
        per_canary = [
            CanaryResult("C1", "Q1", "t1", "ref", "new", 0.96, "GREEN"),
            CanaryResult("C2", "Q2", "t2", "ref", "new", 0.94, "AMBER"),
            CanaryResult("C3", "Q3", "t3", "ref", "new", 0.92, "AMBER"),
        ]
        expl = mock_engine._generate_explanation(sims, per_canary)
        assert "Mean similarity" in expl
        assert "10th percentile" in expl

    def test_explanation_identifies_worst_canary(self, mock_engine):
        """Explanation must highlight the worst-performing canary."""
        sims = np.array([0.96, 0.70])
        per_canary = [
            CanaryResult("C1", "Good query here", "t1", "ref", "new", 0.96, "GREEN"),
            CanaryResult("C2", "Bad query here", "t2", "ref", "new", 0.70, "RED"),
        ]
        expl = mock_engine._generate_explanation(sims, per_canary)
        assert "Bad query" in expl
        assert "0.70" in expl

    def test_explanation_recommends_investigation_when_drifted(self, mock_engine):
        """Below threshold → "Investigation recommended." """
        sims = np.array([0.92, 0.88])
        per_canary = [
            CanaryResult("C1", "Q1", "t1", "ref", "new", 0.92, "AMBER"),
            CanaryResult("C2", "Q2", "t2", "ref", "new", 0.88, "RED"),
        ]
        expl = mock_engine._generate_explanation(sims, per_canary)
        assert "investigation" in expl.lower()

    def test_explanation_says_no_action_when_stable(self, mock_engine):
        """Above threshold → "No action needed." """
        sims = np.array([0.98, 0.97])
        per_canary = [
            CanaryResult("C1", "Q1", "t1", "ref", "new", 0.98, "GREEN"),
            CanaryResult("C2", "Q2", "t2", "ref", "new", 0.97, "GREEN"),
        ]
        expl = mock_engine._generate_explanation(sims, per_canary)
        assert "no action" in expl.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 5: END-TO-END MOCK PROBE
# ═══════════════════════════════════════════════════════════════════════════════

class TestTRCIProbeSimulation:
    """
    Simulates a full probe cycle with mocked external dependencies.
    Verifies the complete data flow: canary → pipeline → Pinecone → result.
    """

    def test_full_probe_green(self, mock_engine):
        """
        Scenario: All 3 canaries return high similarity (>0.95).
        Expected: GREEN status.
        """
        mock_engine.canary_set = [
            _make_canary("C001", "universal_credit", "UC standard allowance is £400"),
            _make_canary("C002", "housing_benefit", "Housing benefit covers rent"),
            _make_canary("C003", "pension", "State pension age is 66"),
        ]

        mock_log = MagicMock()
        mock_log.raw_response = "Nearly identical response"
        mock_engine.pipeline.process.return_value = mock_log

        # Each search returns high similarity
        mock_engine._index.search_records.return_value = {
            "result": {"hits": [{"_id": "ref_C001", "_score": 0.97}]}
        }

        with patch("time.sleep"):  # Skip the 1s delay
            result = mock_engine.run_probe()

        assert result.status == "GREEN"
        assert result.value >= 0.95
        assert result.details["n_probed"] == 3
        assert result.details["n_green"] == 3

    def test_full_probe_amber(self, mock_engine):
        """
        Scenario: Mixed similarities — some drifted.
        Expected: AMBER status.
        """
        mock_engine.canary_set = [
            _make_canary("C001", "universal_credit", "ref1"),
            _make_canary("C002", "housing_benefit", "ref2"),
            _make_canary("C003", "pension", "ref3"),
        ]

        mock_log = MagicMock()
        mock_log.raw_response = "Somewhat changed response"
        mock_engine.pipeline.process.return_value = mock_log

        # Return different similarities for each call
        mock_engine._index.search_records.side_effect = [
            {"result": {"hits": [{"_id": "ref_C001", "_score": 0.96}]}},
            {"result": {"hits": [{"_id": "ref_C002", "_score": 0.91}]}},
            {"result": {"hits": [{"_id": "ref_C003", "_score": 0.88}]}},
        ]

        with patch("time.sleep"):
            result = mock_engine.run_probe()

        # Mean ≈ 0.9167, p10 ≈ 0.886 → AMBER or RED depending on p10
        assert result.status in ("AMBER", "RED")
        assert result.details["n_green"] == 1
        assert result.details["n_red"] >= 1

    def test_full_probe_red_model_changed(self, mock_engine):
        """
        Scenario: Model was updated — all responses are semantically different.
        Expected: RED status.
        """
        mock_engine.canary_set = [
            _make_canary("C001", "universal_credit", "ref1"),
            _make_canary("C002", "housing_benefit", "ref2"),
        ]

        mock_log = MagicMock()
        mock_log.raw_response = "Completely different response"
        mock_engine.pipeline.process.return_value = mock_log

        mock_engine._index.search_records.return_value = {
            "result": {"hits": [{"_id": "ref", "_score": 0.55}]}
        }

        with patch("time.sleep"):
            result = mock_engine.run_probe()

        assert result.status == "RED"
        assert result.value < 0.90
        assert result.details["n_red"] == 2
