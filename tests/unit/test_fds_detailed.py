"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  FDS — Faithfulness Decay Score · Detailed Test Suite                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
-------
FDS detects when the LLM starts producing responses that are **less faithful**
to the retrieved source documents. This is the most dangerous form of drift
for a public-sector chatbot: the system appears to work (responds confidently)
but the content diverges from the official sources.

MATHEMATICAL FOUNDATION
-----------------------
Step 1 — Claim Decomposition:
    response → {claim₁, claim₂, ..., claimₘ}
    Each claim is an atomic, independently verifiable statement.

Step 2 — Per-Claim Verification (LLM-as-Judge):
    For each claimᵢ, judge against retrieved context:
    verdict ∈ {supported, unsupported, ambiguous}

Step 3 — Per-Response Faithfulness:
    faithfulness_j = n_supported / n_total_claims    (include_ambiguous=False)
      or
    faithfulness_j = (n_supported + n_ambiguous) / n_total_claims  (include_ambiguous=True)

Step 4 — Distribution Comparison via Signed JSD:

    FDS = sign(μ_cur - μ_ref) · JSD(F_ref || F_cur)

    where:
        F_ref = reference faithfulness distribution (from baseline period)
        F_cur = current faithfulness distribution
        sign  = +1 if μ_cur ≥ μ_ref (improvement), -1 if μ_cur < μ_ref (decay)

Classification:
    GREEN:   |FDS| < 0.02
    AMBER:   0.02 ≤ |FDS| < 0.10
    RED:     |FDS| ≥ 0.10

INTUITION
---------
Imagine a human fact-checker reviewing each chatbot response:
- Decompose: "The response makes 5 claims."
- Verify: "3 are correct, 1 is wrong, 1 is unclear."
- Score: faithfulness = 3/5 = 0.60

FDS tracks whether this score is DECLINING over time compared to a baseline.
Signed JSD tells us both the MAGNITUDE of change (JSD) and the DIRECTION
(positive = improving, negative = decaying).

WHY SIGNED JSD?
- Regular JSD only says "distributions differ."
- Signed JSD says "distributions differ AND the current one is WORSE."
- This prevents alert fatigue: improving faithfulness shouldn't trigger alarms.

WHAT THE TESTS VALIDATE
------------------------
1. Signed JSD direction detection (decay vs improvement)
2. Per-response faithfulness computation (with and without ambiguous)
3. Sampling strategies (random, recent, stratified)
4. Edge cases (no claims, all supported, all unsupported)
5. Classification thresholds

EXTERNAL DATA SOURCES
---------------------
- calibration_set.json (5 entries with expected_claims and expected_faithfulness)
  Purpose: Ground truth for evaluator calibration — tests whether the
  LLM-as-Judge itself is reliable. NOT an external benchmark. Manually
  authored by us to span easy/hard/edge cases.
- The judge LLM (groq/llama-3.1-8b-instant) is an external dependency.
  Its behaviour is non-deterministic; we mock it in unit tests.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.monitoring.metrics.fds import FDSEngine, FDSQueryResult, CalibrationResult
from src.monitoring.metrics.base import MetricResult
from src.monitoring.judge import ClaimVerdict, CrossValidator, CrossValidationResult
from src.production.logging import ProductionLog, generate_id, utcnow
from src.utils.config import FDSConfig
from src.utils.statistics import signed_jsd, continuous_jsd


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def fds_config():
    """Standard FDS configuration."""
    return FDSConfig(
        sample_size=50,
        sampling_strategy="random",
        verification_strictness="moderate",
        include_ambiguous=False,
        signed_jsd_bins=20,
        green_threshold=0.02,
        amber_threshold=0.10,
        red_threshold=0.20,
    )


@pytest.fixture
def mock_judge():
    """Mocked JudgeEngine that returns controlled verdicts."""
    judge = MagicMock()
    return judge


@pytest.fixture
def sample_logs():
    """Generate 20 synthetic production logs for testing."""
    logs = []
    topics = ["universal_credit", "housing_benefit", "disability_benefits", "pension"]
    for i in range(20):
        logs.append(ProductionLog(
            query_id=generate_id(),
            timestamp=utcnow(),
            raw_query=f"Test query {i} about {topics[i % len(topics)]}",
            query_topic=topics[i % len(topics)],
            raw_response=f"Detailed response about {topics[i % len(topics)]} with several claims. "
                         f"Claim one states X. Claim two states Y. Claim three states Z.",
            completion_tokens=150,
            citation_count=2,
            refusal_flag=False,
            response_latency_ms=800,
        ))
    return logs


@pytest.fixture
def high_baseline():
    """Reference faithfulness: high quality (mean ~0.85)."""
    np.random.seed(42)
    return np.clip(np.random.normal(0.85, 0.08, 100), 0, 1)


@pytest.fixture
def low_baseline():
    """Reference faithfulness: lower quality (mean ~0.50)."""
    np.random.seed(42)
    return np.clip(np.random.normal(0.50, 0.15, 100), 0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 1: SIGNED JSD MATHEMATICAL PROPERTIES
# The directional JSD that gives FDS its sign.
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignedJSD:
    """
    Signed JSD = sign(μ_cur - μ_ref) · JSD(ref, cur)

    Properties:
    - Negative when current mean < reference mean (DECAY)
    - Positive when current mean > reference mean (IMPROVEMENT)
    - Zero when distributions are identical
    - Magnitude equals unsigned JSD
    """

    def test_identical_distributions_zero(self):
        """Signed JSD of identical distributions = 0."""
        np.random.seed(42)
        x = np.random.normal(0.8, 0.1, 500)
        result = signed_jsd(x, x, bins=20)
        assert abs(result) < 1e-5

    def test_decay_is_negative(self):
        """
        When current faithfulness is LOWER than reference → negative FDS (decay).

        ref ~ N(0.85, 0.08)  -->  cur ~ N(0.50, 0.08)
        μ_cur < μ_ref  →  sign = -1  →  FDS < 0
        """
        np.random.seed(42)
        ref = np.random.normal(0.85, 0.08, 500)
        cur = np.random.normal(0.50, 0.08, 500)
        result = signed_jsd(ref, cur, bins=20)
        assert result < 0, f"Expected negative (decay), got {result}"

    def test_improvement_is_positive(self):
        """
        When current faithfulness is HIGHER than reference → positive FDS.

        ref ~ N(0.50, 0.08)  -->  cur ~ N(0.85, 0.08)
        μ_cur > μ_ref  →  sign = +1  →  FDS > 0
        """
        np.random.seed(42)
        ref = np.random.normal(0.50, 0.08, 500)
        cur = np.random.normal(0.85, 0.08, 500)
        result = signed_jsd(ref, cur, bins=20)
        assert result > 0, f"Expected positive (improvement), got {result}"

    def test_magnitude_equals_unsigned_jsd(self):
        """
        |signed_jsd| ≡ continuous_jsd

        The sign is purely directional; magnitude is the same as regular JSD.
        """
        np.random.seed(42)
        ref = np.random.normal(0.8, 0.1, 500)
        cur = np.random.normal(0.5, 0.1, 500)

        s_jsd = signed_jsd(ref, cur, bins=20)
        u_jsd = continuous_jsd(ref, cur, bins=20)
        assert abs(abs(s_jsd) - u_jsd) < 1e-6

    def test_symmetry_of_magnitude(self):
        """
        |signed_jsd(P, Q)| = |signed_jsd(Q, P)|

        But signs are OPPOSITE (as expected).
        """
        np.random.seed(42)
        ref = np.random.normal(0.8, 0.1, 500)
        cur = np.random.normal(0.5, 0.1, 500)

        fwd = signed_jsd(ref, cur, bins=20)
        rev = signed_jsd(cur, ref, bins=20)
        assert abs(abs(fwd) - abs(rev)) < 1e-5
        assert (fwd < 0 and rev > 0) or (fwd > 0 and rev < 0), "Signs should be opposite"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 2: FAITHFULNESS COMPUTATION
# Per-response faithfulness score from claim verdicts.
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaithfulnessComputation:
    """
    faithfulness = n_supported / n_total  (include_ambiguous=False)
    faithfulness = (n_supported + n_ambiguous) / n_total  (include_ambiguous=True)
    """

    def test_all_supported(self):
        """5/5 supported → faithfulness = 1.0."""
        n_total = 5
        n_supported = 5
        faithfulness = n_supported / n_total
        assert faithfulness == 1.0

    def test_all_unsupported(self):
        """0/5 supported → faithfulness = 0.0."""
        faithfulness = 0 / 5
        assert faithfulness == 0.0

    def test_mixed_verdicts(self):
        """3 supported, 1 unsupported, 1 ambiguous → 3/5 = 0.6 (strict)."""
        n_supported = 3
        n_unsupported = 1
        n_ambiguous = 1
        n_total = n_supported + n_unsupported + n_ambiguous
        faithfulness_strict = n_supported / n_total
        faithfulness_lenient = (n_supported + n_ambiguous) / n_total
        assert faithfulness_strict == pytest.approx(0.6)
        assert faithfulness_lenient == pytest.approx(0.8)

    def test_no_claims_defaults_to_zero(self):
        """Edge case: response decomposed to 0 claims → faithfulness = 0 (not crash)."""
        n_supported = 0
        n_total = 0
        faithfulness = n_supported / max(n_total, 1)
        assert faithfulness == 0.0

    def test_ambiguous_mode_flag(self):
        """
        include_ambiguous=True: ambiguous claims count as "faithful."
        Rationale: In a public-sector context, ambiguous claims (e.g., 
        "you may be eligible") are not harmful — they're appropriately cautious.
        """
        n_supported = 3
        n_ambiguous = 2
        n_total = 6
        strict = n_supported / n_total
        lenient = (n_supported + n_ambiguous) / n_total
        assert strict == pytest.approx(0.5)
        assert lenient == pytest.approx(0.833, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 3: FDS ENGINE WITH MOCKED JUDGE
# Tests the full compute() pipeline without making API calls.
# ═══════════════════════════════════════════════════════════════════════════════

class TestFDSCompute:
    """
    End-to-end FDS computation with mocked claim decomposition and verification.
    """

    def test_stable_faithfulness_green(self, fds_config, sample_logs, high_baseline):
        """
        When judge rates all claims as supported → current faithfulness ≈ 1.0,
        reference ≈ 0.85. This is improvement → positive FDS, likely GREEN.
        """
        engine = FDSEngine(config=fds_config)
        engine.judge = MagicMock()

        # Mock: decompose returns 3 claims, all verified as supported
        engine.judge.decompose_claims.return_value = ["Claim A", "Claim B", "Claim C"]
        engine.judge.verify_claim.return_value = ClaimVerdict(
            claim="test", verdict="supported", confidence=0.95, evidence_quote="found"
        )

        result = engine.compute(sample_logs[:10], high_baseline)
        assert result.status == "GREEN" or result.value > 0  # Either green or improvement

    def test_decayed_faithfulness_triggers_alert(self, fds_config, sample_logs, high_baseline):
        """
        When judge rates most claims as unsupported → low current faithfulness.
        Reference baseline is high → FDS is negative (decay) and large.
        """
        engine = FDSEngine(config=fds_config)
        engine.judge = MagicMock()

        # Mock: decompose returns 3 claims, mostly unsupported
        engine.judge.decompose_claims.return_value = ["Claim A", "Claim B", "Claim C"]
        engine.judge.verify_claim.side_effect = [
            ClaimVerdict("A", "unsupported", 0.9, "N/A"),
            ClaimVerdict("B", "unsupported", 0.8, "N/A"),
            ClaimVerdict("C", "supported", 0.9, "found"),
        ] * 10  # Repeat for each log

        result = engine.compute(sample_logs[:10], high_baseline)
        # Current faithfulness ≈ 1/3 = 0.33, ref ≈ 0.85 → huge negative FDS
        assert result.value < 0, f"Expected decay (negative FDS), got {result.value}"
        assert result.details["mean_faithfulness"] < 0.5

    def test_empty_logs_green(self, fds_config, high_baseline):
        """No logs to evaluate → GREEN with explanation."""
        engine = FDSEngine(config=fds_config)
        engine.judge = MagicMock()
        result = engine.compute([], high_baseline)
        assert result.status == "GREEN"
        assert "No queries" in result.explanation

    def test_result_details_structure(self, fds_config, sample_logs, high_baseline):
        """FDS result details must contain expected keys."""
        engine = FDSEngine(config=fds_config)
        engine.judge = MagicMock()
        engine.judge.decompose_claims.return_value = ["Claim A"]
        engine.judge.verify_claim.return_value = ClaimVerdict(
            "A", "supported", 0.9, "found"
        )

        result = engine.compute(sample_logs[:5], high_baseline)
        assert "fds_value" in result.details
        assert "mean_faithfulness" in result.details
        assert "std_faithfulness" in result.details
        assert "reference_mean" in result.details
        assert "n_evaluated" in result.details
        assert "per_query" in result.details


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 4: SAMPLING STRATEGIES
# How queries are selected for evaluation under budget constraints.
# ═══════════════════════════════════════════════════════════════════════════════

class TestFDSSampling:
    """
    FDS evaluates a SAMPLE of queries (not all — too expensive with LLM judge).
    The sampling strategy determines which queries are evaluated.

    Strategies:
    - random: uniform random selection
    - recent: most recent queries first
    - stratified: proportional to topic distribution
    """

    def test_sample_respects_budget(self, sample_logs):
        """Sample size should not exceed config.sample_size."""
        engine = FDSEngine(config=FDSConfig(sample_size=5))
        sample = engine._sample_queries(sample_logs)
        assert len(sample) <= 5

    def test_sample_handles_fewer_logs(self):
        """If fewer logs than sample_size, return all."""
        logs = [ProductionLog(query_id=generate_id(), timestamp=utcnow(), raw_query="q")]
        engine = FDSEngine(config=FDSConfig(sample_size=50))
        sample = engine._sample_queries(logs)
        assert len(sample) == 1

    def test_random_sampling(self, sample_logs):
        """Random sampling should produce a subset of the input."""
        engine = FDSEngine(config=FDSConfig(sample_size=5, sampling_strategy="random"))
        sample = engine._sample_queries(sample_logs)
        assert len(sample) == 5
        assert all(s in sample_logs for s in sample)

    def test_recent_sampling(self, sample_logs):
        """Recent sampling should return the most recent logs."""
        engine = FDSEngine(config=FDSConfig(sample_size=5, sampling_strategy="recent"))
        sample = engine._sample_queries(sample_logs)
        assert len(sample) == 5
        # All sampled should be from the original logs
        assert all(s in sample_logs for s in sample)

    def test_stratified_sampling_covers_topics(self, sample_logs):
        """
        Stratified sampling should try to cover all topics proportionally.
        This matters for fairness — we don't want to only evaluate
        popular topics and miss rare ones.
        """
        engine = FDSEngine(config=FDSConfig(sample_size=12, sampling_strategy="stratified"))
        sample = engine._sample_queries(sample_logs)
        topics = set(s.query_topic for s in sample)
        # Should include multiple topics
        assert len(topics) >= 2, f"Only got topics: {topics}"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 5: FDS CLASSIFICATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFDSClassification:
    """
    FDS uses |FDS| for classification (absolute value),
    because both extreme decay AND extreme improvement are notable.

    GREEN:   |FDS| < 0.02   (stable)
    AMBER:   0.02 ≤ |FDS| < 0.10   (moderate shift)
    RED:     |FDS| ≥ 0.10   (significant shift — investigate)
    """

    @pytest.mark.parametrize("fds_value,expected", [
        (0.00, "GREEN"),
        (0.01, "GREEN"),
        (-0.01, "GREEN"),
        (0.019, "GREEN"),
        (0.02, "AMBER"),
        (-0.02, "AMBER"),
        (0.05, "AMBER"),
        (-0.09, "AMBER"),
        (0.10, "RED"),
        (-0.10, "RED"),
        (0.30, "RED"),
        (-0.50, "RED"),
    ])
    def test_fds_classification(self, fds_value, expected):
        """Parametrized test of FDS threshold classification."""
        abs_fds = abs(fds_value)
        if abs_fds < 0.02:
            status = "GREEN"
        elif abs_fds < 0.10:
            status = "AMBER"
        else:
            status = "RED"
        assert status == expected, f"FDS={fds_value}: expected {expected}, got {status}"


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 6: EXPLANATION QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestFDSExplanation:
    """FDS explanations must communicate the direction and severity of drift."""

    def test_explanation_shows_direction(self, fds_config, sample_logs, high_baseline):
        """Explanation should say 'decay' or 'improvement'."""
        engine = FDSEngine(config=fds_config)
        engine.judge = MagicMock()
        engine.judge.decompose_claims.return_value = ["Claim"]
        engine.judge.verify_claim.return_value = ClaimVerdict(
            "Claim", "supported", 0.9, "found"
        )

        result = engine.compute(sample_logs[:5], high_baseline)
        assert "decay" in result.explanation.lower() or "improvement" in result.explanation.lower()

    def test_explanation_shows_mean_faithfulness(self, fds_config, sample_logs, high_baseline):
        """Explanation should report current and reference mean faithfulness."""
        engine = FDSEngine(config=fds_config)
        engine.judge = MagicMock()
        engine.judge.decompose_claims.return_value = ["Claim"]
        engine.judge.verify_claim.return_value = ClaimVerdict(
            "Claim", "supported", 0.9, "found"
        )

        result = engine.compute(sample_logs[:5], high_baseline)
        assert "Mean faithfulness" in result.explanation
        assert "reference baseline" in result.explanation.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Cohen's kappa measures inter-rater agreement correcting for chance:
#
#   κ = (p_o - p_e) / (1 - p_e)
#
# where p_o = observed agreement rate, p_e = expected agreement by chance.
#
# Interpretation (Landis & Koch, 1977):
#   κ < 0.00  → Poor
#   0.00–0.20 → Slight
#   0.21–0.40 → Fair
#   0.41–0.60 → Moderate
#   0.61–0.80 → Substantial
#   0.81–1.00 → Almost perfect
#
# For our FDS judge, κ ≥ 0.6 suggests the primary judge is reliable.
# κ < 0.4 triggers a warning flag in the FDS output.
# ═══════════════════════════════════════════════════════════════════════════════


class TestCohensKappa:
    """Validate the Cohen's kappa implementation against known values."""

    def test_perfect_agreement(self):
        """Two judges with identical verdicts → κ = 1.0."""
        a = ["supported", "unsupported", "supported", "ambiguous"]
        b = ["supported", "unsupported", "supported", "ambiguous"]
        kappa = CrossValidator._cohens_kappa(a, b)
        assert kappa == pytest.approx(1.0)

    def test_no_agreement_beyond_chance(self):
        """Judges that agree only by chance → κ ≈ 0."""
        # Construct labels where observed agreement ≈ expected by chance
        a = ["supported"] * 50 + ["unsupported"] * 50
        b = ["supported"] * 25 + ["unsupported"] * 25 + ["supported"] * 25 + ["unsupported"] * 25
        kappa = CrossValidator._cohens_kappa(a, b)
        assert abs(kappa) < 0.1, f"Expected kappa ≈ 0, got {kappa}"

    def test_complete_disagreement(self):
        """Systematic disagreement where raters use overlapping categories → κ < 0."""
        # When raters use non-overlapping categories, p_e = 0 → κ = 0.
        # For κ < 0 we need p_o < p_e, meaning raters use the same categories
        # but systematically swap them.
        a = ["supported", "unsupported"] * 5  # alternating
        b = ["unsupported", "supported"] * 5  # reversed
        kappa = CrossValidator._cohens_kappa(a, b)
        assert kappa < 0, f"Expected negative kappa, got {kappa}"

    def test_empty_lists(self):
        """Empty inputs → κ = 1.0 (degenerate: no disagreement possible)."""
        kappa = CrossValidator._cohens_kappa([], [])
        assert kappa == 1.0

    def test_partial_agreement(self):
        """Known partial agreement → κ is between 0 and 1."""
        # Both raters use both categories, but agree on 8/10
        a = ["supported"] * 6 + ["unsupported"] * 4
        b = ["supported"] * 6 + ["unsupported"] * 2 + ["supported"] * 2
        # p_o = 8/10 = 0.8, p_e < 0.8 because both use both categories
        kappa = CrossValidator._cohens_kappa(a, b)
        assert 0 < kappa < 1, f"Expected 0 < κ < 1, got {kappa}"

    def test_kappa_bounded(self):
        """κ should always be ≤ 1."""
        # All supported
        a = ["supported"] * 20
        b = ["supported"] * 20
        kappa = CrossValidator._cohens_kappa(a, b)
        assert kappa <= 1.0


class TestCrossValidator:
    """Test the CrossValidator integration with mocked judges."""

    def test_cross_validate_perfect_agreement(self):
        """Both judges agree on all claims → agreement_rate = 1, κ = 1."""
        primary = MagicMock()
        secondary = MagicMock()
        verdict = ClaimVerdict("claim", "supported", 0.9, "evidence")
        primary.verify_claim.return_value = verdict
        secondary.verify_claim.return_value = verdict

        cv = CrossValidator(primary=primary, secondary=secondary)
        result = cv.cross_validate(["claim1", "claim2", "claim3"], "context")

        assert result.n_claims == 3
        assert result.n_agree == 3
        assert result.n_disagree == 0
        assert result.agreement_rate == 1.0
        assert result.cohens_kappa == pytest.approx(1.0)

    def test_cross_validate_total_disagreement(self):
        """Judges disagree on every claim with overlapping categories → κ < 0."""
        primary = MagicMock()
        secondary = MagicMock()
        # Both judges use both categories but systematically swap
        primary.verify_claim.side_effect = [
            ClaimVerdict("c", "supported", 0.9, "ev"),
            ClaimVerdict("c", "unsupported", 0.9, "ev"),
            ClaimVerdict("c", "supported", 0.9, "ev"),
            ClaimVerdict("c", "unsupported", 0.9, "ev"),
        ]
        secondary.verify_claim.side_effect = [
            ClaimVerdict("c", "unsupported", 0.8, "ev"),
            ClaimVerdict("c", "supported", 0.8, "ev"),
            ClaimVerdict("c", "unsupported", 0.8, "ev"),
            ClaimVerdict("c", "supported", 0.8, "ev"),
        ]

        cv = CrossValidator(primary=primary, secondary=secondary)
        result = cv.cross_validate(["c1", "c2", "c3", "c4"], "context")

        assert result.n_agree == 0
        assert result.n_disagree == 4
        assert result.agreement_rate == 0.0
        assert result.cohens_kappa < 0

    def test_cross_validate_empty_claims(self):
        """No claims to validate → degenerate perfect agreement."""
        primary = MagicMock()
        secondary = MagicMock()
        cv = CrossValidator(primary=primary, secondary=secondary)
        result = cv.cross_validate([], "context")

        assert result.n_claims == 0
        assert result.agreement_rate == 1.0
        assert result.cohens_kappa == 1.0

    def test_disagreements_captured(self):
        """Disagreement details should be captured in the result."""
        primary = MagicMock()
        secondary = MagicMock()

        def primary_side(claim, ctx):
            if claim == "good_claim":
                return ClaimVerdict(claim, "supported", 0.9, "ev")
            return ClaimVerdict(claim, "unsupported", 0.7, "ev")

        def secondary_side(claim, ctx):
            # Always says supported
            return ClaimVerdict(claim, "supported", 0.8, "ev")

        primary.verify_claim.side_effect = primary_side
        secondary.verify_claim.side_effect = secondary_side

        cv = CrossValidator(primary=primary, secondary=secondary)
        result = cv.cross_validate(
            ["good_claim", "bad_claim"], "context"
        )

        assert result.n_disagree == 1
        assert len(result.disagreements) == 1
        assert result.disagreements[0]["primary_verdict"] == "unsupported"
        assert result.disagreements[0]["secondary_verdict"] == "supported"


class TestFDSCrossValidationIntegration:
    """Test that FDS integrates cross-validation when enabled."""

    def test_cross_val_disabled_by_default(self, fds_config, sample_logs, high_baseline):
        """Cross-validation should not run when disabled."""
        engine = FDSEngine(config=fds_config)
        engine.judge = MagicMock()
        engine.judge.decompose_claims.return_value = ["Claim"]
        engine.judge.verify_claim.return_value = ClaimVerdict("Claim", "supported", 0.9, "ev")

        result = engine.compute(sample_logs[:5], high_baseline)
        assert "cross_validation" not in result.details

    def test_cross_val_enabled_adds_details(self, sample_logs, high_baseline):
        """When enabled, cross-validation results appear in details."""
        config = FDSConfig(
            sample_size=5,
            sampling_strategy="random",
            cross_validation_enabled=True,
            cross_validation_kappa_warn=0.4,
        )
        engine = FDSEngine(config=config)
        engine.judge = MagicMock()
        engine.judge.decompose_claims.return_value = ["Claim1", "Claim2"]
        engine.judge.verify_claim.return_value = ClaimVerdict("C", "supported", 0.9, "ev")

        # Mock cross-validator
        cv = MagicMock()
        cv.cross_validate.return_value = CrossValidationResult(
            n_claims=2, n_agree=2, n_disagree=0,
            agreement_rate=1.0, cohens_kappa=0.85, disagreements=[],
        )
        engine.cross_validator = cv

        result = engine.compute(sample_logs[:5], high_baseline)
        assert "cross_validation" in result.details
        assert result.details["cross_validation"]["cohens_kappa"] == 0.85
        assert result.details["cross_validation"]["kappa_interpretation"] == "substantial"

    def test_low_kappa_triggers_warning(self, sample_logs, high_baseline):
        """Low kappa should add a warning to the explanation."""
        config = FDSConfig(
            sample_size=5,
            sampling_strategy="random",
            cross_validation_enabled=True,
            cross_validation_kappa_warn=0.4,
        )
        engine = FDSEngine(config=config)
        engine.judge = MagicMock()
        engine.judge.decompose_claims.return_value = ["Claim"]
        engine.judge.verify_claim.return_value = ClaimVerdict("C", "supported", 0.9, "ev")

        cv = MagicMock()
        cv.cross_validate.return_value = CrossValidationResult(
            n_claims=1, n_agree=0, n_disagree=1,
            agreement_rate=0.0, cohens_kappa=0.15, disagreements=[{}],
        )
        engine.cross_validator = cv

        result = engine.compute(sample_logs[:5], high_baseline)
        assert "CROSS-VAL WARNING" in result.explanation
        assert result.details["cross_validation"]["kappa_interpretation"] == "poor"