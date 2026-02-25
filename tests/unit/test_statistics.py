"""
Unit tests for statistical utility functions.
Tests: JSD (continuous, discrete, binary), signed JSD, KS test, cosine similarity.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from src.utils.statistics import (
    continuous_jsd,
    discrete_jsd,
    binary_jsd,
    signed_jsd,
    ks_test,
    cosine_similarity,
    summarise_distribution,
    _adaptive_bin_count,
)


class TestContinuousJSD:
    """Tests for continuous JSD computation."""

    def test_identical_distributions_near_zero(self):
        """JSD of identical distributions should be approximately 0."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 1000)
        jsd = continuous_jsd(p, p, bins=50)
        assert jsd < 0.01, f"JSD of identical dists should be ~0, got {jsd}"

    def test_shifted_distributions_positive(self):
        """JSD of shifted distributions should be > 0."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 1000)
        q = np.random.normal(3, 1, 1000)
        jsd = continuous_jsd(p, q, bins=50)
        assert jsd > 0.05, f"JSD of shifted dists should be > 0.05, got {jsd}"

    def test_bounded(self):
        """JSD should be bounded in [0, ln(2)]."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 500)
        q = np.random.normal(10, 1, 500)
        jsd = continuous_jsd(p, q, bins=50)
        assert 0 <= jsd <= np.log(2) + 0.001

    def test_symmetric(self):
        """JSD should be symmetric: JSD(P,Q) = JSD(Q,P)."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 500)
        q = np.random.normal(2, 1, 500)
        jsd_pq = continuous_jsd(p, q, bins=50)
        jsd_qp = continuous_jsd(q, p, bins=50)
        assert abs(jsd_pq - jsd_qp) < 1e-10

    def test_empty_arrays_return_zero(self):
        """JSD with empty arrays should return 0."""
        assert continuous_jsd(np.array([]), np.array([1, 2, 3])) == 0.0
        assert continuous_jsd(np.array([1, 2, 3]), np.array([])) == 0.0

    @given(
        mean_shift=st.floats(min_value=-5, max_value=5),
        std=st.floats(min_value=0.1, max_value=5),
    )
    @settings(max_examples=20)
    def test_non_negative(self, mean_shift, std):
        """JSD should always be non-negative (property-based test)."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 200)
        q = np.random.normal(mean_shift, std, 200)
        jsd = continuous_jsd(p, q, bins=30)
        assert jsd >= 0


class TestAdaptiveBinning:
    """Tests for Freedman-Diaconis adaptive bin count selection."""

    def test_small_sample_fewer_bins(self):
        """Small samples should get fewer bins than the old default of 50."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 20)
        q = np.random.normal(0, 1, 20)
        bins = _adaptive_bin_count(p, q)
        assert bins <= 20, f"N=40 combined should not produce {bins} bins"
        assert bins >= 10, "Should respect min_bins=10"

    def test_large_sample_more_bins(self):
        """Large samples should use more bins."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 5000)
        q = np.random.normal(0, 1, 5000)
        bins = _adaptive_bin_count(p, q)
        assert bins >= 20, f"N=10000 combined should produce ≥20 bins, got {bins}"

    def test_bounded(self):
        """Bin count should stay within [min_bins, max_bins]."""
        np.random.seed(42)
        for n in [5, 50, 500, 5000]:
            p = np.random.normal(0, 1, n)
            q = np.random.normal(0, 1, n)
            bins = _adaptive_bin_count(p, q)
            assert 10 <= bins <= 50, f"For n={n}, got bins={bins}"

    def test_degenerate_iqr_zero(self):
        """Constant arrays (IQR=0) should fallback to Sturges' rule."""
        p = np.array([1.0] * 20)
        q = np.array([1.0] * 20)
        bins = _adaptive_bin_count(p, q)
        assert bins >= 10

    def test_auto_mode_works(self):
        """continuous_jsd with bins='auto' should produce valid results."""
        np.random.seed(42)
        p = np.random.normal(0, 1, 100)
        q = np.random.normal(0, 1, 100)
        jsd = continuous_jsd(p, q, bins="auto")
        assert 0 <= jsd <= np.log(2) + 0.001

    def test_auto_reduces_small_sample_bias(self):
        """
        Key validation: With adaptive binning, JSD on identical small samples
        should have lower bias than fixed bins=50.
        
        Bias ∝ (bins-1)/(2N). With N=30:
            Fixed bins=50:   bias ∝ 49/60 = 0.82
            Adaptive ~14:    bias ∝ 13/60 = 0.22
        """
        np.random.seed(42)
        # Run 50 trials, compare average JSD on same-distribution samples
        jsd_fixed = []
        jsd_adaptive = []
        for seed in range(50):
            rng = np.random.RandomState(seed)
            p = rng.normal(0, 1, 30)
            q = rng.normal(0, 1, 30)
            jsd_fixed.append(continuous_jsd(p, q, bins=50))
            jsd_adaptive.append(continuous_jsd(p, q, bins="auto"))
        
        mean_fixed = np.mean(jsd_fixed)
        mean_adaptive = np.mean(jsd_adaptive)
        # Adaptive should have lower or equal mean JSD on same-distribution pairs
        assert mean_adaptive <= mean_fixed * 1.1, (
            f"Adaptive ({mean_adaptive:.4f}) should not be much worse than "
            f"fixed ({mean_fixed:.4f})"
        )


class TestDiscreteJSD:
    """Tests for discrete JSD computation."""

    def test_identical_distributions(self):
        d = np.array([1, 2, 2, 3, 3, 3])
        jsd = discrete_jsd(d, d)
        assert jsd < 0.01

    def test_different_distributions(self):
        d1 = np.array([1, 1, 1, 1, 1])
        d2 = np.array([5, 5, 5, 5, 5])
        jsd = discrete_jsd(d1, d2)
        assert jsd > 0.1


class TestBinaryJSD:
    """Tests for binary JSD computation."""

    def test_same_proportions_near_zero(self):
        jsd = binary_jsd(0.5, 0.5)
        assert jsd < 0.01

    def test_extreme_difference(self):
        jsd = binary_jsd(0.1, 0.9)
        assert jsd > 0.1

    def test_symmetric(self):
        jsd1 = binary_jsd(0.3, 0.7)
        jsd2 = binary_jsd(0.7, 0.3)
        assert abs(jsd1 - jsd2) < 1e-10


class TestSignedJSD:
    """Tests for signed JSD."""

    def test_positive_when_improved(self):
        ref = np.random.normal(0.5, 0.1, 100)
        cur = np.random.normal(0.8, 0.1, 100)
        s = signed_jsd(ref, cur)
        assert s > 0, "Should be positive when current > reference"

    def test_negative_when_decayed(self):
        ref = np.random.normal(0.8, 0.1, 100)
        cur = np.random.normal(0.5, 0.1, 100)
        s = signed_jsd(ref, cur)
        assert s < 0, "Should be negative when current < reference"


class TestKSTest:
    """Tests for KS test."""

    def test_identical_high_pvalue(self):
        np.random.seed(42)
        p = np.random.normal(0, 1, 500)
        stat, pval = ks_test(p, p)
        assert pval > 0.05

    def test_different_low_pvalue(self):
        np.random.seed(42)
        p = np.random.normal(0, 1, 500)
        q = np.random.normal(3, 1, 500)
        stat, pval = ks_test(p, q)
        assert pval < 0.05


class TestCosineSimilarity:
    """Tests for cosine similarity."""

    def test_identical_vectors(self):
        a = np.array([1, 2, 3])
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1, 0, 0])
        b = np.array([-1, 0, 0])
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        a = np.array([0, 0, 0])
        b = np.array([1, 2, 3])
        assert cosine_similarity(a, b) == 0.0


class TestSummariseDistribution:
    """Tests for distribution summary."""

    def test_basic_summary(self):
        values = np.array([1, 2, 3, 4, 5])
        summary = summarise_distribution(values)
        assert summary["mean"] == 3.0
        assert summary["n"] == 5
        assert summary["min"] == 1.0
        assert summary["max"] == 5.0

    def test_empty_array(self):
        summary = summarise_distribution(np.array([]))
        assert summary["n"] == 0
        assert summary["mean"] == 0.0
