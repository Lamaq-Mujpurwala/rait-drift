"""
Statistical utility functions for drift monitoring.
Core implementations: JSD, cosine similarity, KS tests, histogram helpers.
All functions are pure (no side effects) and well-typed for testing.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy, ks_2samp
from scipy.spatial.distance import cosine as cosine_distance


# ── Jensen-Shannon Divergence ────────────────────────────────────────────────

def _adaptive_bin_count(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    min_bins: int = 10,
    max_bins: int = 50,
) -> int:
    """
    Compute adaptive bin count using Freedman-Diaconis rule.
    
    The FD rule sets bin width h = 2·IQR·n^(-1/3), which adapts to both
    sample spread and size. This reduces the small-sample bias of JSD:
        Bias ∝ (bins - 1) / (2N)
    
    For N=30, fixed bins=50 gives bias ∝ 0.82 (HIGH).
    With adaptive bins ≈ 14, bias ∝ 0.22 (much lower).
    
    Falls back to Sturges' rule if IQR = 0 (degenerate case).
    
    Args:
        p_samples: Reference distribution samples
        q_samples: Current distribution samples
        min_bins: Floor — too few bins lose distributional detail
        max_bins: Ceiling — too many bins cause sparsity
    
    Returns:
        int: Optimal bin count bounded in [min_bins, max_bins]
    """
    combined = np.concatenate([p_samples, q_samples])
    n = len(combined)
    
    if n < 4:
        return min_bins
    
    iqr = float(np.percentile(combined, 75) - np.percentile(combined, 25))
    
    if iqr > 0:
        # Freedman-Diaconis rule
        bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
        data_range = float(np.max(combined) - np.min(combined))
        fd_bins = max(1, int(np.ceil(data_range / bin_width)))
    else:
        # Degenerate case: fall back to Sturges' rule
        fd_bins = max(1, int(np.ceil(np.log2(n) + 1)))
    
    return int(np.clip(fd_bins, min_bins, max_bins))


def continuous_jsd(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bins: int | str = "auto",
) -> float:
    """
    JSD between two continuous distributions via histogram binning.
    
    Returns a value in [0, ln(2)] ≈ [0, 0.693]. 
    Bounded, symmetric, defined even when supports differ.
    
    Args:
        p_samples: Reference distribution samples
        q_samples: Current distribution samples
        bins: Number of histogram bins (shared bin edges).
              "auto" → Freedman-Diaconis adaptive rule (recommended).
              int → fixed bin count (legacy behaviour).
    
    Returns:
        float: JSD value
    """
    if len(p_samples) == 0 or len(q_samples) == 0:
        return 0.0

    # Resolve adaptive bin count
    if bins == "auto":
        n_bins = _adaptive_bin_count(p_samples, q_samples)
    else:
        n_bins = int(bins)

    all_values = np.concatenate([p_samples, q_samples])
    bin_edges = np.histogram_bin_edges(all_values, bins=n_bins)

    p_hist, _ = np.histogram(p_samples, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=True)

    # Epsilon to avoid log(0)
    eps = 1e-10
    p_hist = p_hist.astype(float) + eps
    q_hist = q_hist.astype(float) + eps

    # Normalise to probability distributions
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = (P+Q)/2
    m = 0.5 * (p_hist + q_hist)
    jsd = 0.5 * entropy(p_hist, m) + 0.5 * entropy(q_hist, m)
    return float(jsd)


def discrete_jsd(
    p_values: np.ndarray,
    q_values: np.ndarray,
) -> float:
    """
    JSD for discrete distributions (integer-valued).
    Builds probability distributions from value counts over shared support.
    """
    if len(p_values) == 0 or len(q_values) == 0:
        return 0.0

    # Union of all unique values
    all_vals = np.union1d(p_values, q_values)
    eps = 1e-10

    p_counts = np.array([np.sum(p_values == v) for v in all_vals], dtype=float) + eps
    q_counts = np.array([np.sum(q_values == v) for v in all_vals], dtype=float) + eps

    p_dist = p_counts / p_counts.sum()
    q_dist = q_counts / q_counts.sum()

    m = 0.5 * (p_dist + q_dist)
    jsd = 0.5 * entropy(p_dist, m) + 0.5 * entropy(q_dist, m)
    return float(jsd)


def binary_jsd(p_proportion: float, q_proportion: float) -> float:
    """
    JSD for binary distributions.
    Compares [p, 1-p] vs [q, 1-q].
    """
    eps = 1e-10
    p_dist = np.array([p_proportion + eps, 1 - p_proportion + eps])
    q_dist = np.array([q_proportion + eps, 1 - q_proportion + eps])

    p_dist = p_dist / p_dist.sum()
    q_dist = q_dist / q_dist.sum()

    m = 0.5 * (p_dist + q_dist)
    jsd = 0.5 * entropy(p_dist, m) + 0.5 * entropy(q_dist, m)
    return float(jsd)


def signed_jsd(
    ref_samples: np.ndarray,
    cur_samples: np.ndarray,
    bins: int | str = "auto",
) -> float:
    """
    Signed JSD: positive = distribution shifted higher, negative = shifted lower.
    Sign determined by direction of mean shift.
    
    Used by FDS to indicate improvement vs decay.
    """
    jsd = continuous_jsd(ref_samples, cur_samples, bins=bins)
    sign = 1 if np.mean(cur_samples) >= np.mean(ref_samples) else -1
    return float(sign * jsd)


# ── Kolmogorov-Smirnov Test ──────────────────────────────────────────────────

def ks_test(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
) -> tuple[float, float]:
    """
    Two-sample KS test.
    Returns (statistic, p_value).
    Useful as a secondary confirmation of distribution shift.
    """
    if len(p_samples) < 2 or len(q_samples) < 2:
        return (0.0, 1.0)
    stat, pval = ks_2samp(p_samples, q_samples)
    return (float(stat), float(pval))


# ── Cosine Similarity ────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(1.0 - cosine_distance(a, b))


# ── Summary Statistics ───────────────────────────────────────────────────────

def summarise_distribution(values: np.ndarray) -> dict:
    """Compute summary statistics for a distribution."""
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "p10": 0.0, "p50": 0.0, "p90": 0.0, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "n": int(len(values)),
    }
