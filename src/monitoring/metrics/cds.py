"""
CDS — Composite Drift Signal.
Monitors distribution shifts across K text descriptors using JSD.

Mechanism:
- For each descriptor, compute JSD between reference and current windows
- Weighted fusion into single composite score
- Persistence filter: only flag if drift persists for P consecutive windows

Thresholds:
- GREEN:  CDS < 0.05
- AMBER:  0.05 ≤ CDS < 0.15
- RED:    CDS ≥ 0.15 (persisting for P=3 consecutive windows)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.monitoring.metrics.base import BaseMetric, MetricResult
from src.monitoring.descriptors import DescriptorSet
from src.utils.config import CDSConfig, CDSDescriptorConfig, load_config
from src.utils.statistics import (
    continuous_jsd, discrete_jsd, binary_jsd, summarise_distribution,
)


# ── Default Descriptor Config ────────────────────────────────────────────────

DEFAULT_DESCRIPTORS: dict[str, CDSDescriptorConfig] = {
    "response_length":       CDSDescriptorConfig(weight=0.15, type="continuous"),
    "response_sentiment":    CDSDescriptorConfig(weight=0.10, type="continuous"),
    "response_readability":  CDSDescriptorConfig(weight=0.10, type="continuous"),
    "citation_count":        CDSDescriptorConfig(weight=0.15, type="discrete"),
    "hedge_word_count":      CDSDescriptorConfig(weight=0.10, type="discrete"),
    "refusal_flag":          CDSDescriptorConfig(weight=0.15, type="binary"),
    "response_latency_ms":   CDSDescriptorConfig(weight=0.10, type="continuous"),
    "mean_retrieval_distance": CDSDescriptorConfig(weight=0.10, type="continuous"),
    "context_token_ratio":   CDSDescriptorConfig(weight=0.05, type="continuous"),
}


@dataclass
class DescriptorDriftResult:
    """Drift result for a single descriptor."""
    name: str
    weight: float
    jsd: float
    weighted_contribution: float
    ref_summary: dict
    cur_summary: dict
    status: str


class CDSEngine(BaseMetric):
    """
    Composite Drift Signal engine.
    Weighted JSD fusion across 9 descriptors with persistence filtering.
    """

    name = "CDS"

    def __init__(self, config: Optional[CDSConfig] = None):
        self.config = config or load_config().cds
        if not self.config.descriptors:
            self.config.descriptors = DEFAULT_DESCRIPTORS
        self.persistence_history: list[float] = []

    def compute(
        self,
        reference: DescriptorSet,
        current: DescriptorSet,
        **kwargs,
    ) -> MetricResult:
        """Compute CDS between reference and current descriptor distributions."""

        per_descriptor: dict[str, DescriptorDriftResult] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for name, desc_cfg in self.config.descriptors.items():
            if not desc_cfg.enabled:
                continue

            ref_values = reference.get_values(name)
            cur_values = current.get_values(name)

            if len(ref_values) == 0 or len(cur_values) == 0:
                continue

            # Compute JSD based on descriptor type
            if desc_cfg.type == "binary":
                ref_prop = float(np.mean(ref_values))
                cur_prop = float(np.mean(cur_values))
                jsd = binary_jsd(ref_prop, cur_prop)
            elif desc_cfg.type == "discrete":
                jsd = discrete_jsd(ref_values, cur_values)
            else:  # continuous
                jsd = continuous_jsd(
                    ref_values, cur_values, bins=self.config.jsd_bins
                )

            weighted_contrib = jsd * desc_cfg.weight
            per_descriptor[name] = DescriptorDriftResult(
                name=name,
                weight=desc_cfg.weight,
                jsd=float(jsd),
                weighted_contribution=float(weighted_contrib),
                ref_summary=summarise_distribution(ref_values),
                cur_summary=summarise_distribution(cur_values),
                status=self._classify_descriptor(jsd),
            )

            weighted_sum += weighted_contrib
            total_weight += desc_cfg.weight

        cds_value = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Persistence check
        self.persistence_history.append(cds_value)
        persistence_met = self._check_persistence(cds_value)

        status = self._classify_overall(cds_value, persistence_met)
        explanation = self._generate_explanation(cds_value, per_descriptor, persistence_met)

        return MetricResult(
            metric_name="CDS",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=float(cds_value),
            status=status,
            explanation=explanation,
            details={
                "cds_value": float(cds_value),
                "per_descriptor": {k: vars(v) for k, v in per_descriptor.items()},
                "persistence_history": self.persistence_history[-10:],
                "persistence_met": persistence_met,
            },
        )

    def _classify_descriptor(self, jsd: float) -> str:
        if jsd < self.config.green_threshold:
            return "GREEN"
        elif jsd < self.config.amber_threshold:
            return "AMBER"
        return "RED"

    def _classify_overall(self, cds: float, persistence_met: bool) -> str:
        if cds < self.config.green_threshold:
            return "GREEN"
        elif cds < self.config.amber_threshold:
            return "AMBER"
        elif persistence_met:
            return "RED"
        return "AMBER"

    def _check_persistence(self, cds_value: float) -> bool:
        """Check if drift has persisted for P consecutive windows."""
        p = self.config.persistence_threshold
        if len(self.persistence_history) < p:
            return False
        recent = self.persistence_history[-p:]
        return all(v >= self.config.amber_threshold for v in recent)

    def _generate_explanation(
        self, cds_value: float, per_descriptor: dict, persistence_met: bool
    ) -> str:
        if not per_descriptor:
            return f"CDS = {cds_value:.4f}. No descriptors available for analysis."

        top_contributors = sorted(
            per_descriptor.values(),
            key=lambda x: x.weighted_contribution,
            reverse=True,
        )[:3]

        contrib_text = "; ".join(
            f"{c.name} (JSD={c.jsd:.4f}, weight={c.weight})"
            for c in top_contributors
        )

        return (
            f"CDS = {cds_value:.4f}. "
            f"Top drift contributors: {contrib_text}. "
            f"{'Persistence threshold met — action required.' if persistence_met else 'Within acceptable bounds or not yet persistent.'}"
        )
