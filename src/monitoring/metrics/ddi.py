"""
DDI — Differential Drift Index.
Detects non-uniform drift across topic segments as a proxy for fairness.

Mechanism:
- Segment queries by topic (benefits, housing, council tax, disability, etc.)
- Compute per-segment quality proxy
- Measure how differently segments are drifting using std/range of drift scores

Thresholds:
- GREEN:  DDI < 0.05
- AMBER:  0.05 ≤ DDI < 0.15
- RED:    DDI ≥ 0.15

Justification: Equality Act 2010, S.149 PSED — topic-based segmentation
serves as a proxy for protected characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.monitoring.metrics.base import BaseMetric, MetricResult
from src.monitoring.data_collection import MetricDataset
from src.production.logging import ProductionLog
from src.utils.config import DDIConfig, load_config
from src.utils.statistics import continuous_jsd


# ── Topic segments ───────────────────────────────────────────────────────────

TOPIC_SEGMENTS: dict[str, dict] = {
    "universal_credit": {
        "keywords": ["universal credit", "uc", "work coach", "claimant commitment"],
        "protected_proxy": "low-income, working-age adults",
    },
    "housing_benefit": {
        "keywords": ["housing benefit", "rent", "spare bedroom", "lha", "council housing"],
        "protected_proxy": "low-income tenants, elderly (State Pension age)",
    },
    "disability_benefits": {
        "keywords": ["pip", "disability", "attendance allowance", "dla", "esa"],
        "protected_proxy": "disabled persons (Equality Act protected characteristic)",
    },
    "council_tax": {
        "keywords": ["council tax", "band", "council tax reduction", "discount"],
        "protected_proxy": "all citizens (geographic variation)",
    },
    "homelessness": {
        "keywords": ["homeless", "emergency housing", "temporary accommodation", "evict"],
        "protected_proxy": "vulnerable persons, rough sleepers",
    },
    "pension": {
        "keywords": ["pension", "retirement", "state pension age", "pension credit"],
        "protected_proxy": "elderly persons (age-related)",
    },
}


@dataclass
class SegmentResult:
    """Drift result for a single topic segment."""
    topic: str
    n_ref: int
    n_cur: int
    ref_quality_mean: float = 0.0
    cur_quality_mean: float = 0.0
    quality_shift: float = 0.0
    drift_score: Optional[float] = None
    protected_proxy: str = ""
    status: str = "INSUFFICIENT_DATA"
    explanation: str = ""


class DDIEngine(BaseMetric):
    """
    Differential Drift Index engine.
    Non-uniform drift detection across topic segments for fairness monitoring.
    """

    name = "DDI"

    def __init__(self, config: Optional[DDIConfig] = None):
        self.config = config or load_config().ddi

    def compute(
        self,
        reference: MetricDataset,
        current: MetricDataset,
        **kwargs,
    ) -> MetricResult:
        """Compute DDI between reference and current windows."""

        # Step 1: Segment queries by topic
        ref_segments = self._segment_by_topic(reference.logs)
        cur_segments = self._segment_by_topic(current.logs)

        # Step 2: Per-segment quality proxy + drift
        per_segment: dict[str, SegmentResult] = {}
        segment_drift_scores: dict[str, float] = {}

        for topic, config in TOPIC_SEGMENTS.items():
            ref_logs = ref_segments.get(topic, [])
            cur_logs = cur_segments.get(topic, [])

            if (len(ref_logs) < self.config.min_segment_size or
                    len(cur_logs) < self.config.min_segment_size):
                per_segment[topic] = SegmentResult(
                    topic=topic,
                    n_ref=len(ref_logs),
                    n_cur=len(cur_logs),
                    protected_proxy=config["protected_proxy"],
                    status="INSUFFICIENT_DATA",
                    explanation=f"Need ≥{self.config.min_segment_size} queries per window",
                )
                continue

            ref_quality = self._compute_quality_proxy(ref_logs)
            cur_quality = self._compute_quality_proxy(cur_logs)

            segment_jsd = continuous_jsd(ref_quality, cur_quality, bins="auto")
            segment_drift_scores[topic] = segment_jsd

            per_segment[topic] = SegmentResult(
                topic=topic,
                n_ref=len(ref_logs),
                n_cur=len(cur_logs),
                ref_quality_mean=float(np.mean(ref_quality)),
                cur_quality_mean=float(np.mean(cur_quality)),
                quality_shift=float(np.mean(cur_quality) - np.mean(ref_quality)),
                drift_score=float(segment_jsd),
                protected_proxy=config["protected_proxy"],
                status=self._classify_segment(segment_jsd),
            )

        # Step 3: DDI = std of segment-level drift scores
        if len(segment_drift_scores) >= 2:
            drift_values = np.array(list(segment_drift_scores.values()))
            ddi_value = float(np.std(drift_values))
            ddi_range = float(np.max(drift_values) - np.min(drift_values))
        else:
            ddi_value = 0.0
            ddi_range = 0.0

        # Step 4: Identify worst/best segments
        worst_segment = (
            max(segment_drift_scores, key=segment_drift_scores.get)
            if segment_drift_scores else None
        )
        best_segment = (
            min(segment_drift_scores, key=segment_drift_scores.get)
            if segment_drift_scores else None
        )

        intersectional_flag = ddi_range > self.config.intersectional_threshold
        status = self._classify_overall(ddi_value, ddi_range)
        explanation = self._generate_explanation(
            ddi_value, per_segment, worst_segment, best_segment
        )

        return MetricResult(
            metric_name="DDI",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=ddi_value,
            status=status,
            explanation=explanation,
            details={
                "ddi_value": ddi_value,
                "ddi_range": ddi_range,
                "n_segments_evaluated": len(segment_drift_scores),
                "per_segment": {k: vars(v) for k, v in per_segment.items()},
                "worst_segment": worst_segment,
                "best_segment": best_segment,
                "intersectional_flag": intersectional_flag,
            },
        )

    def _segment_by_topic(self, logs: list[ProductionLog]) -> dict[str, list[ProductionLog]]:
        """Segment logs by topic using stored query_topic field."""
        segments: dict[str, list[ProductionLog]] = {}
        for log in logs:
            topic = log.query_topic
            # Map to our segment names
            if topic in TOPIC_SEGMENTS:
                segments.setdefault(topic, []).append(log)
            else:
                # Try keyword matching on the query
                matched = self._match_topic(log.raw_query)
                segments.setdefault(matched, []).append(log)
        return segments

    def _match_topic(self, text: str) -> str:
        """Match text to a topic segment via keywords."""
        text_lower = text.lower()
        for topic, config in TOPIC_SEGMENTS.items():
            if any(kw in text_lower for kw in config["keywords"]):
                return topic
        return "unknown"

    def _compute_quality_proxy(self, logs: list[ProductionLog]) -> np.ndarray:
        """
        Quality proxy per response:
        - completeness: token count / expected range
        - citation: has at least 1 GOV.UK citation
        - non_refusal: not a refusal response
        - latency: inverse, normalised
        """
        w = self.config.quality_proxy_weights
        quality_scores = []

        for log in logs:
            completeness = min(log.completion_tokens / 200, 1.0) if log.completion_tokens else 0.5
            has_citation = 1.0 if log.citation_count > 0 else 0.0
            non_refusal = 0.0 if log.refusal_flag else 1.0
            latency_score = max(0, 1.0 - log.response_latency_ms / 5000) if log.response_latency_ms else 0.5

            quality = (
                w["completeness"] * completeness +
                w["citation"] * has_citation +
                w["non_refusal"] * non_refusal +
                w["latency"] * latency_score
            )
            quality_scores.append(quality)

        return np.array(quality_scores)

    def _classify_segment(self, jsd: float) -> str:
        if jsd < self.config.green_threshold:
            return "GREEN"
        elif jsd < self.config.amber_threshold:
            return "AMBER"
        return "RED"

    def _classify_overall(self, ddi: float, ddi_range: float) -> str:
        if ddi < self.config.green_threshold:
            return "GREEN"
        elif ddi < self.config.amber_threshold:
            return "AMBER"
        return "RED"

    def _generate_explanation(
        self, ddi: float, per_segment: dict, worst: Optional[str], best: Optional[str]
    ) -> str:
        evaluated = len([s for s in per_segment.values() if s.drift_score is not None])
        if worst and best and worst in per_segment and best in per_segment:
            w_seg = per_segment[worst]
            b_seg = per_segment[best]
            return (
                f"DDI = {ddi:.4f}. Evaluated {evaluated} topic segments. "
                f"Most drifted: '{worst}' (JSD={w_seg.drift_score:.4f}, "
                f"proxy: {w_seg.protected_proxy}). "
                f"Least drifted: '{best}' (JSD={b_seg.drift_score:.4f}). "
                f"{'Non-uniform drift detected — potential fairness concern under Equality Act S.149.' if ddi >= 0.05 else 'Drift is relatively uniform across segments.'}"
            )
        return f"DDI = {ddi:.4f}. Insufficient data for detailed segment analysis."
