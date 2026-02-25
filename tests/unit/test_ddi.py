"""
Unit tests for DDI metric engine.
Tests the fairness monitoring across topic segments.
"""

import numpy as np
import pytest

from src.production.logging import ProductionLog, generate_id, utcnow
from src.monitoring.data_collection import MetricDataset
from src.monitoring.descriptors import extract_descriptors
from src.monitoring.metrics.ddi import DDIEngine

from datetime import datetime, timezone, timedelta


def _make_logs(topic: str, n: int, quality_base: float = 0.7) -> list[ProductionLog]:
    """Generate synthetic logs for a given topic."""
    logs = []
    for i in range(n):
        logs.append(ProductionLog(
            query_id=generate_id(),
            timestamp=utcnow(),
            raw_query=f"Test {topic} query {i}",
            query_topic=topic,
            completion_tokens=int(200 * quality_base) + i,
            citation_count=2 if quality_base > 0.5 else 0,
            refusal_flag=quality_base < 0.3,
            response_latency_ms=int(1000 / max(quality_base, 0.1)),
            response_length=int(150 * quality_base),
        ))
    return logs


class TestDDIComputation:

    def test_uniform_drift_green(self):
        """Uniform drift across segments should produce GREEN DDI."""
        now = datetime.now(timezone.utc)
        
        # Same quality across all topics in both windows
        ref_logs = []
        cur_logs = []
        for topic in ["universal_credit", "housing_benefit", "disability_benefits",
                       "council_tax", "homelessness", "pension"]:
            ref_logs.extend(_make_logs(topic, 25, quality_base=0.7))
            cur_logs.extend(_make_logs(topic, 25, quality_base=0.7))

        ref = MetricDataset(
            window_start=now - timedelta(days=30),
            window_end=now,
            logs=ref_logs,
            descriptors=extract_descriptors(ref_logs),
        )
        cur = MetricDataset(
            window_start=now - timedelta(days=7),
            window_end=now,
            logs=cur_logs,
            descriptors=extract_descriptors(cur_logs),
        )

        ddi = DDIEngine()
        ddi.config.min_segment_size = 5  # Lower for test
        result = ddi.compute(ref, cur)
        assert result.status == "GREEN", f"Expected GREEN, got {result.status}"

    def test_non_uniform_drift_amber_or_red(self):
        """Non-uniform drift (one topic degraded) should produce AMBER/RED DDI."""
        now = datetime.now(timezone.utc)

        ref_logs = []
        cur_logs = []
        for topic in ["universal_credit", "housing_benefit", "disability_benefits",
                       "council_tax", "homelessness", "pension"]:
            ref_logs.extend(_make_logs(topic, 25, quality_base=0.7))
            # Disability benefits degraded significantly
            if topic == "disability_benefits":
                cur_logs.extend(_make_logs(topic, 25, quality_base=0.2))
            else:
                cur_logs.extend(_make_logs(topic, 25, quality_base=0.7))

        ref = MetricDataset(
            window_start=now - timedelta(days=30),
            window_end=now,
            logs=ref_logs,
            descriptors=extract_descriptors(ref_logs),
        )
        cur = MetricDataset(
            window_start=now - timedelta(days=7),
            window_end=now,
            logs=cur_logs,
            descriptors=extract_descriptors(cur_logs),
        )

        ddi = DDIEngine()
        ddi.config.min_segment_size = 5
        result = ddi.compute(ref, cur)
        # Should detect non-uniform drift
        assert result.value > 0, "DDI should be > 0 with non-uniform drift"

    def test_insufficient_data(self):
        """DDI with too few queries should report INSUFFICIENT_DATA."""
        now = datetime.now(timezone.utc)
        ref_logs = _make_logs("universal_credit", 3)
        cur_logs = _make_logs("universal_credit", 3)

        ref = MetricDataset(
            window_start=now - timedelta(days=30),
            window_end=now,
            logs=ref_logs,
        )
        cur = MetricDataset(
            window_start=now - timedelta(days=7),
            window_end=now,
            logs=cur_logs,
        )

        ddi = DDIEngine()
        result = ddi.compute(ref, cur)
        assert result.value == 0.0  # Not enough data for any segment
