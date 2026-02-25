"""
Unit tests for CDS metric engine.
Tests the full CDS computation pipeline with synthetic data.
"""

import numpy as np
import pytest

from src.monitoring.descriptors import DescriptorSet
from src.monitoring.metrics.cds import CDSEngine


class TestCDSComputation:
    """Tests for CDS metric computation."""

    def test_identical_distributions_green(self, reference_descriptors):
        """CDS of identical distributions should be GREEN."""
        cds = CDSEngine()
        result = cds.compute(reference_descriptors, reference_descriptors)
        assert result.status == "GREEN", f"Expected GREEN, got {result.status}"
        assert result.value < 0.05, f"CDS should be < 0.05, got {result.value}"

    def test_drifted_distributions_amber_or_red(self, reference_descriptors, drifted_descriptors):
        """CDS of drifted distributions should be AMBER or RED."""
        cds = CDSEngine()
        result = cds.compute(reference_descriptors, drifted_descriptors)
        assert result.status in ("AMBER", "RED"), f"Expected AMBER/RED, got {result.status}"
        assert result.value > 0.05, f"CDS should be > 0.05, got {result.value}"

    def test_per_descriptor_breakdown(self, reference_descriptors, drifted_descriptors):
        """CDS result should include per-descriptor breakdown."""
        cds = CDSEngine()
        result = cds.compute(reference_descriptors, drifted_descriptors)
        per_desc = result.details.get("per_descriptor", {})
        assert len(per_desc) > 0, "Should have per-descriptor results"
        assert "response_length" in per_desc

    def test_cds_value_is_bounded(self, reference_descriptors, drifted_descriptors):
        """CDS value should be bounded [0, ln(2)]."""
        cds = CDSEngine()
        result = cds.compute(reference_descriptors, drifted_descriptors)
        assert 0 <= result.value <= np.log(2) + 0.01

    def test_persistence_detection(self, reference_descriptors, drifted_descriptors):
        """Persistence should trigger after P consecutive amber+ windows."""
        cds = CDSEngine()
        # Run 3 times with drifted data to trigger persistence
        for _ in range(3):
            result = cds.compute(reference_descriptors, drifted_descriptors)
        
        history = result.details.get("persistence_history", [])
        assert len(history) == 3

    def test_empty_descriptors(self):
        """CDS with empty descriptors should not crash."""
        cds = CDSEngine()
        empty = DescriptorSet()
        result = cds.compute(empty, empty)
        assert result.value == 0.0

    def test_explanation_contains_top_contributors(self, reference_descriptors, drifted_descriptors):
        """CDS explanation should mention top drift contributors."""
        cds = CDSEngine()
        result = cds.compute(reference_descriptors, drifted_descriptors)
        assert "Top drift contributors" in result.explanation
