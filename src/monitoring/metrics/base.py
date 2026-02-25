"""
Base metric class — shared interface for all four drift metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class MetricResult:
    """Base result for any metric computation."""
    metric_name: str
    timestamp: str
    value: float
    status: str  # "GREEN" | "AMBER" | "RED"
    explanation: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_alert(self) -> bool:
        return self.status in ("AMBER", "RED")


class BaseMetric(ABC):
    """Abstract base class for drift metrics."""

    name: str = "base"

    @abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        """Compute the metric and return a MetricResult."""
        ...

    def classify(self, value: float, green: float, amber: float) -> str:
        """Classify a metric value into GREEN/AMBER/RED."""
        if value <= green:
            return "GREEN"
        elif value <= amber:
            return "AMBER"
        return "RED"
