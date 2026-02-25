"""
Data collection layer — reads production logs and prepares metric-ready datasets.
Runs as a batch job (configurable: hourly/daily/weekly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

from src.production.logging import LogStore, ProductionLog
from src.monitoring.descriptors import DescriptorSet, extract_descriptors


@dataclass
class MetricDataset:
    """A collection of production logs within a time window, enriched with descriptors."""
    window_start: datetime
    window_end: datetime
    logs: list[ProductionLog] = field(default_factory=list)
    descriptors: Optional[DescriptorSet] = None
    n_queries: int = 0

    def __post_init__(self):
        self.n_queries = len(self.logs)


class DataCollectionLayer:
    """
    Taps into production logs and prepares data for metric computation.
    """

    def __init__(self, log_store: LogStore):
        self.log_store = log_store

    def collect_window(
        self,
        window_start: datetime,
        window_end: datetime,
    ) -> MetricDataset:
        """Collect all production logs in the given time window."""
        logs = self.log_store.read_window(
            window_start.isoformat(),
            window_end.isoformat(),
        )

        dataset = MetricDataset(
            window_start=window_start,
            window_end=window_end,
            logs=logs,
        )

        # Enrich with descriptor extraction
        dataset.descriptors = extract_descriptors(logs)

        return dataset

    def collect_reference(self, days: int = 30) -> MetricDataset:
        """Collect the reference window (last N days as baseline)."""
        now = datetime.now(timezone.utc)
        return self.collect_window(now - timedelta(days=days), now)

    def collect_current(self, days: int = 7) -> MetricDataset:
        """Collect the current monitoring window."""
        now = datetime.now(timezone.utc)
        return self.collect_window(now - timedelta(days=days), now)
