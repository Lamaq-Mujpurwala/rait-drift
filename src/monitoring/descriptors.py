"""
Descriptor extraction engine.
Extracts the K text descriptors needed by CDS from production logs.
Each descriptor is a function: log_entry → scalar value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import json
import numpy as np

from src.production.logging import ProductionLog


@dataclass
class DescriptorSet:
    """
    Container for extracted descriptors across a set of logs.
    Each key is a descriptor name, values are numpy arrays of per-query values.
    """
    data: dict[str, np.ndarray] = field(default_factory=dict)
    query_ids: list[str] = field(default_factory=list)

    def get_values(self, name: str) -> np.ndarray:
        """Get descriptor values by name."""
        return self.data.get(name, np.array([]))

    @property
    def n_queries(self) -> int:
        return len(self.query_ids)


def extract_descriptors(logs: list[ProductionLog]) -> DescriptorSet:
    """
    Extract all 9 CDS descriptors from a list of production logs.
    These descriptors are stored in the production log table (computed at query time).
    """
    if not logs:
        return DescriptorSet()

    query_ids = [log.query_id for log in logs]

    data = {
        "response_length": np.array([log.response_length for log in logs], dtype=float),
        "response_sentiment": np.array([log.response_sentiment for log in logs], dtype=float),
        "response_readability": np.array([log.response_readability for log in logs], dtype=float),
        "citation_count": np.array([log.citation_count for log in logs], dtype=float),
        "hedge_word_count": np.array([log.hedge_word_count for log in logs], dtype=float),
        "refusal_flag": np.array([float(log.refusal_flag) for log in logs], dtype=float),
        "response_latency_ms": np.array([log.response_latency_ms for log in logs], dtype=float),
        "mean_retrieval_distance": np.array(
            [log.mean_retrieval_distance for log in logs], dtype=float
        ),
        "context_token_ratio": np.array([log.context_token_ratio for log in logs], dtype=float),
    }

    return DescriptorSet(data=data, query_ids=query_ids)
