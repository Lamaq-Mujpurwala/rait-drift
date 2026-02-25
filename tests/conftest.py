"""
Shared test fixtures for RAIT test suite.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.production.logging import LogStore, ProductionLog, generate_id, utcnow
from src.monitoring.descriptors import DescriptorSet


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary SQLite database."""
    return LogStore(tmp_path / "test.db")


@pytest.fixture
def sample_logs():
    """A list of sample production logs for testing."""
    logs = []
    topics = ["universal_credit", "housing_benefit", "disability_benefits",
              "council_tax", "homelessness", "pension"]
    for i in range(60):
        logs.append(ProductionLog(
            query_id=generate_id(),
            timestamp=utcnow(),
            raw_query=f"Test query {i} about {topics[i % len(topics)]}",
            cleaned_query=f"test query {i}",
            query_topic=topics[i % len(topics)],
            query_token_count=5 + i % 10,
            response_length=100 + i * 2,
            response_sentiment=0.3 + (i % 5) * 0.1,
            response_readability=60.0 + i % 20,
            citation_count=i % 4,
            hedge_word_count=i % 3,
            refusal_flag=i % 15 == 0,
            response_latency_ms=800 + i * 10,
            mean_retrieval_distance=0.1 + (i % 10) * 0.02,
            context_token_ratio=2.0 + (i % 5) * 0.3,
            raw_response=f"Response to query {i}",
            final_response=f"Response to query {i} [disclaimer]",
            model_name="groq/llama-3.3-70b-versatile",
            completion_tokens=100 + i * 2,
        ))
    return logs


@pytest.fixture
def reference_descriptors():
    """Reference window descriptor set with stable distributions."""
    np.random.seed(42)
    return DescriptorSet(
        data={
            "response_length": np.random.normal(150, 20, 100),
            "response_sentiment": np.random.normal(0.3, 0.1, 100),
            "response_readability": np.random.normal(65, 10, 100),
            "citation_count": np.random.choice([1, 2, 3], 100).astype(float),
            "hedge_word_count": np.random.choice([0, 1, 2], 100).astype(float),
            "refusal_flag": np.random.choice([0, 1], 100, p=[0.9, 0.1]).astype(float),
            "response_latency_ms": np.random.normal(1000, 200, 100),
            "mean_retrieval_distance": np.random.normal(0.15, 0.05, 100),
            "context_token_ratio": np.random.normal(2.5, 0.5, 100),
        },
        query_ids=[f"ref_{i}" for i in range(100)],
    )


@pytest.fixture
def drifted_descriptors():
    """Current window descriptor set with significant drift."""
    np.random.seed(99)
    return DescriptorSet(
        data={
            "response_length": np.random.normal(80, 30, 100),
            "response_sentiment": np.random.normal(-0.1, 0.2, 100),
            "response_readability": np.random.normal(45, 15, 100),
            "citation_count": np.random.choice([0, 1], 100).astype(float),
            "hedge_word_count": np.random.choice([2, 3, 4], 100).astype(float),
            "refusal_flag": np.random.choice([0, 1], 100, p=[0.5, 0.5]).astype(float),
            "response_latency_ms": np.random.normal(2000, 500, 100),
            "mean_retrieval_distance": np.random.normal(0.35, 0.1, 100),
            "context_token_ratio": np.random.normal(1.0, 0.3, 100),
        },
        query_ids=[f"drift_{i}" for i in range(100)],
    )
