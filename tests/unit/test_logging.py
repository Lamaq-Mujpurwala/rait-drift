"""
Unit tests for production logging (SQLite round-trip).
"""

import tempfile
from pathlib import Path

import pytest

from src.production.logging import LogStore, ProductionLog, generate_id, utcnow


@pytest.fixture
def log_store(tmp_path):
    return LogStore(tmp_path / "test.db")


class TestLogStore:

    def test_write_and_read(self, log_store):
        log = ProductionLog(
            query_id=generate_id(),
            timestamp=utcnow(),
            raw_query="What is Universal Credit?",
            response_length=150,
        )
        log_store.write(log)
        assert log_store.count() == 1

    def test_read_all(self, log_store):
        for i in range(5):
            log_store.write(ProductionLog(
                query_id=generate_id(),
                timestamp=utcnow(),
                raw_query=f"Query {i}",
            ))
        logs = log_store.read_all()
        assert len(logs) == 5

    def test_round_trip_preserves_data(self, log_store):
        original = ProductionLog(
            query_id=generate_id(),
            timestamp=utcnow(),
            raw_query="What is PIP?",
            cleaned_query="what is pip?",
            query_topic="disability_benefits",
            query_token_count=4,
            response_length=200,
            response_sentiment=0.5,
            response_readability=72.0,
            citation_count=3,
            hedge_word_count=1,
            refusal_flag=False,
            response_latency_ms=900,
            mean_retrieval_distance=0.12,
            context_token_ratio=2.5,
            raw_response="PIP is a benefit...",
            final_response="PIP is a benefit... [disclaimer]",
            model_name="groq/llama-3.3-70b-versatile",
            completion_tokens=50,
        )
        log_store.write(original)
        retrieved = log_store.read_all()[0]
        assert retrieved.query_id == original.query_id
        assert retrieved.raw_query == original.raw_query
        assert retrieved.query_topic == original.query_topic
        assert retrieved.response_length == original.response_length
        assert retrieved.citation_count == original.citation_count
        assert retrieved.refusal_flag == original.refusal_flag
        assert abs(retrieved.response_sentiment - original.response_sentiment) < 1e-6

    def test_read_window(self, log_store):
        # Write logs at different times
        from datetime import datetime, timezone, timedelta
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for i in range(10):
            log_store.write(ProductionLog(
                query_id=generate_id(),
                timestamp=(base + timedelta(days=i)).isoformat(),
                raw_query=f"Query {i}",
            ))
        # Read a window
        start = (base + timedelta(days=2)).isoformat()
        end = (base + timedelta(days=5)).isoformat()
        window_logs = log_store.read_window(start, end)
        assert 3 <= len(window_logs) <= 4  # days 2,3,4,5

    def test_empty_store(self, log_store):
        assert log_store.count() == 0
        assert log_store.read_all() == []
