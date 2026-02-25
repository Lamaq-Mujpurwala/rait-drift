"""
Production logging — SQLite schema + read/write operations.
Every query through the production pipeline produces a unified log entry.
This log is the single source of truth consumed by Part B (monitoring).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def generate_id() -> str:
    """Generate a unique query ID."""
    return str(uuid.uuid4())


def utcnow() -> str:
    """ISO-format UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS production_logs (
    query_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,

    -- Query data
    raw_query TEXT NOT NULL,
    cleaned_query TEXT,
    query_topic TEXT,
    query_token_count INTEGER,

    -- Retrieval data
    top_k_chunk_ids TEXT,          -- JSON array
    top_k_scores TEXT,             -- JSON array of floats
    reranked_chunk_ids TEXT,       -- JSON array
    selected_context_ids TEXT,     -- JSON array
    context_token_count INTEGER,

    -- Generation data
    model_name TEXT,
    temperature REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    raw_response TEXT,
    final_response TEXT,
    finish_reason TEXT,

    -- Derived descriptors (computed post-generation for CDS)
    response_length INTEGER,
    response_sentiment REAL,
    response_readability REAL,
    citation_count INTEGER,
    hedge_word_count INTEGER,
    refusal_flag INTEGER,         -- boolean stored as 0/1

    -- Metadata
    session_id TEXT,
    response_latency_ms INTEGER,
    mean_retrieval_distance REAL,
    context_token_ratio REAL
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON production_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_topic ON production_logs(query_topic);
CREATE INDEX IF NOT EXISTS idx_session ON production_logs(session_id);
"""


# ── Data class ───────────────────────────────────────────────────────────────

@dataclass
class ProductionLog:
    """A single production log entry."""

    query_id: str
    timestamp: str
    raw_query: str
    cleaned_query: str = ""
    query_topic: str = "unknown"
    query_token_count: int = 0

    # Retrieval
    top_k_chunk_ids: list[str] = field(default_factory=list)
    top_k_scores: list[float] = field(default_factory=list)
    reranked_chunk_ids: list[str] = field(default_factory=list)
    selected_context_ids: list[str] = field(default_factory=list)
    context_token_count: int = 0

    # Generation
    model_name: str = ""
    temperature: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: str = ""
    final_response: str = ""
    finish_reason: str = ""

    # Descriptors
    response_length: int = 0
    response_sentiment: float = 0.0
    response_readability: float = 0.0
    citation_count: int = 0
    hedge_word_count: int = 0
    refusal_flag: bool = False

    # Metadata
    session_id: str = ""
    response_latency_ms: int = 0
    mean_retrieval_distance: float = 0.0
    context_token_ratio: float = 0.0


# ── Log Store ────────────────────────────────────────────────────────────────

class LogStore:
    """SQLite-backed production log store."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)

    def write(self, log: ProductionLog) -> None:
        """Write a production log entry to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO production_logs (
                    query_id, timestamp, raw_query, cleaned_query, query_topic,
                    query_token_count, top_k_chunk_ids, top_k_scores,
                    reranked_chunk_ids, selected_context_ids, context_token_count,
                    model_name, temperature, prompt_tokens, completion_tokens,
                    raw_response, final_response, finish_reason,
                    response_length, response_sentiment, response_readability,
                    citation_count, hedge_word_count, refusal_flag,
                    session_id, response_latency_ms, mean_retrieval_distance,
                    context_token_ratio
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (
                    log.query_id,
                    log.timestamp,
                    log.raw_query,
                    log.cleaned_query,
                    log.query_topic,
                    log.query_token_count,
                    json.dumps(log.top_k_chunk_ids),
                    json.dumps(log.top_k_scores),
                    json.dumps(log.reranked_chunk_ids),
                    json.dumps(log.selected_context_ids),
                    log.context_token_count,
                    log.model_name,
                    log.temperature,
                    log.prompt_tokens,
                    log.completion_tokens,
                    log.raw_response,
                    log.final_response,
                    log.finish_reason,
                    log.response_length,
                    log.response_sentiment,
                    log.response_readability,
                    log.citation_count,
                    log.hedge_word_count,
                    int(log.refusal_flag),
                    log.session_id,
                    log.response_latency_ms,
                    log.mean_retrieval_distance,
                    log.context_token_ratio,
                ),
            )

    def read_window(
        self,
        start: str | datetime,
        end: str | datetime,
    ) -> list[ProductionLog]:
        """Read all logs in a time window."""
        start_str = start.isoformat() if isinstance(start, datetime) else start
        end_str = end.isoformat() if isinstance(end, datetime) else end

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM production_logs WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                (start_str, end_str),
            ).fetchall()

        return [self._row_to_log(row) for row in rows]

    def read_all(self) -> list[ProductionLog]:
        """Read all logs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM production_logs ORDER BY timestamp"
            ).fetchall()
        return [self._row_to_log(row) for row in rows]

    def count(self) -> int:
        """Count total log entries."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM production_logs").fetchone()[0]

    def _row_to_log(self, row: sqlite3.Row) -> ProductionLog:
        """Convert a SQLite row to a ProductionLog dataclass."""
        return ProductionLog(
            query_id=row["query_id"],
            timestamp=row["timestamp"],
            raw_query=row["raw_query"],
            cleaned_query=row["cleaned_query"] or "",
            query_topic=row["query_topic"] or "unknown",
            query_token_count=row["query_token_count"] or 0,
            top_k_chunk_ids=json.loads(row["top_k_chunk_ids"] or "[]"),
            top_k_scores=json.loads(row["top_k_scores"] or "[]"),
            reranked_chunk_ids=json.loads(row["reranked_chunk_ids"] or "[]"),
            selected_context_ids=json.loads(row["selected_context_ids"] or "[]"),
            context_token_count=row["context_token_count"] or 0,
            model_name=row["model_name"] or "",
            temperature=row["temperature"] or 0.0,
            prompt_tokens=row["prompt_tokens"] or 0,
            completion_tokens=row["completion_tokens"] or 0,
            raw_response=row["raw_response"] or "",
            final_response=row["final_response"] or "",
            finish_reason=row["finish_reason"] or "",
            response_length=row["response_length"] or 0,
            response_sentiment=row["response_sentiment"] or 0.0,
            response_readability=row["response_readability"] or 0.0,
            citation_count=row["citation_count"] or 0,
            hedge_word_count=row["hedge_word_count"] or 0,
            refusal_flag=bool(row["refusal_flag"]),
            session_id=row["session_id"] or "",
            response_latency_ms=row["response_latency_ms"] or 0,
            mean_retrieval_distance=row["mean_retrieval_distance"] or 0.0,
            context_token_ratio=row["context_token_ratio"] or 0.0,
        )
