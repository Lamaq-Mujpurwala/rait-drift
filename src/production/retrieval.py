"""
Retrieval pipeline — Query → Pinecone search (integrated embedding) → 
Re-ranking → Context assembly. Every stage is logged.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.utils.config import PINECONE_API_KEY, RetrievalConfig, load_config
from src.utils.text_processing import preprocess_query, token_count


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Result of a retrieval query."""
    context: str
    chunk_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    reranked_ids: list[str] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)
    context_token_count: int = 0
    mean_retrieval_distance: float = 0.0


# ── Retrieval Pipeline ───────────────────────────────────────────────────────

class RetrievalPipeline:
    """
    Query → Pinecone Search (integrated embedding) → Re-ranking → Context Assembly.
    Pinecone embeds queries server-side via llama-text-embed-v2 (1024-dim, cosine).
    """

    def __init__(self, config: Optional[RetrievalConfig] = None, namespace: str = "govuk"):
        self.config = config or load_config().retrieval
        self.namespace = namespace
        self._pc = None
        self._index = None

    @property
    def index(self):
        """Lazy-init Pinecone index."""
        if self._index is None:
            from pinecone import Pinecone
            self._pc = Pinecone(api_key=PINECONE_API_KEY)
            self._index = self._pc.Index("rait-chatbot")
        return self._index

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Full retrieval pipeline:
        1. Query preprocessing
        2. Pinecone search_records (integrated embedding)
        3. Metadata-boosted reranking
        4. Context assembly
        """
        # Stage 1: Preprocess
        cleaned = preprocess_query(query)

        # Stage 2: Vector search via Pinecone integrated embedding
        results = self.index.search_records(
            namespace=self.namespace,
            query={"inputs": {"text": cleaned}, "top_k": self.config.top_k},
        )

        hits = results.get("result", {}).get("hits", [])
        if not hits:
            return RetrievalResult(
                context="No relevant documents found.",
                context_token_count=0,
            )

        # Normalise hits into a common format for reranking
        matches = []
        for h in hits:
            fields = h.get("fields", {})
            matches.append({
                "id": h["_id"],
                "score": h.get("_score", 0.0),
                "metadata": fields,
            })

        # Extract IDs and scores
        chunk_ids = [m["id"] for m in matches]
        scores = [m.get("score", 0.0) for m in matches]

        # Stage 3: Re-ranking (metadata-boosted)
        reranked = self._rerank(matches, query)
        reranked_ids = [m["id"] for m in reranked]

        # Stage 4: Context assembly (top context_window after reranking)
        selected = reranked[: self.config.context_window]
        selected_ids = [m["id"] for m in selected]

        context_parts = []
        for m in selected:
            meta = m.get("metadata", {})
            source = meta.get("source", "unknown")
            part_title = meta.get("part_title", "")
            updated = meta.get("last_updated", "")
            text = meta.get("text", "")
            context_parts.append(
                f"[Source: {source} | Part: {part_title} | Updated: {updated}]\n{text}"
            )

        assembled = "\n\n---\n\n".join(context_parts)
        ctx_tokens = token_count(assembled)

        # Compute mean retrieval distance (1 - score for cosine)
        mean_dist = float(np.mean([1 - s for s in scores])) if scores else 0.0

        return RetrievalResult(
            context=assembled,
            chunk_ids=chunk_ids,
            scores=scores,
            reranked_ids=reranked_ids,
            selected_ids=selected_ids,
            context_token_count=ctx_tokens,
            mean_retrieval_distance=mean_dist,
        )

    def _rerank(self, matches: list[dict], query: str) -> list[dict]:
        """
        Metadata-boosted reranking:
        - Base score: Pinecone cosine similarity
        - Boost for recency (newer content scores higher)
        - Boost for category match with query
        """
        scored = []
        query_lower = query.lower()

        for m in matches:
            base_score = m.get("score", 0.0)
            meta = m.get("metadata", {})

            # Recency boost (max 0.05 for content updated in last 30 days)
            recency_boost = 0.0
            updated = meta.get("last_updated", "")
            if updated:
                try:
                    from datetime import datetime, timezone, timedelta
                    updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    age_days = (datetime.now(timezone.utc) - updated_dt).days
                    if age_days < 30:
                        recency_boost = 0.05
                    elif age_days < 90:
                        recency_boost = 0.02
                except (ValueError, TypeError):
                    pass

            # Title relevance boost
            title_boost = 0.0
            part_title = meta.get("part_title", "").lower()
            if any(word in part_title for word in query_lower.split() if len(word) > 3):
                title_boost = 0.03

            final_score = base_score + recency_boost + title_boost
            scored.append((final_score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]
