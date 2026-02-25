"""
TRCI — Temporal Response Consistency Index.
Active canary probing to detect model/system changes.

Mechanism:
- Maintain a set of N canary queries with known reference responses
- Periodically re-submit canaries and compare new responses to references
- Uses cosine similarity via Pinecone's integrated embedding

Thresholds:
- GREEN:  TRCI ≥ 0.95  (no detectable drift)
- AMBER:  0.90 ≤ TRCI < 0.95  (investigate)
- RED:    TRCI < 0.90  (significant drift — possible silent model update)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.monitoring.metrics.base import BaseMetric, MetricResult
from src.utils.config import (
    PINECONE_API_KEY, TRCIConfig, load_config, CONFIG_DIR,
)


@dataclass
class CanaryResult:
    """Result for a single canary query."""
    canary_id: str
    query: str
    topic: str
    reference_response_preview: str
    new_response_preview: str
    similarity: float
    status: str


class TRCIEngine(BaseMetric):
    """
    Temporal Response Consistency Index engine.
    Probes system with canary queries, measures response consistency.
    """

    name = "TRCI"

    def __init__(self, config: Optional[TRCIConfig] = None, pipeline=None):
        self.config = config or load_config().trci
        self.pipeline = pipeline  # QueryPipeline instance (injected)
        self.canary_set = self._load_canary_set()
        self._pc = None
        self._index = None

    @property
    def index(self):
        if self._index is None:
            from pinecone import Pinecone
            self._pc = Pinecone(api_key=PINECONE_API_KEY)
            self._index = self._pc.Index("rait-chatbot")
        return self._index

    def _load_canary_set(self) -> list[dict]:
        """Load canary queries from config."""
        canary_path = Path(self.config.canary_path)
        if not canary_path.is_absolute():
            canary_path = CONFIG_DIR.parent / canary_path
        if canary_path.exists():
            with open(canary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def init_reference_embeddings(self, responses: dict[str, str]) -> None:
        """
        Upsert reference response embeddings into Pinecone 'canary_refs' namespace.
        Uses upsert_records() for integrated embedding.
        Called once during initial system setup.
        
        Args:
            responses: dict mapping canary_id → reference_response text
        """
        records = []
        for canary_id, response_text in responses.items():
            records.append({
                "_id": f"ref_{canary_id}",
                "text": response_text,
                "canary_id": canary_id,
            })

        # Batch upsert
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.index.upsert_records(namespace="canary_refs", records=batch)

    def compute(self, **kwargs) -> MetricResult:
        """Run canary probe and compute TRCI metrics."""
        return self.run_probe()

    def run_probe(self) -> MetricResult:
        """
        Execute canary probe:
        1. Re-submit each canary through production pipeline
        2. Embed new response via Pinecone
        3. Compute cosine similarity between new and reference embeddings
        """
        if not self.pipeline:
            return MetricResult(
                metric_name="TRCI",
                timestamp=datetime.now(timezone.utc).isoformat(),
                value=0.0,
                status="RED",
                explanation="No pipeline configured for canary probing.",
            )

        similarities: list[float] = []
        per_canary: list[CanaryResult] = []

        # Only probe canaries that have reference responses
        probing_set = [c for c in self.canary_set if c.get("reference_response")]
        if not probing_set:
            return MetricResult(
                metric_name="TRCI",
                timestamp=datetime.now(timezone.utc).isoformat(),
                value=0.0,
                status="GREEN",
                explanation="No canaries with reference responses to probe. Run init first.",
            )

        for canary in probing_set:
            canary_id = canary["id"]
            query = canary["query"]
            topic = canary.get("topic", "unknown")
            ref_response = canary.get("reference_response", "")

            try:
                # Re-submit through production pipeline
                log = self.pipeline.process(query)
                new_response = log.raw_response

                # Upsert new response embedding via integrated embedding
                self.index.upsert_records(
                    namespace="canary_probes",
                    records=[{
                        "_id": f"probe_{canary_id}",
                        "text": new_response,
                        "canary_id": canary_id,
                    }],
                )

                # Search reference namespace for similarity
                import time
                time.sleep(1)  # Allow index to sync
                match = self.index.search_records(
                    namespace="canary_refs",
                    query={"inputs": {"text": new_response}, "top_k": 1},
                )

                hits = match.get("result", {}).get("hits", [])
                sim = hits[0].get("_score", 0.0) if hits else 0.0
                similarities.append(sim)

                per_canary.append(CanaryResult(
                    canary_id=canary_id,
                    query=query,
                    topic=topic,
                    reference_response_preview=(ref_response or "")[:200],
                    new_response_preview=new_response[:200],
                    similarity=float(sim),
                    status=self._classify_similarity(sim),
                ))

            except Exception as e:
                per_canary.append(CanaryResult(
                    canary_id=canary_id,
                    query=query,
                    topic=topic,
                    reference_response_preview="",
                    new_response_preview="",
                    similarity=0.0,
                    status=f"ERROR: {str(e)[:80]}",
                ))

            # Rate-limit delay between canary probes (Groq free tier: 30 RPM)
            import time as _time
            _time.sleep(3)

        if not similarities:
            return MetricResult(
                metric_name="TRCI",
                timestamp=datetime.now(timezone.utc).isoformat(),
                value=0.0,
                status="RED",
                explanation="No canary probes succeeded.",
                details={"per_canary": [vars(c) for c in per_canary]},
            )

        sim_arr = np.array(similarities)
        trci_mean = float(np.mean(sim_arr))
        trci_p10 = float(np.percentile(sim_arr, 10))
        trci_std = float(np.std(sim_arr))

        n_green = int(np.sum(sim_arr >= 0.95))
        n_amber = int(np.sum((sim_arr >= 0.90) & (sim_arr < 0.95)))
        n_red = int(np.sum(sim_arr < 0.90))

        status = self._classify_overall(trci_mean, trci_p10)
        explanation = self._generate_explanation(sim_arr, per_canary)

        return MetricResult(
            metric_name="TRCI",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=trci_mean,
            status=status,
            explanation=explanation,
            details={
                "trci_mean": trci_mean,
                "trci_p10": trci_p10,
                "trci_std": trci_std,
                "n_canaries": len(self.canary_set),
                "n_probed": len(similarities),
                "n_green": n_green,
                "n_amber": n_amber,
                "n_red": n_red,
                "per_canary": [vars(c) for c in per_canary],
            },
        )

    def _classify_similarity(self, sim: float) -> str:
        if sim >= self.config.green_threshold:
            return "GREEN"
        elif sim >= self.config.amber_threshold:
            return "AMBER"
        return "RED"

    def _classify_overall(self, mean: float, p10: float) -> str:
        if p10 < self.config.red_p10_threshold:
            return "RED"
        if mean < self.config.amber_threshold:
            return "RED"
        if mean < self.config.green_threshold:
            return "AMBER"
        return "GREEN"

    def _generate_explanation(
        self, similarities: np.ndarray, per_canary: list[CanaryResult]
    ) -> str:
        valid = [c for c in per_canary if c.similarity > 0]
        if not valid:
            return "No valid canary results to analyse."
        worst = min(valid, key=lambda x: x.similarity)
        return (
            f"TRCI probe ran {len(valid)} canary queries. "
            f"Mean similarity: {np.mean(similarities):.4f}, "
            f"10th percentile: {np.percentile(similarities, 10):.4f}. "
            f"Worst canary: '{worst.query[:80]}...' "
            f"(topic: {worst.topic}, similarity: {worst.similarity:.4f}). "
            f"{'No action needed.' if np.mean(similarities) >= 0.95 else 'Investigation recommended.'}"
        )
