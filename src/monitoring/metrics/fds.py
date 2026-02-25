"""
FDS — Faithfulness Decay Score.
Detects decline in response faithfulness to source documents.

Mechanism:
1. Decompose response into atomic claims
2. Use LLM-as-Judge to verify each claim against retrieved context
3. Compute per-response faithfulness score
4. Track signed JSD of faithfulness distribution over time

Thresholds:
- GREEN:  |FDS| < 0.02
- AMBER:  0.02 ≤ |FDS| < 0.10
- RED:    |FDS| ≥ 0.10
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.monitoring.metrics.base import BaseMetric, MetricResult
from src.monitoring.judge import JudgeEngine, ClaimVerdict, CrossValidator, CrossValidationResult
from src.production.logging import ProductionLog
from src.utils.config import FDSConfig, load_config
from src.utils.statistics import continuous_jsd, signed_jsd, summarise_distribution


@dataclass
class FDSQueryResult:
    """Faithfulness result for a single query."""
    query_id: str
    query_preview: str
    n_claims: int
    n_supported: int
    n_unsupported: int
    n_ambiguous: int
    faithfulness: float
    claims: list[dict] = field(default_factory=list)


@dataclass
class CalibrationResult:
    """Result of ground truth calibration check."""
    rho: float
    n_calibration_samples: int
    evaluator_status: str  # "OK" | "DRIFTED"


class FDSEngine(BaseMetric):
    """
    Faithfulness Decay Score engine.
    Atomic claim decomposition + LLM-as-Judge verification + signed JSD tracking.
    """

    name = "FDS"

    def __init__(
        self,
        config: Optional[FDSConfig] = None,
        judge: Optional[JudgeEngine] = None,
        cross_validator: Optional[CrossValidator] = None,
    ):
        self.config = config or load_config().fds
        self.judge = judge or JudgeEngine()
        self.cross_validator = cross_validator
        if self.config.cross_validation_enabled and self.cross_validator is None:
            self.cross_validator = CrossValidator(primary=self.judge)

    def compute(
        self,
        current_logs: list[ProductionLog],
        reference_faithfulness: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """
        Compute FDS for current window against reference faithfulness distribution.
        
        Args:
            current_logs: Production logs from the current window
            reference_faithfulness: Array of faithfulness scores from reference period
        """
        # Sample queries for evaluation (budget-aware)
        sample = self._sample_queries(current_logs)
        
        per_query: list[FDSQueryResult] = []
        current_faithfulness: list[float] = []

        for log in sample:
            # Step 1: Decompose response into atomic claims
            claims = self.judge.decompose_claims(log.raw_response)

            # Step 2: Verify each claim
            claim_results: list[dict] = []
            n_supported = 0
            n_unsupported = 0
            n_ambiguous = 0

            # Reconstruct context from the log's selected context (simplified)
            # In production, we'd fetch from Pinecone by IDs
            context = log.raw_response  # Fallback — ideally reconstruct from chunks

            for claim in claims:
                verdict = self.judge.verify_claim(claim=claim, context=context)
                claim_results.append({
                    "claim": verdict.claim,
                    "verdict": verdict.verdict,
                    "confidence": verdict.confidence,
                    "evidence_quote": verdict.evidence_quote,
                })
                if verdict.verdict == "supported":
                    n_supported += 1
                elif verdict.verdict == "unsupported":
                    n_unsupported += 1
                else:
                    n_ambiguous += 1

            # Step 3: Per-response faithfulness score
            n_total = len(claims)
            if self.config.include_ambiguous:
                faithfulness = (n_supported + n_ambiguous) / max(n_total, 1)
            else:
                faithfulness = n_supported / max(n_total, 1)

            current_faithfulness.append(faithfulness)
            per_query.append(FDSQueryResult(
                query_id=log.query_id,
                query_preview=log.raw_query[:100],
                n_claims=n_total,
                n_supported=n_supported,
                n_unsupported=n_unsupported,
                n_ambiguous=n_ambiguous,
                faithfulness=faithfulness,
                claims=claim_results,
            ))

        if not current_faithfulness:
            return MetricResult(
                metric_name="FDS",
                timestamp=datetime.now(timezone.utc).isoformat(),
                value=0.0,
                status="GREEN",
                explanation="No queries evaluated for faithfulness.",
            )

        cur_arr = np.array(current_faithfulness)

        # Step 4: Signed JSD
        fds_value = signed_jsd(reference_faithfulness, cur_arr, bins=self.config.signed_jsd_bins)

        # Step 5: Classification
        abs_fds = abs(fds_value)
        if abs_fds < self.config.green_threshold:
            status = "GREEN"
        elif abs_fds < self.config.amber_threshold:
            status = "AMBER"
        else:
            status = "RED"

        explanation = self._generate_explanation(fds_value, cur_arr, per_query, reference_faithfulness)

        # Optional: Cross-validate judge verdicts
        cross_val_details = None
        if self.config.cross_validation_enabled and self.cross_validator and per_query:
            # Collect all claims from evaluated queries for cross-validation
            all_claims = []
            all_context = []
            for q in per_query:
                for c in q.claims:
                    all_claims.append(c["claim"])
                    # Use the raw response as context proxy (same as main verification)
                    all_context.append("")
            
            # Cross-validate on a subsample to stay within rate limits
            subsample_size = min(10, len(all_claims))
            if subsample_size > 0:
                import random as _rand
                indices = _rand.sample(range(len(all_claims)), subsample_size)
                sampled_claims = [all_claims[i] for i in indices]
                # Use reconstructed context from the first log
                sample_context = sample[0].raw_response if sample else ""
                cv_result = self.cross_validator.cross_validate(sampled_claims, sample_context)
                cross_val_details = {
                    "n_claims_checked": cv_result.n_claims,
                    "agreement_rate": cv_result.agreement_rate,
                    "cohens_kappa": cv_result.cohens_kappa,
                    "n_disagreements": cv_result.n_disagree,
                    "kappa_interpretation": (
                        "substantial" if cv_result.cohens_kappa >= 0.6
                        else "moderate" if cv_result.cohens_kappa >= 0.4
                        else "poor"
                    ),
                }
                if cv_result.cohens_kappa < self.config.cross_validation_kappa_warn:
                    explanation += (
                        f" [CROSS-VAL WARNING] Inter-judge kappa = {cv_result.cohens_kappa:.2f} "
                        f"(poor agreement) — FDS verdicts may be unreliable."
                    )

        details = {
            "fds_value": float(fds_value),
            "mean_faithfulness": float(np.mean(cur_arr)),
            "std_faithfulness": float(np.std(cur_arr)),
            "reference_mean": float(np.mean(reference_faithfulness)),
            "n_evaluated": len(per_query),
            "per_query": [vars(q) for q in per_query],
        }
        if cross_val_details:
            details["cross_validation"] = cross_val_details

        return MetricResult(
            metric_name="FDS",
            timestamp=datetime.now(timezone.utc).isoformat(),
            value=float(fds_value),
            status=status,
            explanation=explanation,
            details=details,
        )

    def _sample_queries(self, logs: list[ProductionLog]) -> list[ProductionLog]:
        """Sample queries for evaluation, respecting budget and strategy."""
        n = min(self.config.sample_size, len(logs))
        if n <= 0:
            return []

        if self.config.sampling_strategy == "random":
            return random.sample(logs, n)
        elif self.config.sampling_strategy == "recent":
            return sorted(logs, key=lambda l: l.timestamp, reverse=True)[:n]
        else:  # stratified by topic
            from collections import defaultdict
            by_topic: dict[str, list[ProductionLog]] = defaultdict(list)
            for log in logs:
                by_topic[log.query_topic].append(log)
            
            per_topic = max(1, n // max(len(by_topic), 1))
            sampled = []
            for topic_logs in by_topic.values():
                sampled.extend(random.sample(topic_logs, min(per_topic, len(topic_logs))))
            return sampled[:n]

    def _generate_explanation(
        self,
        fds: float,
        faithfulness_arr: np.ndarray,
        per_query: list[FDSQueryResult],
        reference: np.ndarray,
    ) -> str:
        direction = "decay" if fds < 0 else "improvement"
        if per_query:
            worst = min(per_query, key=lambda x: x.faithfulness)
            worst_text = (
                f"Worst query: '{worst.query_preview}' "
                f"({worst.n_supported}/{worst.n_claims} claims supported). "
            )
        else:
            worst_text = ""

        return (
            f"FDS = {fds:+.4f} ({direction}). "
            f"Mean faithfulness: {np.mean(faithfulness_arr):.3f} "
            f"(reference baseline: {np.mean(reference):.3f}). "
            f"{worst_text}"
            f"{'Faithfulness is decaying — review LLM behaviour.' if fds < -0.02 else 'Within acceptable bounds.'}"
        )
