"""
LLM-as-Judge — claim verification and quality assessment via Groq.
Used by FDS for atomic claim decomposition + verification.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from src.utils.config import GROQ_API_KEY, LLMConfig, load_config
from src.utils.rate_limiter import rate_limited_completion


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ClaimVerdict:
    """Result of verifying a single claim."""
    claim: str
    verdict: str  # "supported" | "unsupported" | "ambiguous"
    confidence: float
    evidence_quote: str
    reasoning: str = ""


# ── Prompts ──────────────────────────────────────────────────────────────────

CLAIM_DECOMPOSITION_PROMPT = """Decompose the following response into a list of atomic, \
factually verifiable claims. Each claim should be a single sentence that \
can be independently verified against the source documents.

Response: {response}

Return ONLY a JSON object with a "claims" key containing an array of claim strings.
Example: {{"claims": ["Claim 1", "Claim 2"]}}"""

CLAIM_VERIFICATION_PROMPT = """You are a factual accuracy evaluator for a UK government services chatbot.

Your task: Determine whether the following CLAIM is supported by the provided CONTEXT.

CONTEXT (from official GOV.UK documents):
{context}

CLAIM to verify:
{claim}

Evaluate the claim and respond with exactly this JSON format:
{{
    "verdict": "supported" or "unsupported" or "ambiguous",
    "confidence": <float 0.0-1.0>,
    "evidence_quote": "<exact quote from context that supports or contradicts the claim, or N/A>",
    "reasoning": "<1-2 sentence explanation>"
}}

Rules:
- "supported": The claim is directly stated or logically entailed by the context.
- "unsupported": The claim contradicts the context OR makes a statement not present in the context.
- "ambiguous": The context is relevant but neither clearly supports nor contradicts the claim.
- Be strict: if the context says "savings over £16,000" and the claim says "savings over £15,000", that is "unsupported".
- Numerical thresholds, dates, and eligibility criteria must match exactly."""


# ── Judge Engine ─────────────────────────────────────────────────────────────

class JudgeEngine:
    """LLM-as-Judge for claim decomposition and verification."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_config().judge_llm
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    def decompose_claims(self, response: str) -> list[str]:
        """Decompose a response into atomic, verifiable claims."""
        prompt = CLAIM_DECOMPOSITION_PROMPT.format(response=response)

        try:
            result = rate_limited_completion(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )

            content = result.choices[0].message.content or "{}"
            parsed = json.loads(content)
            claims = parsed.get("claims", [])
            return claims if isinstance(claims, list) else []
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: split on sentences
            return [s.strip() for s in response.split(".") if s.strip() and len(s.strip()) > 10]

    def verify_claim(self, claim: str, context: str) -> ClaimVerdict:
        """Verify a single claim against provided context."""
        prompt = CLAIM_VERIFICATION_PROMPT.format(claim=claim, context=context)

        try:
            result = rate_limited_completion(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )

            content = result.choices[0].message.content or "{}"
            parsed = json.loads(content)

            return ClaimVerdict(
                claim=claim,
                verdict=parsed.get("verdict", "ambiguous"),
                confidence=float(parsed.get("confidence", 0.5)),
                evidence_quote=parsed.get("evidence_quote", "N/A"),
                reasoning=parsed.get("reasoning", ""),
            )
        except (json.JSONDecodeError, Exception):
            return ClaimVerdict(
                claim=claim,
                verdict="ambiguous",
                confidence=0.0,
                evidence_quote="N/A",
                reasoning="Judge evaluation failed",
            )


# ── Cross-Validation Engine ─────────────────────────────────────────────────


@dataclass
class CrossValidationResult:
    """Result of cross-validating two judges on the same claims."""
    n_claims: int
    n_agree: int
    n_disagree: int
    agreement_rate: float
    cohens_kappa: float
    disagreements: list[dict]


class CrossValidator:
    """
    Cross-validate two LLM judges on the same set of claims.

    Computes:
    - Raw agreement rate = n_agree / n_total
    - Cohen's kappa = (p_o - p_e) / (1 - p_e)
      where p_o = observed agreement, p_e = expected agreement by chance

    A kappa > 0.6 indicates substantial agreement (Landis & Koch, 1977).
    A kappa < 0.4 suggests the judge is unreliable and verdicts should be
    treated with caution.
    """

    def __init__(
        self,
        primary: Optional[JudgeEngine] = None,
        secondary: Optional[JudgeEngine] = None,
    ):
        cfg = load_config()
        self.primary = primary or JudgeEngine(cfg.judge_llm)
        # Use the primary LLM as secondary judge (different model perspective)
        # In practice, you'd use a distinct model; here we use the 70B as cross-check
        self.secondary = secondary or JudgeEngine(cfg.primary_llm)

    def cross_validate(
        self,
        claims: list[str],
        context: str,
    ) -> CrossValidationResult:
        """
        Run both judges on each claim and compute agreement metrics.

        Args:
            claims: List of atomic claim strings
            context: Source context to verify against

        Returns:
            CrossValidationResult with agreement stats and Cohen's kappa
        """
        if not claims:
            return CrossValidationResult(
                n_claims=0, n_agree=0, n_disagree=0,
                agreement_rate=1.0, cohens_kappa=1.0, disagreements=[],
            )

        primary_verdicts: list[str] = []
        secondary_verdicts: list[str] = []
        disagreements: list[dict] = []

        for claim in claims:
            v1 = self.primary.verify_claim(claim, context)
            v2 = self.secondary.verify_claim(claim, context)
            primary_verdicts.append(v1.verdict)
            secondary_verdicts.append(v2.verdict)

            if v1.verdict != v2.verdict:
                disagreements.append({
                    "claim": claim,
                    "primary_verdict": v1.verdict,
                    "secondary_verdict": v2.verdict,
                    "primary_confidence": v1.confidence,
                    "secondary_confidence": v2.confidence,
                })

        n = len(claims)
        n_agree = sum(1 for a, b in zip(primary_verdicts, secondary_verdicts) if a == b)
        agreement_rate = n_agree / n

        # Cohen's kappa
        kappa = self._cohens_kappa(primary_verdicts, secondary_verdicts)

        return CrossValidationResult(
            n_claims=n,
            n_agree=n_agree,
            n_disagree=n - n_agree,
            agreement_rate=agreement_rate,
            cohens_kappa=kappa,
            disagreements=disagreements,
        )

    @staticmethod
    def _cohens_kappa(a: list[str], b: list[str]) -> float:
        """
        Compute Cohen's kappa for two lists of categorical labels.

        κ = (p_o - p_e) / (1 - p_e)

        where:
          p_o = observed proportion of agreement
          p_e = expected proportion of agreement under independence

        Categories: supported, unsupported, ambiguous
        """
        n = len(a)
        if n == 0:
            return 1.0

        categories = ["supported", "unsupported", "ambiguous"]

        # Observed agreement
        p_o = sum(1 for x, y in zip(a, b) if x == y) / n

        # Expected agreement by chance
        p_e = 0.0
        for cat in categories:
            p_a = sum(1 for x in a if x == cat) / n
            p_b = sum(1 for x in b if x == cat) / n
            p_e += p_a * p_b

        if p_e >= 1.0:
            return 1.0  # Perfect expected agreement (degenerate case)

        return (p_o - p_e) / (1.0 - p_e)
