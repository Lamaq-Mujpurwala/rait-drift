"""
Text processing utilities for RAG pipeline and descriptor extraction.
Covers: HTML stripping, tokenisation, readability, sentiment, hedge words,
refusal detection, citation counting, and topic classification.
"""

from __future__ import annotations

import re
from typing import Optional

import tiktoken
import textstat
from bs4 import BeautifulSoup


# ── Tokenisation ─────────────────────────────────────────────────────────────

# Use cl100k_base (GPT-4 / Llama-compatible) for consistent counting
_ENCODING: Optional[tiktoken.Encoding] = None


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def token_count(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(_get_encoding().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens."""
    enc = _get_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


# ── HTML Processing ──────────────────────────────────────────────────────────

def strip_html(html: str) -> str:
    """Strip HTML tags and normalise whitespace. Preserve paragraph breaks."""
    soup = BeautifulSoup(html, "html.parser")
    # Replace block-level elements with newlines
    for tag in soup.find_all(["p", "br", "h1", "h2", "h3", "h4", "li"]):
        tag.insert_before("\n")
    text = soup.get_text()
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_by_headings(text: str) -> list[str]:
    """
    Split text at heading boundaries (lines starting with #, or
    all-caps lines, or lines followed by === or ---).
    Returns list of sections.
    """
    # Simple heuristic: split on lines that look like headings
    lines = text.split("\n")
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Detect heading patterns
        is_heading = (
            stripped.startswith("#") or
            (len(stripped) > 3 and stripped.isupper() and not stripped.isdigit())
        )
        if is_heading and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return [s for s in sections if s.strip()]


def split_with_overlap(text: str, max_tokens: int = 1024, overlap: int = 128) -> list[str]:
    """
    Split text into chunks of ≤ max_tokens with overlap.
    Splits at sentence boundaries when possible.
    """
    enc = _get_encoding()
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current_tokens: list[str] = []
    current_count = 0

    for sentence in sentences:
        s_tokens = enc.encode(sentence)
        s_count = len(s_tokens)

        if current_count + s_count > max_tokens and current_tokens:
            # Emit current chunk
            chunk_text = " ".join(current_tokens)
            chunks.append(chunk_text)

            # Build overlap from the tail of current_tokens
            overlap_tokens: list[str] = []
            overlap_count = 0
            for prev in reversed(current_tokens):
                prev_count = len(enc.encode(prev))
                if overlap_count + prev_count > overlap:
                    break
                overlap_tokens.insert(0, prev)
                overlap_count += prev_count

            current_tokens = overlap_tokens + [sentence]
            current_count = overlap_count + s_count
        else:
            current_tokens.append(sentence)
            current_count += s_count

    if current_tokens:
        chunks.append(" ".join(current_tokens))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Basic sentence splitting."""
    # Split on sentence-ending punctuation followed by space or newline
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ── Query Preprocessing ──────────────────────────────────────────────────────

def preprocess_query(query: str) -> str:
    """Clean and normalise a user query."""
    text = query.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ── Descriptor Extraction Functions ──────────────────────────────────────────

HEDGE_WORDS = {
    "might", "could", "possibly", "perhaps", "may", "likely", "unlikely",
    "probably", "sometimes", "generally", "usually", "often", "typically",
    "approximately", "around", "about", "roughly", "estimated", "suggest",
    "appears", "seems", "indicates",
}

REFUSAL_PATTERNS = [
    r"i don't have enough information",
    r"i cannot (provide|answer|help)",
    r"i'm (not able|unable) to",
    r"please visit gov\.uk",
    r"contact your local council",
    r"i can't (provide|answer|help)",
    r"outside (my|the) scope",
    r"i don't have (the |any )?(relevant |specific )?information",
]


def compute_sentiment(text: str) -> float:
    """
    Simple rule-based sentiment score in [-1, 1].
    Positive = helpful/affirming, Negative = refusal/uncertainty.
    For a gov chatbot, most responses should be neutral-positive.
    """
    positive_words = {
        "eligible", "entitled", "can", "yes", "available", "free",
        "help", "support", "apply", "receive", "get", "benefit",
    }
    negative_words = {
        "cannot", "not", "ineligible", "denied", "refused", "unfortunately",
        "unable", "don't", "won't", "isn't", "aren't",
    }
    words = text.lower().split()
    if not words:
        return 0.0
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def compute_readability(text: str) -> float:
    """Flesch-Kincaid reading ease score. Higher = easier to read."""
    if not text.strip():
        return 0.0
    return float(textstat.flesch_reading_ease(text))


def count_citations(text: str) -> int:
    """Count GOV.UK citations in a response."""
    patterns = [
        r"gov\.uk",
        r"GOV\.UK",
        r"\[Source:",
        r"www\.gov\.uk",
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text))
    return count


def count_hedge_words(text: str) -> int:
    """Count hedge/uncertainty words."""
    words = text.lower().split()
    return sum(1 for w in words if w in HEDGE_WORDS)


def detect_refusal(text: str) -> bool:
    """Detect if a response is a refusal / out-of-scope response."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in REFUSAL_PATTERNS)


def classify_topic(text: str) -> str:
    """
    Classify text into one of the DDI topic segments.
    Uses keyword matching against the topic segment definitions.
    """
    topic_keywords: dict[str, list[str]] = {
        "universal_credit": ["universal credit", "uc", "work coach", "claimant commitment"],
        "housing_benefit": ["housing benefit", "rent", "spare bedroom", "lha", "council housing"],
        "disability_benefits": ["pip", "disability", "attendance allowance", "dla", "esa"],
        "council_tax": ["council tax", "band", "council tax reduction", "discount"],
        "homelessness": ["homeless", "emergency housing", "temporary accommodation", "evict"],
        "pension": ["pension", "retirement", "state pension age", "pension credit"],
    }
    text_lower = text.lower()
    best_topic = "unknown"
    best_count = 0
    for topic, keywords in topic_keywords.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_topic = topic
    return best_topic
