"""
Configuration management for RAIT system.
Loads defaults from YAML, environment variables from .env, and provides
a unified config interface used by all modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Load .env (if present) — override system env vars to ensure .env takes priority
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ── Environment variables ────────────────────────────────────────────────────
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GOVUK_BASE_URL: str = os.getenv("GOVUK_BASE_URL", "https://www.gov.uk")


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass
class IngestionConfig:
    chunk_max_tokens: int = 1024
    chunk_overlap_tokens: int = 128
    pinecone_batch_size: int = 90
    pinecone_namespace: str = "govuk"
    pinecone_index: str = "rait-chatbot"


@dataclass
class RetrievalConfig:
    top_k: int = 10
    context_window: int = 5


@dataclass
class TRCIConfig:
    canary_path: str = "config/canary_queries.json"
    probe_frequency: str = "daily"
    similarity_metric: str = "cosine"
    green_threshold: float = 0.95
    amber_threshold: float = 0.90
    red_p10_threshold: float = 0.80
    canary_count: int = 50


@dataclass
class CDSDescriptorConfig:
    weight: float
    type: str  # "continuous" | "discrete" | "binary"
    enabled: bool = True


@dataclass
class CDSConfig:
    reference_window_days: int = 30
    current_window_days: int = 7
    jsd_bins: int | str = "auto"  # "auto" = Freedman-Diaconis adaptive; int = fixed
    persistence_threshold: int = 3
    green_threshold: float = 0.05
    amber_threshold: float = 0.15
    red_threshold: float = 0.30
    normalise_weights: bool = True
    descriptors: dict[str, CDSDescriptorConfig] = field(default_factory=dict)


@dataclass
class FDSConfig:
    sample_size: int = 50
    sampling_strategy: str = "stratified"
    verification_strictness: str = "moderate"
    include_ambiguous: bool = False
    signed_jsd_bins: int | str = "auto"  # "auto" = Freedman-Diaconis adaptive; int = fixed
    green_threshold: float = 0.02
    amber_threshold: float = 0.10
    red_threshold: float = 0.20
    calibration_enabled: bool = True
    calibration_rho_threshold: float = 0.6
    cross_validation_enabled: bool = False  # Enable second-judge cross-validation
    cross_validation_kappa_warn: float = 0.4  # Warn if kappa below this


@dataclass
class DDIConfig:
    min_segment_size: int = 20
    quality_proxy_weights: dict[str, float] = field(
        default_factory=lambda: {
            "completeness": 0.4,
            "citation": 0.3,
            "non_refusal": 0.2,
            "latency": 0.1,
        }
    )
    formula: str = "std"
    intersectional_threshold: float = 0.20
    green_threshold: float = 0.05
    amber_threshold: float = 0.15
    red_threshold: float = 0.30


@dataclass
class AppConfig:
    """Unified application configuration."""

    # LLM
    primary_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig("groq/llama-3.3-70b-versatile", 0.1, 1024)
    )
    judge_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig("groq/llama-3.1-8b-instant", 0.0, 512)
    )
    fallback_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig("groq/llama-3.1-8b-instant", 0.1, 1024)
    )

    # Subsystems
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    trci: TRCIConfig = field(default_factory=TRCIConfig)
    cds: CDSConfig = field(default_factory=CDSConfig)
    fds: FDSConfig = field(default_factory=FDSConfig)
    ddi: DDIConfig = field(default_factory=DDIConfig)

    # Paths
    db_path: Path = field(default_factory=lambda: DATA_DIR / "logs" / "production.db")


def load_config(overrides: dict[str, Any] | None = None) -> AppConfig:
    """
    Load configuration from defaults.yaml, apply optional overrides.
    Returns a fully-populated AppConfig.
    """
    defaults_path = CONFIG_DIR / "defaults.yaml"
    raw: dict[str, Any] = {}
    if defaults_path.exists():
        raw = _load_yaml(defaults_path)
    if overrides:
        raw = _deep_merge(raw, overrides)

    # Build LLM configs
    llm_raw = raw.get("llm", {})
    primary_llm = LLMConfig(**llm_raw.get("primary", {
        "model": "groq/llama-3.3-70b-versatile",
        "temperature": 0.1,
        "max_tokens": 1024,
    }))
    judge_llm = LLMConfig(**llm_raw.get("judge", {
        "model": "groq/llama-3.1-8b-instant",
        "temperature": 0.0,
        "max_tokens": 512,
    }))
    fallback_llm = LLMConfig(**llm_raw.get("fallback", {
        "model": "groq/llama-3.1-8b-instant",
        "temperature": 0.1,
        "max_tokens": 1024,
    }))

    # Ingestion
    ing_raw = raw.get("ingestion", {})
    ingestion = IngestionConfig(**ing_raw) if ing_raw else IngestionConfig()

    # Retrieval
    ret_raw = raw.get("retrieval", {})
    retrieval = RetrievalConfig(**ret_raw) if ret_raw else RetrievalConfig()

    # TRCI
    trci_raw = raw.get("trci", {})
    trci = TRCIConfig(**trci_raw) if trci_raw else TRCIConfig()

    # CDS
    cds_raw = raw.get("cds", {})
    desc_raw = cds_raw.pop("descriptors", {})
    descriptors = {
        k: CDSDescriptorConfig(**v) for k, v in desc_raw.items()
    }
    cds = CDSConfig(**cds_raw, descriptors=descriptors) if cds_raw else CDSConfig()

    # FDS
    fds_raw = raw.get("fds", {})
    # remove decomposition_model since it's not in the dataclass
    fds_raw.pop("decomposition_model", None)
    fds = FDSConfig(**fds_raw) if fds_raw else FDSConfig()

    # DDI
    ddi_raw = raw.get("ddi", {})
    ddi = DDIConfig(**ddi_raw) if ddi_raw else DDIConfig()

    return AppConfig(
        primary_llm=primary_llm,
        judge_llm=judge_llm,
        fallback_llm=fallback_llm,
        ingestion=ingestion,
        retrieval=retrieval,
        trci=trci,
        cds=cds,
        fds=fds,
        ddi=ddi,
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
