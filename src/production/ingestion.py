"""
GOV.UK document ingestion pipeline.
Fetches → parses → chunks → upserts to Pinecone with integrated embedding.
Every stage produces inspectable artefacts for auditability.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import yaml

from src.utils.config import (
    GOVUK_BASE_URL, PINECONE_API_KEY, CONFIG_DIR, DATA_DIR,
    IngestionConfig, load_config,
)
from src.utils.text_processing import (
    strip_html, split_by_headings, split_with_overlap, token_count,
    classify_topic,
)


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A parsed GOV.UK document part."""
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A text chunk ready for embedding/upserting."""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    token_count: int = 0


# ── Ingestion Pipeline ───────────────────────────────────────────────────────

class GOVUKIngester:
    """
    Fetches, parses, chunks, and upserts GOV.UK content to Pinecone.
    Uses Pinecone integrated embedding (llama-text-embed-v2, 1024-dim).
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or load_config().ingestion
        self._pc = None
        self._index = None

    @property
    def index(self):
        """Lazy-init Pinecone index."""
        if self._index is None:
            from pinecone import Pinecone
            self._pc = Pinecone(api_key=PINECONE_API_KEY)
            self._index = self._pc.Index(self.config.pinecone_index)
        return self._index

    def load_page_list(self) -> list[dict]:
        """Load the list of GOV.UK pages to ingest from config."""
        pages_path = CONFIG_DIR / "govuk_pages.yaml"
        with open(pages_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("pages", [])

    def fetch_page(self, path: str) -> dict:
        """Fetch structured JSON from GOV.UK Content API."""
        url = f"{GOVUK_BASE_URL}/api/content/{path}"
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        raw_json = response.json()

        # Save raw JSON for auditability
        raw_dir = DATA_DIR / "raw" / path.replace("/", "_")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with open(raw_dir / "raw.json", "w", encoding="utf-8") as f:
            json.dump(raw_json, f, indent=2, ensure_ascii=False)

        return raw_json

    def extract_parts(self, raw_json: dict, category: str = "") -> list[Document]:
        """Extract each 'part' as a separate Document with metadata."""
        documents: list[Document] = []
        base_path = raw_json.get("base_path", "")
        title = raw_json.get("title", "")
        updated = raw_json.get("public_updated_at", "")
        details = raw_json.get("details", {})

        parts = details.get("parts", [])

        if parts:
            for part in parts:
                body_html = part.get("body", "")
                content = strip_html(body_html)
                if not content.strip():
                    continue
                doc = Document(
                    content=content,
                    metadata={
                        "source": f"gov.uk{base_path}",
                        "part_title": part.get("title", ""),
                        "part_slug": part.get("slug", ""),
                        "page_title": title,
                        "last_updated": updated,
                        "licence": "OGL-v3.0",
                        "category": category or classify_topic(title),
                        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                documents.append(doc)
        else:
            # Single-part page — use body from details
            body_html = details.get("body", "")
            content = strip_html(body_html) if body_html else ""
            if content.strip():
                documents.append(Document(
                    content=content,
                    metadata={
                        "source": f"gov.uk{base_path}",
                        "part_title": title,
                        "part_slug": base_path.strip("/"),
                        "page_title": title,
                        "last_updated": updated,
                        "licence": "OGL-v3.0",
                        "category": category or classify_topic(title),
                        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                ))

        return documents

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Section-aware chunking with configurable max tokens and overlap.
        Each chunk retains full metadata lineage.
        """
        chunks: list[Chunk] = []

        for doc in documents:
            sections = split_by_headings(doc.content)

            for section in sections:
                if not section.strip():
                    continue
                tc = token_count(section)

                if tc <= self.config.chunk_max_tokens:
                    chunk_id = self._make_chunk_id(doc.metadata, section)
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=section,
                        metadata={**doc.metadata, "chunk_strategy": "section-boundary"},
                        token_count=tc,
                    ))
                else:
                    sub_chunks = split_with_overlap(
                        section,
                        max_tokens=self.config.chunk_max_tokens,
                        overlap=self.config.chunk_overlap_tokens,
                    )
                    for i, sc in enumerate(sub_chunks):
                        chunk_id = self._make_chunk_id(doc.metadata, sc, suffix=str(i))
                        chunks.append(Chunk(
                            chunk_id=chunk_id,
                            text=sc,
                            metadata={**doc.metadata, "chunk_strategy": "recursive-sentence"},
                            token_count=token_count(sc),
                        ))

        return chunks

    def upsert_to_pinecone(self, chunks: list[Chunk]) -> dict:
        """
        Upsert chunks to Pinecone with integrated embedding.
        Uses upsert_records() — Pinecone embeds via llama-text-embed-v2 server-side.
        The 'text' field is the field_mapping target for embedding.
        Returns upsert stats.
        """
        batch_size = self.config.pinecone_batch_size
        total_upserted = 0

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            records = []
            for chunk in batch:
                record = {
                    "_id": chunk.chunk_id,
                    "text": chunk.text,  # field_mapping target for embedding
                    "token_count": chunk.token_count,
                }
                # Flatten metadata into record (Pinecone records are flat dicts)
                for k, v in chunk.metadata.items():
                    if k != "text":  # avoid overwriting the text field
                        record[k] = v
                records.append(record)
            self.index.upsert_records(
                namespace=self.config.pinecone_namespace,
                records=records,
            )
            total_upserted += len(batch)
            # Small delay to respect rate limits
            time.sleep(0.5)

        stats = {
            "total_chunks": len(chunks),
            "total_upserted": total_upserted,
            "namespace": self.config.pinecone_namespace,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save upsert summary
        summary_path = DATA_DIR / "processed" / "pinecone_upsert_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return stats

    def run_full_ingestion(self) -> dict:
        """
        Run the complete ingestion pipeline:
        1. Load page list
        2. Fetch each page
        3. Extract parts
        4. Chunk
        5. Upsert to Pinecone
        
        Returns summary stats.
        """
        pages = self.load_page_list()
        all_chunks: list[Chunk] = []
        fetch_results: list[dict] = []

        for page_info in pages:
            path = page_info["path"]
            category = page_info.get("category", "")
            try:
                raw_json = self.fetch_page(path)
                documents = self.extract_parts(raw_json, category=category)
                chunks = self.chunk_documents(documents)
                all_chunks.extend(chunks)
                fetch_results.append({
                    "path": path,
                    "status": "ok",
                    "n_documents": len(documents),
                    "n_chunks": len(chunks),
                })
            except Exception as e:
                fetch_results.append({
                    "path": path,
                    "status": "error",
                    "error": str(e),
                })

        # Save chunk metadata (text is also stored in Pinecone)
        chunks_meta_path = DATA_DIR / "processed" / "chunks_metadata.json"
        with open(chunks_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"id": c.chunk_id, "tokens": c.token_count, **c.metadata}
                 for c in all_chunks],
                f, indent=2, ensure_ascii=False,
            )

        # Upsert to Pinecone
        upsert_stats = self.upsert_to_pinecone(all_chunks)

        summary = {
            "pages_attempted": len(pages),
            "pages_succeeded": sum(1 for r in fetch_results if r["status"] == "ok"),
            "total_chunks": len(all_chunks),
            "upsert_stats": upsert_stats,
            "per_page": fetch_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save full summary
        summary_path = DATA_DIR / "processed" / "ingestion_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    @staticmethod
    def _make_chunk_id(metadata: dict, text: str, suffix: str = "") -> str:
        """Generate a deterministic chunk ID from metadata + content hash."""
        source = metadata.get("source", "")
        slug = metadata.get("part_slug", "")
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        parts = [source.replace("/", "_"), slug, content_hash]
        if suffix:
            parts.append(suffix)
        return "_".join(parts)
