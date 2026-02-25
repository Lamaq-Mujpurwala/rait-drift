# System Design Document — Data & Model Drift Monitoring for UK Public Sector RAG Chatbot

**Author:** Lamaq  
**Assessment:** RAIT Intern — Ethical Dimension: Data & Model Drift  
**Compliance Question:** *"How does the organisation detect and respond to changes in data or model behaviour over time to ensure the system remains valid, unbiased, and fit for purpose?"*  
**Date:** February 2026  
**Status:** Complete System Architecture

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Part A — Production RAG+Chatbot System](#2-part-a--production-ragchatbot-system)
3. [Part B — Metric Monitoring System](#3-part-b--metric-monitoring-system)
4. [Toggle-Friendly Experimentation Dashboard](#4-toggle-friendly-experimentation-dashboard)
5. [LLM-as-Judge Design](#5-llm-as-judge-design)
6. [Publicly Available Data & Datasets](#6-publicly-available-data--datasets)
7. [Testing Strategies Per Metric](#7-testing-strategies-per-metric)
8. [Descriptive Indicators & Charts](#8-descriptive-indicators--charts)
9. [Explainability & Export Framework](#9-explainability--export-framework)
10. [Implementation Plan & Dependencies](#10-implementation-plan--dependencies)
11. [Appendices](#11-appendices)

---

## 1. System Overview

### 1.1 Architecture Philosophy

The system is designed as **two distinct but coupled subsystems**:

| Subsystem | Purpose | Users | Cadence |
|---|---|---|---|
| **Part A — Production System** | RAG-based chatbot answering citizen queries about UK housing, benefits, and local services | Citizens, council staff | Real-time (per query) |
| **Part B — Monitoring System** | Detects and quantifies data/model drift using four custom metrics | Governance officers, ML engineers, auditors | Batch (daily/weekly) + on-demand |

Both systems expose **every intermediate computation** — no black boxes. Every pipeline stage produces inspectable artefacts.

### 1.2 End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PART A: PRODUCTION SYSTEM                           │
│                                                                            │
│  Citizen Query                                                             │
│       │                                                                    │
│       ▼                                                                    │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌───────────────┐   │
│  │ Query    │───▶│ Embedding    │───▶│ Vector     │───▶│ Re-ranking    │   │
│  │ Intake & │    │ (Pinecone    │    │ Search     │    │ + Context     │   │
│  │ Logging  │    │  integrated) │    │ (Pinecone) │    │ Assembly      │   │
│  └──────────┘    └──────────────┘    └────────────┘    └───────┬───────┘   │
│                                                                │           │
│                                                                ▼           │
│                                          ┌──────────────────────────────┐  │
│                                          │ LLM API Call                 │  │
│                                          │ (prompt + retrieved context) │  │
│                                          └──────────────┬───────────────┘  │
│                                                         │                  │
│                                                         ▼                  │
│                                          ┌──────────────────────────────┐  │
│                                          │ Response Post-Processing     │  │
│                                          │ + Mandatory Disclaimers      │  │
│                                          └──────────────┬───────────────┘  │
│                                                         │                  │
│            ┌────────────────────────┐                   │                  │
│            │    PRODUCTION LOG      │◀──────────────────┘                  │
│            │  (every intermediate)  │                                      │
│            └────────────┬───────────┘                                      │
│                         │                                                  │
└─────────────────────────┼──────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PART B: MONITORING SYSTEM                            │
│                                                                            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────┐   │
│  │ Data         │──▶│ Descriptor   │──▶│ Metric       │──▶│ Dashboard │   │
│  │ Collection   │   │ Extraction   │   │ Computation  │   │ + Alerts  │   │
│  │ Layer        │   │ Engine       │   │ Engine       │   │ + Export  │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   └───────────┘   │
│                                                                            │
│  Metrics: TRCI | CDS | FDS | DDI                                           │
│  Cadence: Daily canaries, Weekly batch, Quarterly governance reports       │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Design Constraints (from Assignment)

| Constraint | Implication |
|---|---|
| **Available:** User queries, LLM responses, Model API (client only), Partial ground truth, System performance metrics | All metrics must derive from these 5 sources only |
| **Unavailable:** RAG-specific logs, model internals, user demographics, additional data | Cannot inspect attention weights, retrieval scores, or user protected characteristics directly |
| **Model access:** API-only (black-box) | No gradient-based drift detection; must use output-distribution methods |
| **Ground truth:** Partial only | Cannot rely on full supervised evaluation; must design self-calibrating metrics |

### 1.4 Technology Stack

| Component | Choice | Justification |
|---|---|---|
| **Language** | Python ≥ 3.11 | Project requirement (pyproject.toml) |
| **Package manager** | uv | Project requirement |
| **Embedding model** | `llama-text-embed-v2` (NVIDIA, via Pinecone integrated embedding) | 1024-dim, cosine, 2048 max input tokens; managed server-side by Pinecone — no local GPU dependency |
| **Vector store** | Pinecone (Serverless) | Cloud-managed, index `rait-chatbot`, cosine metric, integrated embedding; persistent with no infrastructure overhead |
| **LLM API** | Groq API (Llama 3.3 70B primary, Llama 3.1 8B for Judge) | Free tier, OpenAI-compatible, ~280 tps; mirrors UK gov use of OpenAI API pattern; interchangeable via `litellm` |
| **Dashboard** | Streamlit | Rapid prototyping, native toggle/slider widgets, Python-native, exportable charts |
| **Statistical compute** | NumPy, SciPy | JSD, cosine similarity, KS tests |
| **NLP utilities** | spaCy (topic extraction), tiktoken (tokenisation) | Lightweight, deterministic |
| **Data storage** | SQLite + JSON logs | Lightweight, portable, zero-config — appropriate for assessment scope |
| **Visualisation** | Plotly (interactive) + Matplotlib (static export) | Plotly for dashboard interactivity; Matplotlib for PDF reports |
| **Testing** | pytest + hypothesis (property-based) | Thorough coverage with generative edge cases |

---

## 2. Part A — Production RAG+Chatbot System

### 2.1 Document Corpus Design

#### 2.1.1 Data Source: GOV.UK Content API

All documents are sourced from the **GOV.UK Content API**, which provides structured JSON at:
```
https://www.gov.uk/api/content/{path}
```

**Licence:** Open Government Licence v3.0 — freely reusable, no registration required.

**Confirmed API structure** (from live inspection of `/api/content/housing-benefit`):
```json
{
  "base_path": "/housing-benefit",
  "title": "Housing Benefit",
  "description": "Housing Benefit or Local Housing Allowance (LHA)...",
  "details": {
    "parts": [
      {
        "title": "Eligibility",
        "slug": "eligibility",
        "body": "<p>Housing Benefit can help you pay your rent...</p>"
      },
      {
        "title": "What you'll get",
        "slug": "what-youll-get",
        "body": "..."
      }
    ]
  },
  "public_updated_at": "2024-10-17T14:05:33Z"
}
```

#### 2.1.2 Document Scope

| Category | GOV.UK Pages | Content Type | Estimated Chunks |
|---|---|---|---|
| **Benefits — Universal Credit** | `/universal-credit` (8 parts), `/apply-universal-credit` | Multi-part guide | ~40 chunks |
| **Benefits — Housing Benefit** | `/housing-benefit` (6 parts) | Multi-part guide | ~30 chunks |
| **Benefits — Council Tax** | `/council-tax` (multiple parts) | Multi-part guide | ~25 chunks |
| **Benefits — Disability** | `/pip`, `/attendance-allowance`, `/disability-living-allowance` | Guides | ~35 chunks |
| **Housing — Local services** | `/council-housing`, `/apply-for-council-housing`, `/right-to-buy-buying-your-council-home` | Guides + transactions | ~20 chunks |
| **Housing — Renting** | `/private-renting`, `/housing-benefit/what-youll-get` | Guides | ~15 chunks |
| **Housing — Homelessness** | `/homelessness-help-from-council`, `/if-youre-homeless-at-risk-of-homelessness` | Guides | ~15 chunks |
| **Local Services** | `/find-local-council`, `/report-problem-council`, `/council-tax-bands` | Answers + transactions | ~10 chunks |
| **Cross-cutting** | `/benefits-calculators`, `/benefit-cap`, `/benefit-overpayments` | Answers | ~10 chunks |
| **Total** | **~25 GOV.UK pages** | | **~200 chunks** |

#### 2.1.3 Document Ingestion Pipeline

```python
# Pseudocode — Document Ingestion
class GOVUKDocumentIngester:
    """
    Fetches, parses, chunks, and embeds GOV.UK content.
    Every stage produces inspectable artefacts.
    """
    
    PAGES = [
        "universal-credit", "apply-universal-credit",
        "housing-benefit", "council-tax", "pip",
        "attendance-allowance", "council-housing",
        "right-to-buy-buying-your-council-home",
        "homelessness-help-from-council",
        "find-local-council", "benefits-calculators",
        "benefit-cap", "benefit-overpayments",
        # ... full list in config
    ]
    
    def fetch_page(self, path: str) -> dict:
        """Fetch structured JSON from GOV.UK Content API."""
        response = httpx.get(f"https://www.gov.uk/api/content/{path}")
        raw_json = response.json()
        # LOG: raw_json → logs/ingestion/{path}/raw.json
        return raw_json
    
    def extract_parts(self, raw_json: dict) -> list[Document]:
        """Extract each 'part' as a separate document with metadata."""
        documents = []
        for part in raw_json["details"].get("parts", []):
            doc = Document(
                content=strip_html(part["body"]),
                metadata={
                    "source": f"gov.uk/{raw_json['base_path']}",
                    "part_title": part["title"],
                    "part_slug": part["slug"],
                    "page_title": raw_json["title"],
                    "last_updated": raw_json["public_updated_at"],
                    "licence": "OGL-v3.0",
                    "category": classify_category(raw_json["base_path"]),
                    "ingestion_timestamp": datetime.utcnow().isoformat()
                }
            )
            documents.append(doc)
        # LOG: documents → logs/ingestion/{path}/extracted.json
        return documents
    
    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Chunking strategy: SECTION-AWARE chunking.
        - Preserve GOV.UK's own section boundaries (h2/h3 headings)
        - Maximum chunk size: 1024 tokens (utilises llama-text-embed-v2's 2048-token window)
        - Overlap: 128 tokens (for continuity — proportional to larger chunks)
        - Each chunk retains full metadata lineage
        
        NOTE: llama-text-embed-v2 supports up to 2048 input tokens, so we use
        1024 as max chunk size to leave headroom for metadata/instructions while
        capturing more context per chunk than the previous 512-token limit.
        """
        chunks = []
        for doc in documents:
            sections = split_by_headings(doc.content)
            for section in sections:
                if token_count(section) <= 1024:
                    chunks.append(Chunk(
                        text=section,
                        metadata={**doc.metadata, "chunk_strategy": "section-boundary"},
                        token_count=token_count(section)
                    ))
                else:
                    # Recursive split at sentence boundaries
                    sub_chunks = split_with_overlap(section, max_tokens=1024, overlap=128)
                    for sc in sub_chunks:
                        chunks.append(Chunk(
                            text=sc,
                            metadata={**doc.metadata, "chunk_strategy": "recursive-sentence"},
                            token_count=token_count(sc)
                        ))
        # LOG: chunks → logs/ingestion/{path}/chunks.json
        return chunks
    
    def upsert_to_pinecone(self, chunks: list[Chunk]) -> None:
        """
        Upsert chunks to Pinecone with integrated embedding.
        
        Pinecone manages embedding via llama-text-embed-v2 (NVIDIA hosted):
        - 1024 dimensions, cosine similarity
        - Server-side embedding: we send raw text, Pinecone embeds it
        - No local GPU or sentence-transformers dependency
        """
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index("rait-chatbot")
        
        # Batch upsert (Pinecone recommends batches of 100)
        for batch_start in range(0, len(chunks), 100):
            batch = chunks[batch_start:batch_start + 100]
            index.upsert(
                vectors=[
                    {
                        "id": chunk.chunk_id,
                        "values": None,  # Pinecone handles embedding via integrated model
                        "metadata": {
                            **chunk.metadata,
                            "text": chunk.text,  # Store text in metadata for retrieval
                            "token_count": chunk.token_count,
                        },
                    }
                    for chunk in batch
                ],
                namespace="govuk",
            )
        # LOG: upsert_stats → logs/ingestion/pinecone_upsert_summary.json
```

**Dissectability:** Every stage (fetch → extract → chunk → embed) writes a log artefact. An auditor can inspect exactly what was fetched, how it was split, and what metadata was attached.

#### 2.1.4 Document Update Detection

GOV.UK Content API includes `public_updated_at` timestamps. The ingestion pipeline runs on a **configurable schedule** (default: weekly) and:

1. Compares `public_updated_at` with the stored `ingestion_timestamp`
2. If a page has been updated → re-ingests that page only
3. Logs the diff (old chunks removed, new chunks added)
4. Triggers a **TRCI canary re-evaluation** to detect if the knowledge base change affects response consistency

This is critical for **knowledge/document drift** detection — Requirement 3.3 of the UK AI White Paper (accuracy).

### 2.2 Retrieval Pipeline

```python
class RetrievalPipeline:
    """
    Query → Pinecone Search (with integrated embedding) → Re-ranking → Context Assembly.
    Every stage is logged for monitoring system consumption.
    
    Pinecone's integrated embedding handles query vectorisation server-side
    using llama-text-embed-v2 (1024-dim, cosine). We send raw text queries.
    """
    
    def __init__(self):
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = pc.Index("rait-chatbot")
    
    def process_query(self, user_query: str) -> RetrievalResult:
        # Stage 1: Query preprocessing
        cleaned_query = preprocess(user_query)  # lowercase, strip, normalise
        
        # Stage 2+3: Query embedding + Vector search (Pinecone integrated)
        # Pinecone embeds the query server-side via llama-text-embed-v2
        results = self.index.query(
            namespace="govuk",
            top_k=self.config.top_k,  # default: 10
            include_metadata=True,
            # Pinecone integrated embedding: pass raw text, not vectors
            data=cleaned_query,
        )
        
        # Stage 4: Re-ranking (cross-encoder or heuristic)
        # Option A: Cross-encoder re-ranking (if budget allows)
        # Option B: Metadata-boosted re-ranking (recency + category match)
        reranked = self.rerank(results.matches, user_query)
        
        # Stage 5: Context assembly (top-n after reranking)
        context_chunks = reranked[:self.config.context_window]  # default: 5
        assembled_context = "\n\n---\n\n".join([
            f"[Source: {c.metadata['source']} | Part: {c.metadata['part_title']} | "
            f"Updated: {c.metadata['last_updated']}]\n{c.metadata['text']}"
            for c in context_chunks
        ])
        
        # LOG: Full retrieval trace
        retrieval_log = RetrievalLog(
            query_id=generate_id(),
            timestamp=datetime.utcnow(),
            raw_query=user_query,
            cleaned_query=cleaned_query,
            top_k_results=[{
                "chunk_id": r.id,
                "score": r.score,
                "source": r.metadata["source"],
                "part": r.metadata["part_title"]
            } for r in results.matches],
            reranked_order=[r.id for r in reranked],
            selected_context_ids=[c.id for c in context_chunks],
            context_token_count=token_count(assembled_context)
        )
        self.log_store.write(retrieval_log)
        
        return RetrievalResult(
            context=assembled_context,
            retrieval_log=retrieval_log
        )
```

### 2.3 LLM Response Generation

```python
class ResponseGenerator:
    """
    Assembles prompt, calls LLM API, post-processes response.
    """
    
    SYSTEM_PROMPT = """You are a UK government services assistant. You help citizens 
    understand housing, benefits, and local council services based ONLY on the 
    official GOV.UK information provided below.
    
    RULES:
    1. Only answer based on the provided context. If the context doesn't contain 
       the answer, say "I don't have enough information to answer that. Please 
       visit GOV.UK or contact your local council."
    2. Always cite which GOV.UK page your answer comes from.
    3. Include relevant eligibility criteria, thresholds, and deadlines.
    4. Use plain English (reading age 9, per GOV.UK style guide).
    5. If the question involves financial thresholds or dates, state them explicitly.
    6. Never provide personal financial or legal advice.
    """
    
    DISCLAIMER = (
        "\n\n---\n*This response is generated by an AI assistant using official "
        "GOV.UK information. It is not personal advice. For definitive guidance, "
        "visit [GOV.UK](https://www.gov.uk) or contact your local council. "
        "Information may have changed since the source was last updated.*"
    )
    
    def generate(self, query: str, retrieval_result: RetrievalResult) -> ResponseResult:
        # Assemble prompt
        prompt = f"""Context from GOV.UK (official UK government information):

{retrieval_result.context}

---

Citizen's question: {query}

Please provide a helpful, accurate answer based only on the context above."""
        
        # Call LLM API
        api_response = self.llm_client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,  # default: 0.1 (low for factual)
            max_tokens=self.config.max_tokens,     # default: 1024
        )
        
        raw_response = api_response.choices[0].message.content
        
        # Post-process: add disclaimer
        final_response = raw_response + self.DISCLAIMER
        
        # LOG: Full generation trace
        generation_log = GenerationLog(
            query_id=retrieval_result.retrieval_log.query_id,
            timestamp=datetime.utcnow(),
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            prompt_token_count=api_response.usage.prompt_tokens,
            completion_token_count=api_response.usage.completion_tokens,
            raw_response=raw_response,
            final_response=final_response,
            finish_reason=api_response.choices[0].finish_reason,
            retrieval_context_ids=retrieval_result.retrieval_log.selected_context_ids
        )
        self.log_store.write(generation_log)
        
        return ResponseResult(
            response=final_response,
            generation_log=generation_log
        )
```

### 2.4 Production Logging Schema

Every query produces a **unified production log entry** stored in SQLite:

```sql
CREATE TABLE production_logs (
    query_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    
    -- Query data
    raw_query TEXT NOT NULL,
    cleaned_query TEXT,
    query_topic TEXT,              -- Extracted by spaCy for DDI
    query_token_count INTEGER,
    
    -- Retrieval data
    top_k_chunk_ids TEXT,          -- JSON array
    top_k_scores TEXT,             -- JSON array of floats (Pinecone cosine similarity scores)
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
    
    -- Derived descriptors (computed asynchronously for CDS)
    response_length INTEGER,
    response_sentiment REAL,
    response_readability REAL,
    citation_count INTEGER,
    hedge_word_count INTEGER,
    refusal_flag BOOLEAN,
    
    -- Metadata
    session_id TEXT,
    response_latency_ms INTEGER
);
```

This log is the **single source of truth** consumed by Part B.

### 2.5 UX Design (per UK Governance Requirements)

| Requirement | UX Implementation |
|---|---|
| **ICO Transparency** | Every response includes: (a) AI-generated disclaimer, (b) source citations with GOV.UK links, (c) last-updated date of source |
| **AI White Paper — Explainability** | "Why did I get this answer?" button exposes: retrieved chunks, confidence indicators, source freshness |
| **GOV.UK Style Guide** | Plain English, reading age 9, no jargon, active voice |
| **Equality Act — Accessibility** | WCAG 2.1 AA compliance; screen-reader compatible; keyboard navigable |
| **DPA 2018 — No automated decisions** | Chatbot states it provides information only, not decisions; directs to human contact for claims |
| **Ethics Framework — Human escalation** | "Speak to a person" link always visible; auto-suggest if query is complex/sensitive |

---

## 3. Part B — Metric Monitoring System

### 3.1 Data Collection Layer

The monitoring system reads from the production log (Section 2.4) and produces metric-ready datasets:

```python
class DataCollectionLayer:
    """
    Taps into production logs and prepares data for metric computation.
    Runs as a batch job (configurable: hourly/daily/weekly).
    """
    
    def collect_window(self, window_start: datetime, window_end: datetime) -> MetricDataset:
        """Collect all production logs in the given time window."""
        logs = self.db.query(
            "SELECT * FROM production_logs WHERE timestamp BETWEEN ? AND ?",
            (window_start.isoformat(), window_end.isoformat())
        )
        
        dataset = MetricDataset(
            window_start=window_start,
            window_end=window_end,
            n_queries=len(logs),
            logs=logs
        )
        
        # Enrich with descriptor extraction
        dataset.descriptors = self.extract_descriptors(logs)
        
        return dataset
    
    def extract_descriptors(self, logs: list[ProductionLog]) -> DescriptorSet:
        """
        Extract the K text descriptors needed by CDS.
        Each descriptor is a function: log_entry → scalar value.
        """
        descriptors = {}
        for log in logs:
            descriptors[log.query_id] = {
                # Query-side descriptors
                "query_length": len(log.raw_query.split()),
                "query_topic": log.query_topic,
                
                # Response-side descriptors
                "response_length": len(log.raw_response.split()),
                "response_sentiment": compute_sentiment(log.raw_response),
                "response_readability": compute_flesch_kincaid(log.raw_response),
                "citation_count": count_govuk_citations(log.raw_response),
                "hedge_word_count": count_hedge_words(log.raw_response),
                "refusal_flag": detect_refusal(log.raw_response),
                "response_latency_ms": log.response_latency_ms,
                
                # Retrieval-side descriptors
                "mean_retrieval_distance": np.mean(json.loads(log.top_k_scores)),
                "context_token_ratio": log.context_token_count / max(log.completion_tokens, 1),
            }
        return DescriptorSet(descriptors)
```

### 3.2 Metric Computation Engine

#### 3.2.1 TRCI — Temporal Response Consistency Index

```python
class TRCIEngine:
    """
    Active canary probing to detect model/system changes.
    
    Mechanism:
    - Maintain a set of N canary queries with known reference responses
    - Periodically re-submit canaries and compare new responses to references
    - Uses cosine similarity via Pinecone's integrated embedding (llama-text-embed-v2, 1024-dim)
    
    Thresholds (from Metric Design Document):
    - GREEN:  TRCI ≥ 0.95  (no detectable drift)
    - AMBER:  0.90 ≤ TRCI < 0.95  (investigate)
    - RED:    TRCI < 0.90  (significant drift — possible silent model update)
    - TRCI_p10 < 0.80  (worst-case canary has drifted severely)
    """
    
    def __init__(self, config: TRCIConfig):
        self.canary_set = self.load_canary_set(config.canary_path)
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = pc.Index("rait-chatbot")
        # Store reference response embeddings in a dedicated Pinecone namespace
        self._init_reference_embeddings()
        
    def load_canary_set(self, path: str) -> list[CanaryQuery]:
        """
        Canary set design:
        - 50 queries covering all topic areas (benefits, housing, council tax)
        - 10 queries per category: eligibility, amounts, process, edge cases, cross-topic
        - Each has a reference response from the initial system baseline
        - Canaries include known edge cases (e.g., savings exactly £16,000)
        """
        return json.load(open(path))
    
    def _init_reference_embeddings(self):
        """
        Upsert reference response embeddings into Pinecone 'canary_refs' namespace.
        Uses Pinecone integrated embedding (llama-text-embed-v2) — same model
        used for the main corpus, ensuring embedding space consistency.
        """
        existing = self.index.describe_index_stats()
        if existing.namespaces.get("canary_refs", {}).get("vector_count", 0) == 0:
            # First run: embed and store reference responses
            for canary in self.canary_set:
                self.index.upsert(
                    vectors=[{
                        "id": f"ref_{canary.id}",
                        "metadata": {"text": canary.reference_response, "canary_id": canary.id},
                    }],
                    namespace="canary_refs",
                )
    
    def run_probe(self) -> TRCIResult:
        """
        Execute canary probe and compute TRCI metrics.
        
        For each canary query:
        1. Re-submit through production pipeline to get new response
        2. Embed new response via Pinecone (upsert to temp namespace)
        3. Compute cosine similarity between new and reference embeddings
        """
        similarities = []
        per_canary_results = []
        
        for i, canary in enumerate(self.canary_set):
            # Re-submit canary through the full production pipeline
            new_response = self.production_pipeline.process(canary.query)
            
            # Upsert new response to get its embedding, then query
            # against the reference to get cosine similarity score
            self.index.upsert(
                vectors=[{
                    "id": f"probe_{canary.id}",
                    "metadata": {"text": new_response.raw_response},
                }],
                namespace="canary_probes",
            )
            
            # Query the reference namespace with the new response text
            # The score IS the cosine similarity (Pinecone cosine metric)
            match = self.index.query(
                namespace="canary_refs",
                top_k=1,
                data=new_response.raw_response,
                filter={"canary_id": canary.id},
                include_metadata=True,
            )
            sim = match.matches[0].score if match.matches else 0.0
            
            similarities.append(sim)
            per_canary_results.append(CanaryResult(
                canary_id=canary.id,
                query=canary.query,
                topic=canary.topic,
                reference_response_preview=canary.reference_response[:200],
                new_response_preview=new_response.raw_response[:200],
                similarity=float(sim),
                status=self.classify_similarity(sim)
            ))
        
        similarities = np.array(similarities)
        
        result = TRCIResult(
            timestamp=datetime.utcnow(),
            trci_mean=float(np.mean(similarities)),
            trci_p10=float(np.percentile(similarities, 10)),
            trci_std=float(np.std(similarities)),
            n_canaries=len(self.canary_set),
            n_green=sum(1 for s in similarities if s >= 0.95),
            n_amber=sum(1 for s in similarities if 0.90 <= s < 0.95),
            n_red=sum(1 for s in similarities if s < 0.90),
            per_canary_results=per_canary_results,
            status=self.classify_overall(np.mean(similarities), np.percentile(similarities, 10)),
            explanation=self.generate_explanation(similarities, per_canary_results)
        )
        
        return result
    
    def classify_overall(self, mean: float, p10: float) -> str:
        if p10 < 0.80:
            return "RED — Worst-case canary severely drifted"
        if mean < 0.90:
            return "RED — Mean TRCI below critical threshold"
        if mean < 0.95:
            return "AMBER — Consistent but drifting; investigate"
        return "GREEN — No detectable drift"
    
    def generate_explanation(self, similarities, per_canary) -> str:
        """Human-readable explanation for governance report."""
        worst = min(per_canary, key=lambda x: x.similarity)
        return (
            f"TRCI probe ran {len(per_canary)} canary queries. "
            f"Mean similarity: {np.mean(similarities):.4f}, "
            f"10th percentile: {np.percentile(similarities, 10):.4f}. "
            f"Worst canary: '{worst.query[:80]}...' "
            f"(topic: {worst.topic}, similarity: {worst.similarity:.4f}). "
            f"{'No action needed.' if np.mean(similarities) >= 0.95 else 'Investigation recommended.'}"
        )
```

#### 3.2.2 CDS — Composite Drift Signal

```python
class CDSEngine:
    """
    Monitors distribution shifts across K text descriptors.
    
    Mechanism:
    - For each descriptor, compute JSD between reference and current windows
    - Weighted fusion into single composite score
    - Persistence filter: only flag if drift persists for P consecutive windows
    
    Thresholds:
    - GREEN:  CDS < 0.05
    - AMBER:  0.05 ≤ CDS < 0.15
    - RED:    CDS ≥ 0.15 (persisting for P=3 consecutive windows)
    """
    
    DESCRIPTORS = {
        "response_length":       {"weight": 0.15, "type": "continuous"},
        "response_sentiment":    {"weight": 0.10, "type": "continuous"},
        "response_readability":  {"weight": 0.10, "type": "continuous"},
        "citation_count":        {"weight": 0.15, "type": "discrete"},
        "hedge_word_count":      {"weight": 0.10, "type": "discrete"},
        "refusal_flag":          {"weight": 0.15, "type": "binary"},
        "response_latency_ms":   {"weight": 0.10, "type": "continuous"},
        "mean_retrieval_distance": {"weight": 0.10, "type": "continuous"},
        "context_token_ratio":   {"weight": 0.05, "type": "continuous"},
    }
    
    def compute(self, reference_window: DescriptorSet, 
                current_window: DescriptorSet) -> CDSResult:
        """Compute CDS between reference and current descriptor distributions."""
        
        per_descriptor_results = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, config in self.DESCRIPTORS.items():
            ref_values = reference_window.get_values(name)
            cur_values = current_window.get_values(name)
            
            if config["type"] == "binary":
                # For binary: compare proportions using JSD on [p, 1-p]
                ref_prop = np.mean(ref_values)
                cur_prop = np.mean(cur_values)
                jsd = self.binary_jsd(ref_prop, cur_prop)
            elif config["type"] == "discrete":
                # For discrete: histogram-based JSD
                jsd = self.discrete_jsd(ref_values, cur_values)
            else:
                # For continuous: KDE-based JSD with 50 bins
                jsd = self.continuous_jsd(ref_values, cur_values, bins=50)
            
            per_descriptor_results[name] = DescriptorDriftResult(
                name=name,
                weight=config["weight"],
                jsd=float(jsd),
                weighted_contribution=float(jsd * config["weight"]),
                ref_summary=self.summarise(ref_values),
                cur_summary=self.summarise(cur_values),
                status=self.classify_descriptor(jsd)
            )
            
            weighted_sum += jsd * config["weight"]
            total_weight += config["weight"]
        
        cds_value = weighted_sum / total_weight
        
        # Persistence check
        persistence_met = self.check_persistence(cds_value)
        
        result = CDSResult(
            timestamp=datetime.utcnow(),
            cds_value=float(cds_value),
            per_descriptor=per_descriptor_results,
            persistence_windows=self.persistence_history,
            persistence_met=persistence_met,
            status=self.classify_overall(cds_value, persistence_met),
            explanation=self.generate_explanation(cds_value, per_descriptor_results)
        )
        
        return result
    
    @staticmethod
    def continuous_jsd(p_samples: np.ndarray, q_samples: np.ndarray, bins: int = 50) -> float:
        """JSD between two continuous distributions via histogram binning."""
        # Shared bin edges
        all_values = np.concatenate([p_samples, q_samples])
        bin_edges = np.histogram_bin_edges(all_values, bins=bins)
        
        p_hist, _ = np.histogram(p_samples, bins=bin_edges, density=True)
        q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=True)
        
        # Add epsilon to avoid log(0)
        eps = 1e-10
        p_hist = p_hist + eps
        q_hist = q_hist + eps
        
        # Normalise to probability distributions
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()
        
        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = (P+Q)/2
        m = 0.5 * (p_hist + q_hist)
        jsd = 0.5 * entropy(p_hist, m) + 0.5 * entropy(q_hist, m)
        
        return float(jsd)
    
    def generate_explanation(self, cds_value, per_descriptor) -> str:
        top_contributors = sorted(
            per_descriptor.values(), 
            key=lambda x: x.weighted_contribution, 
            reverse=True
        )[:3]
        
        contrib_text = "; ".join([
            f"{c.name} (JSD={c.jsd:.4f}, weight={c.weight})"
            for c in top_contributors
        ])
        
        return (
            f"CDS = {cds_value:.4f}. "
            f"Top drift contributors: {contrib_text}. "
            f"{'Persistence threshold met — action required.' if self.persistence_met else 'Within acceptable bounds or not yet persistent.'}"
        )
```

#### 3.2.3 FDS — Faithfulness Decay Score

```python
class FDSEngine:
    """
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
    
    Ground Truth Calibration:
    - Where partial ground truth exists, compute rank correlation ρ(f^val, f)
    - If ρ drops below 0.6, the evaluator itself has drifted → escalate
    """
    
    def compute(self, current_logs: list[ProductionLog],
                reference_faithfulness: np.ndarray) -> FDSResult:
        """Compute FDS for current window against reference."""
        
        per_query_results = []
        current_faithfulness = []
        
        # Sample queries for evaluation (budget-aware)
        sample = self.sample_queries(current_logs, n=self.config.sample_size)
        
        for log in sample:
            # Step 1: Atomic claim decomposition
            claims = self.decompose_claims(log.raw_response)
            
            # Step 2: LLM-as-Judge verification per claim
            claim_results = []
            for claim in claims:
                verdict = self.judge.verify_claim(
                    claim=claim,
                    context=self.reconstruct_context(log.selected_context_ids)
                )
                claim_results.append(ClaimVerdict(
                    claim=claim,
                    verdict=verdict.label,  # "supported" | "unsupported" | "ambiguous"
                    confidence=verdict.confidence,
                    evidence_quote=verdict.evidence_quote
                ))
            
            # Step 3: Per-response faithfulness score
            n_supported = sum(1 for c in claim_results if c.verdict == "supported")
            n_total = len(claim_results)
            faithfulness = n_supported / max(n_total, 1)
            
            current_faithfulness.append(faithfulness)
            per_query_results.append(FDSQueryResult(
                query_id=log.query_id,
                query_preview=log.raw_query[:100],
                n_claims=n_total,
                n_supported=n_supported,
                faithfulness=faithfulness,
                claims=claim_results
            ))
        
        current_faithfulness = np.array(current_faithfulness)
        
        # Step 4: Signed JSD
        signed_jsd = self.compute_signed_jsd(
            reference_faithfulness, current_faithfulness
        )
        
        # Step 5: Ground truth calibration (if available)
        calibration = self.calibrate_with_ground_truth(per_query_results)
        
        result = FDSResult(
            timestamp=datetime.utcnow(),
            fds_value=float(signed_jsd),
            mean_faithfulness=float(np.mean(current_faithfulness)),
            std_faithfulness=float(np.std(current_faithfulness)),
            n_evaluated=len(per_query_results),
            per_query=per_query_results,
            calibration=calibration,
            status=self.classify(signed_jsd, calibration),
            explanation=self.generate_explanation(signed_jsd, current_faithfulness, per_query_results)
        )
        
        return result
    
    def compute_signed_jsd(self, ref: np.ndarray, cur: np.ndarray) -> float:
        """
        Signed JSD: positive = faithfulness improved, negative = faithfulness decayed.
        Sign determined by direction of mean shift.
        """
        jsd = CDSEngine.continuous_jsd(ref, cur, bins=20)
        sign = 1 if np.mean(cur) >= np.mean(ref) else -1
        return sign * jsd
    
    def decompose_claims(self, response: str) -> list[str]:
        """Use LLM to decompose response into atomic, verifiable claims."""
        prompt = f"""Decompose the following response into a list of atomic, 
        factually verifiable claims. Each claim should be a single sentence that 
        can be independently verified against the source documents.
        
        Response: {response}
        
        Return a JSON array of claim strings."""
        
        result = self.llm_client.chat.completions.create(
            model=self.config.decomposition_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # deterministic decomposition
            response_format={"type": "json_object"}
        )
        return json.loads(result.choices[0].message.content)["claims"]
    
    def generate_explanation(self, fds, faithfulness_arr, per_query) -> str:
        worst = min(per_query, key=lambda x: x.faithfulness)
        return (
            f"FDS = {fds:+.4f} ({'decay' if fds < 0 else 'improvement'}). "
            f"Mean faithfulness: {np.mean(faithfulness_arr):.3f} "
            f"(reference baseline: {np.mean(self.reference_faithfulness):.3f}). "
            f"Worst query: '{worst.query_preview}' "
            f"({worst.n_supported}/{worst.n_claims} claims supported). "
            f"{'Faithfulness is decaying — review LLM behaviour.' if fds < -0.02 else 'Within acceptable bounds.'}"
        )
```

#### 3.2.4 DDI — Differential Drift Index

```python
class DDIEngine:
    """
    Detects non-uniform drift across topic segments as a proxy for fairness.
    
    Mechanism:
    - Segment queries by topic (benefits, housing, council tax, disability, etc.)
    - Compute per-segment quality proxy (response length, faithfulness proxy, etc.)
    - Measure how differently segments are drifting using std/range of drift scores
    
    Thresholds:
    - GREEN:  DDI < 0.05
    - AMBER:  0.05 ≤ DDI < 0.15
    - RED:    DDI ≥ 0.15
    
    Justification (Equality Act 2010, S.149 PSED):
    Topic-based segmentation serves as a proxy for protected characteristics
    because certain benefits/housing topics disproportionately affect specific
    demographic groups (e.g., disability benefits → disabled persons,
    housing benefit → low-income groups, pension credit → elderly).
    """
    
    TOPIC_SEGMENTS = {
        "universal_credit": {
            "keywords": ["universal credit", "uc", "work coach", "claimant commitment"],
            "protected_proxy": "low-income, working-age adults"
        },
        "housing_benefit": {
            "keywords": ["housing benefit", "rent", "spare bedroom", "lha"],
            "protected_proxy": "low-income tenants, elderly (State Pension age)"
        },
        "disability_benefits": {
            "keywords": ["pip", "disability", "attendance allowance", "dla", "esa"],
            "protected_proxy": "disabled persons (Equality Act protected characteristic)"
        },
        "council_tax": {
            "keywords": ["council tax", "band", "council tax reduction", "discount"],
            "protected_proxy": "all citizens (geographic variation)"
        },
        "homelessness": {
            "keywords": ["homeless", "emergency housing", "temporary accommodation"],
            "protected_proxy": "vulnerable persons, rough sleepers"
        },
        "pension": {
            "keywords": ["pension", "retirement", "state pension age", "pension credit"],
            "protected_proxy": "elderly persons (age-related)"
        }
    }
    
    def compute(self, reference_window: MetricDataset,
                current_window: MetricDataset) -> DDIResult:
        """Compute DDI between reference and current windows."""
        
        # Step 1: Segment queries by topic
        ref_segments = self.segment_by_topic(reference_window.logs)
        cur_segments = self.segment_by_topic(current_window.logs)
        
        # Step 2: Compute per-segment quality proxy
        per_segment_results = {}
        segment_drift_scores = {}
        
        for topic, config in self.TOPIC_SEGMENTS.items():
            ref_logs = ref_segments.get(topic, [])
            cur_logs = cur_segments.get(topic, [])
            
            if len(ref_logs) < self.config.min_segment_size or \
               len(cur_logs) < self.config.min_segment_size:
                per_segment_results[topic] = SegmentResult(
                    topic=topic,
                    n_ref=len(ref_logs),
                    n_cur=len(cur_logs),
                    drift_score=None,
                    status="INSUFFICIENT_DATA",
                    explanation=f"Need ≥{self.config.min_segment_size} queries per window"
                )
                continue
            
            # Quality proxy: composite of response attributes
            ref_quality = self.compute_quality_proxy(ref_logs)
            cur_quality = self.compute_quality_proxy(cur_logs)
            
            # Per-segment drift (JSD of quality distribution)
            segment_jsd = CDSEngine.continuous_jsd(ref_quality, cur_quality, bins=20)
            
            segment_drift_scores[topic] = segment_jsd
            per_segment_results[topic] = SegmentResult(
                topic=topic,
                n_ref=len(ref_logs),
                n_cur=len(cur_logs),
                ref_quality_mean=float(np.mean(ref_quality)),
                cur_quality_mean=float(np.mean(cur_quality)),
                quality_shift=float(np.mean(cur_quality) - np.mean(ref_quality)),
                drift_score=float(segment_jsd),
                protected_proxy=config["protected_proxy"],
                status=self.classify_segment(segment_jsd)
            )
        
        # Step 3: DDI = std of segment-level drift scores (non-uniformity measure)
        if len(segment_drift_scores) >= 2:
            drift_values = np.array(list(segment_drift_scores.values()))
            ddi_value = float(np.std(drift_values))
            ddi_range = float(np.max(drift_values) - np.min(drift_values))
        else:
            ddi_value = 0.0
            ddi_range = 0.0
        
        # Step 4: Intersectional escalation check
        worst_segment = max(segment_drift_scores, key=segment_drift_scores.get) \
                        if segment_drift_scores else None
        best_segment = min(segment_drift_scores, key=segment_drift_scores.get) \
                       if segment_drift_scores else None
        
        result = DDIResult(
            timestamp=datetime.utcnow(),
            ddi_value=ddi_value,
            ddi_range=ddi_range,
            n_segments_evaluated=len(segment_drift_scores),
            per_segment=per_segment_results,
            worst_segment=worst_segment,
            best_segment=best_segment,
            intersectional_flag=ddi_range > 0.20,  # Escalation if gap is extreme
            status=self.classify_overall(ddi_value, ddi_range),
            explanation=self.generate_explanation(
                ddi_value, per_segment_results, worst_segment, best_segment
            )
        )
        
        return result
    
    def compute_quality_proxy(self, logs: list[ProductionLog]) -> np.ndarray:
        """
        Quality proxy per response:
        - 40% response completeness (token count / expected range)
        - 30% citation inclusion (binary: at least 1 GOV.UK citation)
        - 20% non-refusal (binary: not a refusal response)
        - 10% response latency (inverse, normalised)
        """
        quality_scores = []
        for log in logs:
            completeness = min(log.completion_tokens / 200, 1.0)  # normalised
            has_citation = 1.0 if log.citation_count > 0 else 0.0
            non_refusal = 0.0 if log.refusal_flag else 1.0
            latency_score = max(0, 1.0 - log.response_latency_ms / 5000)  # normalised
            
            quality = (0.4 * completeness + 0.3 * has_citation + 
                       0.2 * non_refusal + 0.1 * latency_score)
            quality_scores.append(quality)
        
        return np.array(quality_scores)
    
    def generate_explanation(self, ddi, per_segment, worst, best) -> str:
        if worst and best:
            return (
                f"DDI = {ddi:.4f}. Evaluated {len([s for s in per_segment.values() if s.drift_score is not None])} "
                f"topic segments. Most drifted: '{worst}' "
                f"(JSD={per_segment[worst].drift_score:.4f}, proxy: {per_segment[worst].protected_proxy}). "
                f"Least drifted: '{best}' (JSD={per_segment[best].drift_score:.4f}). "
                f"{'Non-uniform drift detected — potential fairness concern under Equality Act S.149.' if ddi >= 0.05 else 'Drift is relatively uniform across segments.'}"
            )
        return f"DDI = {ddi:.4f}. Insufficient data for detailed segment analysis."
```

### 3.3 Metric Orchestrator

```python
class MetricOrchestrator:
    """
    Coordinates execution of all four metrics with configurable scheduling.
    This is the central entry point for the monitoring system.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.data_layer = DataCollectionLayer(config.db_path)
        self.trci = TRCIEngine(config.trci)
        self.cds = CDSEngine(config.cds)
        self.fds = FDSEngine(config.fds)
        self.ddi = DDIEngine(config.ddi)
        self.alert_engine = AlertEngine(config.alerts)
        self.export_engine = ExportEngine(config.export)
    
    def run_daily(self) -> DailyReport:
        """Daily monitoring run — TRCI + CDS."""
        trci_result = self.trci.run_probe()
        
        ref_window = self.data_layer.collect_window(
            self.config.reference_start, self.config.reference_end
        )
        current_window = self.data_layer.collect_window(
            datetime.utcnow() - timedelta(days=1), datetime.utcnow()
        )
        cds_result = self.cds.compute(ref_window.descriptors, current_window.descriptors)
        
        report = DailyReport(
            date=datetime.utcnow().date(),
            trci=trci_result,
            cds=cds_result,
            alerts=self.alert_engine.evaluate([trci_result, cds_result])
        )
        
        self.store_result(report)
        return report
    
    def run_weekly(self) -> WeeklyReport:
        """Weekly monitoring run — all four metrics."""
        daily = self.run_daily()
        
        # Weekly windows for FDS and DDI
        ref_window = self.data_layer.collect_window(
            self.config.reference_start, self.config.reference_end
        )
        weekly_window = self.data_layer.collect_window(
            datetime.utcnow() - timedelta(weeks=1), datetime.utcnow()
        )
        
        fds_result = self.fds.compute(
            weekly_window.logs, self.config.reference_faithfulness
        )
        ddi_result = self.ddi.compute(ref_window, weekly_window)
        
        report = WeeklyReport(
            week_ending=datetime.utcnow().date(),
            trci=daily.trci,
            cds=daily.cds,
            fds=fds_result,
            ddi=ddi_result,
            cross_metric_correlation=self.compute_cross_metric_correlation(),
            alerts=self.alert_engine.evaluate([
                daily.trci, daily.cds, fds_result, ddi_result
            ])
        )
        
        self.store_result(report)
        return report
    
    def run_quarterly_governance(self) -> QuarterlyReport:
        """
        Quarterly governance report — per Ethics Framework Point 7.
        Aggregates all weekly reports + trend analysis + recommendations.
        """
        weekly_reports = self.load_weekly_reports(last_n_weeks=13)
        
        report = QuarterlyReport(
            quarter=self.current_quarter(),
            weekly_summaries=[w.summary() for w in weekly_reports],
            trci_trend=self.compute_trend([w.trci.trci_mean for w in weekly_reports]),
            cds_trend=self.compute_trend([w.cds.cds_value for w in weekly_reports]),
            fds_trend=self.compute_trend([w.fds.fds_value for w in weekly_reports]),
            ddi_trend=self.compute_trend([w.ddi.ddi_value for w in weekly_reports]),
            governance_compliance=self.assess_compliance(weekly_reports),
            recommendations=self.generate_recommendations(weekly_reports),
            dpia_update=self.generate_dpia_update(weekly_reports)
        )
        
        return report
```

### 3.4 Alert System

```python
class AlertEngine:
    """
    Multi-level alert system aligned with UK governance escalation paths.
    """
    
    ALERT_RULES = [
        # TRCI alerts
        AlertRule(
            name="TRCI_CRITICAL",
            condition=lambda r: isinstance(r, TRCIResult) and r.trci_mean < 0.90,
            severity="CRITICAL",
            action="Immediate investigation — possible silent model update (NCSC 4.1)",
            notification=["ml_engineer", "governance_officer"],
            sla_hours=4
        ),
        AlertRule(
            name="TRCI_WARNING",
            condition=lambda r: isinstance(r, TRCIResult) and r.trci_mean < 0.95,
            severity="WARNING",
            action="Review canary results; check for LLM provider changes",
            notification=["ml_engineer"],
            sla_hours=24
        ),
        
        # CDS alerts
        AlertRule(
            name="CDS_PERSISTENT_DRIFT",
            condition=lambda r: isinstance(r, CDSResult) and r.persistence_met and r.cds_value >= 0.15,
            severity="CRITICAL",
            action="Persistent distribution shift — root cause analysis required (ICO accuracy mandate)",
            notification=["ml_engineer", "governance_officer"],
            sla_hours=8
        ),
        
        # FDS alerts
        AlertRule(
            name="FDS_FAITHFULNESS_DECAY",
            condition=lambda r: isinstance(r, FDSResult) and r.fds_value < -0.10,
            severity="CRITICAL",
            action="Significant faithfulness decay — may be generating hallucinations (DPA Art 5(1)(d))",
            notification=["ml_engineer", "governance_officer", "dpo"],
            sla_hours=4
        ),
        AlertRule(
            name="FDS_EVALUATOR_DRIFT",
            condition=lambda r: isinstance(r, FDSResult) and r.calibration and r.calibration.rho < 0.6,
            severity="CRITICAL",
            action="LLM-as-Judge evaluator has drifted — calibration required (meta-drift)",
            notification=["ml_engineer"],
            sla_hours=8
        ),
        
        # DDI alerts
        AlertRule(
            name="DDI_FAIRNESS_CONCERN",
            condition=lambda r: isinstance(r, DDIResult) and r.ddi_value >= 0.15,
            severity="CRITICAL",
            action="Non-uniform drift across segments — Equality Act S.149 PSED concern",
            notification=["ml_engineer", "governance_officer", "equality_officer"],
            sla_hours=8
        ),
        AlertRule(
            name="DDI_INTERSECTIONAL",
            condition=lambda r: isinstance(r, DDIResult) and r.intersectional_flag,
            severity="CRITICAL",
            action="Extreme inter-segment drift gap — intersectional fairness review required",
            notification=["ml_engineer", "governance_officer", "equality_officer", "dpo"],
            sla_hours=4
        ),
    ]
```

---

## 4. Toggle-Friendly Experimentation Dashboard

### 4.1 Dashboard Architecture

The dashboard is built in **Streamlit** with a modular page structure. Every parameter is exposed as a toggle, slider, or dropdown — enabling hypothesis testing without code changes.

```
streamlit_app/
├── app.py                    # Main entry point
├── pages/
│   ├── 01_overview.py        # System health summary
│   ├── 02_trci.py            # TRCI deep-dive
│   ├── 03_cds.py             # CDS deep-dive
│   ├── 04_fds.py             # FDS deep-dive
│   ├── 05_ddi.py             # DDI deep-dive
│   ├── 06_experiment.py      # A/B hypothesis testing
│   ├── 07_governance.py      # Quarterly report generation
│   └── 08_export.py          # Export centre
├── components/
│   ├── metric_card.py        # Reusable metric status card
│   ├── threshold_editor.py   # Interactive threshold adjustment
│   ├── time_series.py        # Time series chart with annotations
│   ├── distribution_plot.py  # Side-by-side distribution comparison
│   └── explanation_panel.py  # Human-readable explanation display
└── config/
    └── defaults.yaml         # Default parameter values
```

### 4.2 Toggle Controls Per Metric

#### 4.2.1 TRCI Toggles

| Control | Widget | Range/Options | Default | Hypothesis It Tests |
|---|---|---|---|---|
| **Enable/disable TRCI** | Toggle | on/off | on | "Is active probing necessary?" |
| **Canary set size** | Slider | 10–100 | 50 | "Does more canaries improve detection?" |
| **Topic filter** | Multi-select | All topics | All selected | "Which topics are drifting most?" |
| **Similarity metric** | Dropdown | Cosine / Jaccard / BLEU / BERTScore | Cosine | "Which similarity better captures semantic shift?" |
| **Green threshold** | Slider | 0.80–1.00 | 0.95 | "Is 0.95 too conservative?" |
| **Amber threshold** | Slider | 0.70–0.98 | 0.90 | "Where should we start investigating?" |
| **Red threshold (p10)** | Slider | 0.50–0.95 | 0.80 | "What worst-case is acceptable?" |
| **Probe frequency** | Dropdown | Hourly/Daily/Weekly | Daily | "Do we need real-time canaries?" |
| **Show per-canary breakdown** | Toggle | on/off | off | N/A (display control) |

#### 4.2.2 CDS Toggles

| Control | Widget | Range/Options | Default | Hypothesis It Tests |
|---|---|---|---|---|
| **Enable/disable CDS** | Toggle | on/off | on | N/A |
| **Per-descriptor enable** | 9× Toggles | on/off each | All on | "Which descriptors actually matter?" |
| **Per-descriptor weight** | 9× Sliders | 0.0–1.0 | See defaults | "Is response_length really 15% important?" |
| **Window size (reference)** | Slider | 7–90 days | 30 days | "Longer reference = more stable baseline?" |
| **Window size (current)** | Slider | 1–30 days | 7 days | "Shorter window = faster detection?" |
| **JSD bin count** | Slider | 10–200 | 50 | "Does binning resolution affect sensitivity?" |
| **Persistence threshold P** | Slider | 1–10 windows | 3 | "Is 3 windows too slow to react?" |
| **Green/Amber/Red thresholds** | 3× Sliders | 0.0–0.5 | 0.05/0.15/0.30 | "Are thresholds calibrated correctly?" |
| **Normalise weights** | Toggle | on/off | on | "Should disabled descriptors redistribute weight?" |

#### 4.2.3 FDS Toggles

| Control | Widget | Range/Options | Default | Hypothesis It Tests |
|---|---|---|---|---|
| **Enable/disable FDS** | Toggle | on/off | on | N/A |
| **Sample size** | Slider | 10–500 | 50 per window | "More samples = more reliable FDS?" |
| **Sampling strategy** | Dropdown | Random / Stratified-by-topic / Recent-only | Stratified | "Does sampling strategy affect FDS?" |
| **Claim decomposition model** | Dropdown | Same LLM / Separate LLM | Same | "Does evaluator choice introduce bias?" |
| **Verification strictness** | Dropdown | Strict / Moderate / Lenient | Moderate | "How strict should claim verification be?" |
| **Include ambiguous claims** | Toggle | on/off | off | "Should 'ambiguous' count as unfaithful?" |
| **Signed JSD bins** | Slider | 10–100 | 20 | "Binning effect on FDS sensitivity" |
| **Ground truth calibration** | Toggle | on/off | on (when available) | "Is calibration improving accuracy?" |
| **Calibration ρ threshold** | Slider | 0.3–0.9 | 0.6 | "When should we distrust the evaluator?" |

#### 4.2.4 DDI Toggles

| Control | Widget | Range/Options | Default | Hypothesis It Tests |
|---|---|---|---|---|
| **Enable/disable DDI** | Toggle | on/off | on | N/A |
| **Topic segments** | Multi-select | All 6 segments | All | "Which segments should be monitored?" |
| **Min segment size** | Slider | 5–100 | 20 | "When is a segment too small to be reliable?" |
| **Quality proxy weights** | 4× Sliders | Completeness/Citation/Non-refusal/Latency | 0.4/0.3/0.2/0.1 | "Is completeness really 40% of quality?" |
| **DDI formula** | Dropdown | std / range / CV / max-min ratio | std | "Which non-uniformity measure is best?" |
| **Intersectional threshold** | Slider | 0.05–0.50 | 0.20 | "When is the segment gap extreme?" |
| **Green/Amber/Red thresholds** | 3× Sliders | 0.0–0.5 | 0.05/0.15/0.30 | "Are DDI thresholds calibrated?" |

### 4.3 Hypothesis Testing Workflow (Page: `06_experiment.py`)

The experiment page enables side-by-side comparison of metric configurations:

```
┌──────────────────────────────────────────────────────┐
│  EXPERIMENT: Compare Two Metric Configurations       │
│                                                      │
│  ┌─────────────────┐    ┌─────────────────┐          │
│  │ Configuration A  │    │ Configuration B  │         │
│  │ (Current/Default)│    │ (Modified)       │         │
│  ├─────────────────┤    ├─────────────────┤          │
│  │ CDS weights:    │    │ CDS weights:    │          │
│  │  resp_length: 15│    │  resp_length: 25│          │
│  │  refusal: 15    │    │  refusal: 30    │          │
│  │  ...            │    │  ...            │          │
│  │                 │    │                 │          │
│  │ CDS Result:     │    │ CDS Result:     │          │
│  │  value: 0.087   │    │  value: 0.134   │          │
│  │  status: AMBER  │    │  status: AMBER  │          │
│  └─────────────────┘    └─────────────────┘          │
│                                                      │
│  Delta Analysis:                                     │
│  - CDS increased by 0.047 (+54%)                     │
│  - Refusal weight increase amplified drift signal    │
│  - Recommendation: Configuration B is more           │
│    sensitive to refusal-driven drift                  │
│                                                      │
│  [Save Config A] [Save Config B] [Export Comparison] │
└──────────────────────────────────────────────────────┘
```

**Workflow:**
1. User selects **Configuration A** (default or saved preset)
2. User adjusts parameters to create **Configuration B**
3. System re-computes both configurations against the **same data window**
4. Side-by-side results + delta analysis + sensitivity chart
5. User can save either configuration as a preset or export the comparison

### 4.4 Sensitivity Analysis Tool

For each metric, the dashboard provides an automated sensitivity sweep:

```python
def sensitivity_sweep(metric_engine, base_config, param_name, 
                      param_range, data_window):
    """
    Sweep one parameter across a range, holding others fixed.
    Output: chart of metric value vs parameter value.
    """
    results = []
    for value in param_range:
        config = base_config.copy()
        config[param_name] = value
        metric_engine.update_config(config)
        result = metric_engine.compute(data_window)
        results.append({
            "param_value": value,
            "metric_value": result.value,
            "status": result.status
        })
    return results  # → Plotly line chart with status colour bands
```

This answers questions like:
- "At what CDS threshold do we start seeing persistent alerts?"
- "How many canaries do we need before TRCI stabilises?"
- "Does increasing FDS sample size change the diagnosis?"

---

## 5. LLM-as-Judge Design

### 5.1 Purpose and Scope

The LLM-as-Judge serves two functions within the monitoring system:

| Function | Metric | What It Judges |
|---|---|---|
| **Claim verification** | FDS | Whether atomic claims in responses are supported by retrieved context |
| **Quality assessment** | DDI | Response quality rating (used when ground truth is unavailable) |

### 5.2 Claim Verification Prompt (FDS)

```python
CLAIM_VERIFICATION_PROMPT = """You are a factual accuracy evaluator for a UK government services chatbot.

Your task: Determine whether the following CLAIM is supported by the provided CONTEXT.

CONTEXT (from official GOV.UK documents):
{context}

CLAIM to verify:
{claim}

Evaluate the claim and respond with exactly this JSON format:
{{
    "verdict": "supported" | "unsupported" | "ambiguous",
    "confidence": <float 0.0-1.0>,
    "evidence_quote": "<exact quote from context that supports or contradicts the claim, or 'N/A'>",
    "reasoning": "<1-2 sentence explanation>"
}}

Rules:
- "supported": The claim is directly stated or logically entailed by the context.
- "unsupported": The claim contradicts the context OR makes a statement not present in the context.
- "ambiguous": The context is relevant but neither clearly supports nor contradicts the claim.
- Be strict: if the context says "savings over £16,000" and the claim says "savings over £15,000", that is "unsupported".
- Numerical thresholds, dates, and eligibility criteria must match exactly.
"""
```

### 5.3 Evaluator Calibration Protocol

The LLM-as-Judge is itself susceptible to drift (meta-drift). Calibration protocol:

```python
class EvaluatorCalibration:
    """
    Uses partial ground truth to calibrate and monitor the LLM-as-Judge.
    
    Ground truth sources:
    - Manually verified claim-verdict pairs (created during system setup)
    - Known correct answers from GOV.UK page content (e.g., "£16,000 savings limit")
    - Periodically updated calibration set (minimum 30 pairs)
    """
    
    def __init__(self, calibration_set_path: str):
        self.calibration_set = self.load_calibration_set(calibration_set_path)
    
    def run_calibration(self, judge: LLMJudge) -> CalibrationResult:
        """Run calibration set through the judge and compare to ground truth."""
        judge_verdicts = []
        ground_truth_verdicts = []
        
        for item in self.calibration_set:
            judge_result = judge.verify_claim(
                claim=item.claim,
                context=item.context
            )
            judge_verdicts.append(self.verdict_to_score(judge_result.verdict))
            ground_truth_verdicts.append(self.verdict_to_score(item.ground_truth_verdict))
        
        # Spearman rank correlation
        rho, p_value = spearmanr(ground_truth_verdicts, judge_verdicts)
        
        # Agreement rate
        agreement = np.mean([
            1 if j == g else 0 
            for j, g in zip(judge_verdicts, ground_truth_verdicts)
        ])
        
        return CalibrationResult(
            rho=float(rho),
            p_value=float(p_value),
            agreement_rate=float(agreement),
            n_calibration_items=len(self.calibration_set),
            mismatches=[
                CalibrationMismatch(
                    claim=item.claim,
                    ground_truth=item.ground_truth_verdict,
                    judge_verdict=judge_result.verdict,
                    judge_confidence=judge_result.confidence
                )
                for item, judge_result in zip(self.calibration_set, judge_verdicts)
                if item.ground_truth_verdict != judge_result.verdict
            ],
            status="HEALTHY" if rho >= 0.6 else "DRIFTED — RECALIBRATE"
        )
```

### 5.4 Meta-Drift Protection

| Risk | Mitigation |
|---|---|
| LLM-as-Judge changes behaviour after provider update | Run calibration before AND after every TRCI alert |
| Judge becomes lenient over time | Track judge agreement trend; flag if agreement drops >10% over 4 weeks |
| Prompt sensitivity | Pin prompt version; test prompt against calibration set before deployment |
| Model version mismatch | Log model version (from API response headers) with every judge call |

---

## 6. Publicly Available Data & Datasets

### 6.1 Primary Data Source: GOV.UK Content

| Source | URL Pattern | Content | Licence | Use in System |
|---|---|---|---|---|
| GOV.UK Content API | `gov.uk/api/content/{path}` | Structured JSON with parts, titles, bodies | OGL v3.0 | RAG document corpus |
| GOV.UK Search API | `gov.uk/api/search.json?q=...` | Search results across GOV.UK | OGL v3.0 | Discovering relevant pages |
| GOV.UK Change History | Via `public_updated_at` field | Document update timestamps | OGL v3.0 | Knowledge drift detection |

### 6.2 Supplementary Datasets for Testing

| Dataset | Source | Content | Use in System |
|---|---|---|---|
| **UK Parliament Written Questions** | [data.parliament.uk](https://data.parliament.uk/) | Questions asked by MPs about benefits/housing | Realistic query distribution for testing |
| **Citizens Advice Dataset** (public summaries) | [citizensadvice.org.uk](https://www.citizensadvice.org.uk/) | Common citizen queries about benefits | Query template creation |
| **ONS Population Data** | [ons.gov.uk](https://www.ons.gov.uk/) | Demographics, housing statistics | Validating DDI segment representativeness |
| **GOV.UK Feedback Data** (aggregated) | Published in GOV.UK performance dashboards | Page satisfaction ratings | Ground truth proxy for quality |
| **Synthetic Canary Queries** | Self-generated | Carefully crafted queries covering all topics and edge cases | TRCI canary set |

### 6.3 Canary Query Design (for TRCI)

The canary set is designed to cover the **full topic × complexity matrix**:

| Category | Simple Query | Edge Case Query | Cross-Topic Query |
|---|---|---|---|
| **Universal Credit** | "How do I apply for Universal Credit?" | "Can I claim UC if my savings are exactly £16,000?" | "I'm getting Housing Benefit — do I need to switch to UC?" |
| **Housing Benefit** | "Am I eligible for Housing Benefit?" | "I'm over State Pension age but my partner is under — can I claim?" | "I'm in temporary accommodation — which benefit covers rent?" |
| **Council Tax** | "What council tax band am I in?" | "I've built an extension — does my band change?" | "I'm on Universal Credit — can I get Council Tax Reduction?" |
| **Disability** | "How do I apply for PIP?" | "My condition got worse — can I get a reassessment?" | "I get PIP — does that affect my Housing Benefit?" |
| **Homelessness** | "I'm about to be homeless — what help is there?" | "I'm sofa surfing — does that count as homeless?" | "I'm homeless and disabled — which benefits can I get?" |
| **Pension** | "When can I get my State Pension?" | "I deferred my pension — how much extra do I get?" | "I'm retired — am I eligible for Housing Benefit?" |

**Canary set delivery:** 50 queries stored in `config/canary_queries.json`, Version-controlled, reviewed quarterly.

---

## 7. Testing Strategies Per Metric

### 7.1 Testing Philosophy

Each metric is tested at **four levels**:

| Level | What | How | Coverage |
|---|---|---|---|
| **Unit tests** | Individual functions (JSD, similarity, etc.) | pytest with known inputs/outputs | Mathematical correctness |
| **Property-based tests** | Statistical properties that must hold | hypothesis library | Edge cases, boundary conditions |
| **Integration tests** | Full metric pipeline end-to-end | Synthetic production logs → metric result | Pipeline correctness |
| **Scenario tests** | Realistic drift scenarios | Injected drift patterns → expected alerts | Detection effectiveness |

### 7.2 TRCI Test Cases

#### 7.2.1 Unit Tests

```python
class TestTRCIUnit:
    """Unit tests for TRCI computation."""
    
    def test_identical_responses_give_similarity_one(self):
        """When response hasn't changed, similarity should be ~1.0."""
        emb = np.random.randn(1024)  # 1024-dim (llama-text-embed-v2)
        sim = cosine_similarity(emb.reshape(1,-1), emb.reshape(1,-1))[0][0]
        assert abs(sim - 1.0) < 1e-6
    
    def test_orthogonal_responses_give_similarity_zero(self):
        """Completely different responses should have low similarity."""
        emb1 = np.array([1, 0, 0] + [0]*1021)  # 1024-dim
        emb2 = np.array([0, 1, 0] + [0]*1021)
        sim = cosine_similarity(emb1.reshape(1,-1), emb2.reshape(1,-1))[0][0]
        assert abs(sim) < 1e-6
    
    def test_trci_mean_is_average_of_similarities(self):
        """TRCI_mean should be the arithmetic mean of per-canary similarities."""
        sims = [0.95, 0.92, 0.98, 0.88, 0.97]
        assert abs(np.mean(sims) - 0.94) < 0.001
    
    def test_trci_p10_is_10th_percentile(self):
        """TRCI_p10 should be the 10th percentile."""
        sims = np.linspace(0.80, 1.0, 100)
        assert abs(np.percentile(sims, 10) - 0.82) < 0.01
    
    def test_classification_boundaries(self):
        """Verify GREEN/AMBER/RED classification at exact boundaries."""
        engine = TRCIEngine(default_config)
        assert engine.classify_overall(0.95, 0.85) == "GREEN — No detectable drift"
        assert engine.classify_overall(0.94, 0.85) == "AMBER — Consistent but drifting; investigate"
        assert engine.classify_overall(0.89, 0.85) == "RED — Mean TRCI below critical threshold"
        assert engine.classify_overall(0.96, 0.79) == "RED — Worst-case canary severely drifted"
```

#### 7.2.2 Property-Based Tests

```python
class TestTRCIProperties:
    """Property-based tests using hypothesis library."""
    
    @given(similarities=st.lists(st.floats(min_value=0.0, max_value=1.0), 
                                  min_size=1, max_size=100))
    def test_trci_mean_bounded_zero_one(self, similarities):
        """TRCI_mean must always be in [0, 1]."""
        mean = np.mean(similarities)
        assert 0.0 <= mean <= 1.0
    
    @given(similarities=st.lists(st.floats(min_value=0.0, max_value=1.0), 
                                  min_size=10, max_size=100))
    def test_p10_less_than_or_equal_mean(self, similarities):
        """10th percentile should be ≤ mean (except degenerate cases)."""
        p10 = np.percentile(similarities, 10)
        mean = np.mean(similarities)
        assert p10 <= mean + 1e-10  # floating point tolerance
    
    @given(n=st.integers(min_value=1, max_value=100))
    def test_all_identical_similarities_give_zero_std(self, n):
        """If all canaries have same similarity, std should be 0."""
        sims = [0.95] * n
        assert np.std(sims) < 1e-10
```

#### 7.2.3 Scenario Tests

```python
class TestTRCIScenarios:
    """Test TRCI under realistic drift scenarios."""
    
    def test_silent_model_update(self):
        """
        Scenario: LLM provider silently updates the model.
        Expected: TRCI drops below 0.95, several canaries in AMBER/RED.
        
        Simulation: Mock the production pipeline to return slightly different
        responses for each canary (simulating that the new model produces
        different outputs). The Pinecone cosine similarity between new and
        reference response embeddings should drop.
        """
        engine = TRCIEngine(default_config)
        
        # Mock: production pipeline returns perturbed responses
        # (e.g., append random suffix to change embedding slightly)
        engine.production_pipeline = MockPipeline(perturbation_level=0.1)
        
        result = engine.run_probe()
        
        assert result.trci_mean < 0.95, "Should detect model update"
        assert result.n_amber + result.n_red > 0, "Some canaries should be flagged"
    
    def test_knowledge_base_update(self):
        """
        Scenario: GOV.UK updates Housing Benefit eligibility rules.
        Expected: Only housing-related canaries drift; others remain stable.
        """
        engine = TRCIEngine(default_config)
        
        # Inject: perturb only housing-related canary reference embeddings
        for i, canary in enumerate(engine.canary_set):
            if canary.topic == "housing_benefit":
                noise = np.random.randn(384) * 0.15
                engine.reference_embeddings[i] += noise
        
        result = engine.run_probe()
        
        # Housing canaries should drift
        housing_canaries = [c for c in result.per_canary_results if c.topic == "housing_benefit"]
        other_canaries = [c for c in result.per_canary_results if c.topic != "housing_benefit"]
        
        assert np.mean([c.similarity for c in housing_canaries]) < 0.90
        assert np.mean([c.similarity for c in other_canaries]) > 0.95
    
    def test_no_drift_baseline(self):
        """
        Scenario: System is stable, no changes.
        Expected: TRCI ≥ 0.98, all GREEN.
        """
        engine = TRCIEngine(default_config)
        # No injection — system should be self-consistent
        result = engine.run_probe()
        
        assert result.trci_mean >= 0.95
        assert result.status.startswith("GREEN")
    
    def test_gradual_drift(self):
        """
        Scenario: Model slowly degrades over 10 probe cycles.
        Expected: TRCI gradually decreases, transitioning GREEN → AMBER → RED.
        """
        engine = TRCIEngine(default_config)
        results = []
        
        for cycle in range(10):
            # Inject: increasing noise each cycle
            noise = np.random.randn(*engine.reference_embeddings.shape) * (0.02 * cycle)
            engine.reference_embeddings += noise
            result = engine.run_probe()
            results.append(result.trci_mean)
        
        # Should be monotonically (approximately) decreasing
        assert results[-1] < results[0], "TRCI should decrease with increasing drift"
```

#### 7.2.4 Edge Cases

```python
class TestTRCIEdgeCases:
    """Edge cases and boundary conditions."""
    
    def test_empty_canary_set(self):
        """System should handle empty canary set gracefully."""
        config = default_config.copy()
        config.canary_path = "empty_canary_set.json"  # file with []
        engine = TRCIEngine(config)
        result = engine.run_probe()
        assert result.trci_mean is None or result.n_canaries == 0
    
    def test_single_canary(self):
        """System should work with just one canary."""
        engine = TRCIEngine(single_canary_config)
        result = engine.run_probe()
        assert result.n_canaries == 1
        assert result.trci_mean == result.trci_p10  # With 1 canary, mean = p10
    
    def test_llm_api_timeout(self):
        """If LLM API times out for a canary, handle gracefully."""
        engine = TRCIEngine(default_config)
        engine.production_pipeline = TimeoutMockPipeline(timeout_indices=[0, 5, 10])
        result = engine.run_probe()
        
        # Should compute TRCI from successful canaries only
        assert result.n_canaries < 50
        assert result.trci_mean is not None
    
    def test_empty_response_from_llm(self):
        """If LLM returns empty response, similarity should be very low."""
        emb_empty = np.zeros(384)  # zero vector
        emb_ref = np.random.randn(384)
        # Cosine similarity of zero vector is undefined; should handle gracefully
        # Implementation should assign similarity = 0.0 for empty responses
    
    def test_extremely_long_response(self):
        """Very long responses should still embed and compare correctly."""
        # sentence-transformers truncates at 512 tokens; verify this is handled
        long_response = "word " * 10000
        embedding = model.encode(long_response)
        assert embedding.shape == (384,)  # Should still produce valid embedding
    
    def test_unicode_and_special_characters(self):
        """Canary queries with £, €, special chars should work."""
        query = "Can I claim if savings are £16,000?"
        response = model.encode(query)
        assert response.shape == (384,)
```

### 7.3 CDS Test Cases

#### 7.3.1 Unit Tests

```python
class TestCDSUnit:
    """Unit tests for CDS computation."""
    
    def test_jsd_identical_distributions(self):
        """JSD of a distribution with itself should be 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        jsd = compute_jsd(p, p)
        assert abs(jsd) < 1e-10
    
    def test_jsd_opposite_distributions(self):
        """JSD of disjoint distributions should be ln(2) ≈ 0.693."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        jsd = compute_jsd(p + 1e-10, q + 1e-10)
        assert abs(jsd - np.log(2)) < 0.01
    
    def test_jsd_symmetry(self):
        """JSD(P, Q) should equal JSD(Q, P)."""
        p = np.array([0.3, 0.7])
        q = np.array([0.6, 0.4])
        assert abs(compute_jsd(p, q) - compute_jsd(q, p)) < 1e-10
    
    def test_jsd_bounded(self):
        """JSD should be in [0, ln(2)] for base-e."""
        p = np.random.dirichlet(np.ones(10))
        q = np.random.dirichlet(np.ones(10))
        jsd = compute_jsd(p, q)
        assert 0 <= jsd <= np.log(2) + 1e-10
    
    def test_weighted_fusion(self):
        """CDS should be weighted average of per-descriptor JSDs."""
        descriptor_jsds = {"a": 0.1, "b": 0.2, "c": 0.3}
        weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        expected = 0.1*0.5 + 0.2*0.3 + 0.3*0.2  # = 0.05 + 0.06 + 0.06 = 0.17
        cds = sum(descriptor_jsds[k] * weights[k] for k in weights) / sum(weights.values())
        assert abs(cds - 0.17) < 1e-10
    
    def test_binary_jsd_same_proportions(self):
        """Binary JSD with same proportions should be 0."""
        jsd = CDSEngine.binary_jsd(0.5, 0.5)
        assert abs(jsd) < 1e-10
    
    def test_binary_jsd_extreme_swing(self):
        """Binary JSD from 0% to 100% should be maximum."""
        jsd = CDSEngine.binary_jsd(0.01, 0.99)
        assert jsd > 0.5  # Should be near ln(2)
```

#### 7.3.2 Scenario Tests

```python
class TestCDSScenarios:
    """Realistic drift scenarios for CDS."""
    
    def test_refusal_rate_spike(self):
        """
        Scenario: System starts refusing 30% of queries (was 5%).
        Expected: CDS elevated, refusal_flag descriptor is top contributor.
        """
        ref_window = generate_window(n=200, refusal_rate=0.05)
        cur_window = generate_window(n=200, refusal_rate=0.30)
        
        engine = CDSEngine(default_config)
        result = engine.compute(ref_window, cur_window)
        
        assert result.cds_value > 0.05, "Should detect refusal spike"
        refusal_contrib = result.per_descriptor["refusal_flag"].weighted_contribution
        assert refusal_contrib == max(
            d.weighted_contribution for d in result.per_descriptor.values()
        ), "Refusal should be top contributor"
    
    def test_response_length_shift(self):
        """
        Scenario: Responses become significantly shorter (model became terse).
        Expected: CDS elevated, response_length descriptor is top contributor.
        """
        ref_window = generate_window(n=200, response_length_mean=150, response_length_std=30)
        cur_window = generate_window(n=200, response_length_mean=50, response_length_std=10)
        
        engine = CDSEngine(default_config)
        result = engine.compute(ref_window, cur_window)
        
        assert result.cds_value > 0.05
        assert result.per_descriptor["response_length"].jsd > 0.1
    
    def test_single_descriptor_drift_isolated(self):
        """
        Scenario: Only one descriptor drifts; others are stable.
        Expected: CDS reflects only that descriptor's weighted contribution.
        """
        ref_window = generate_window(n=200)
        cur_window = generate_window(n=200)
        # Inject drift only in sentiment
        cur_window.set_values("response_sentiment", 
                              ref_window.get_values("response_sentiment") + 0.5)
        
        engine = CDSEngine(default_config)
        result = engine.compute(ref_window, cur_window)
        
        # Only sentiment should have significant JSD
        for name, desc_result in result.per_descriptor.items():
            if name == "response_sentiment":
                assert desc_result.jsd > 0.05
            else:
                assert desc_result.jsd < 0.02, f"{name} should not drift"
    
    def test_persistence_filter_blocks_blips(self):
        """
        Scenario: CDS spikes for 1 window then returns to normal.
        Expected: Persistence filter prevents RED alert.
        """
        engine = CDSEngine(default_config)  # P=3 persistence
        
        # Window 1: spike
        result1 = engine.compute(normal_ref, drifted_window)
        assert result1.cds_value > 0.15
        assert not result1.persistence_met, "1 window should not meet persistence"
        
        # Window 2: normal
        result2 = engine.compute(normal_ref, normal_window)
        assert result2.cds_value < 0.05
        assert not result2.persistence_met
    
    def test_persistence_filter_fires_on_sustained_drift(self):
        """
        Scenario: CDS elevated for 3 consecutive windows.
        Expected: Persistence threshold met, RED alert.
        """
        engine = CDSEngine(default_config)  # P=3
        
        for i in range(3):
            result = engine.compute(normal_ref, drifted_window)
        
        assert result.persistence_met, "3 consecutive windows should trigger persistence"
    
    def test_no_drift_baseline(self):
        """
        Scenario: Same distribution in reference and current.
        Expected: CDS near 0, all GREEN.
        """
        window = generate_window(n=500)
        engine = CDSEngine(default_config)
        result = engine.compute(window, window)
        
        assert result.cds_value < 0.02, "Same data should give near-zero CDS"
        assert result.status.startswith("GREEN")
```

#### 7.3.3 Edge Cases

```python
class TestCDSEdgeCases:
    """Edge cases for CDS."""
    
    def test_very_small_window(self):
        """Window with only 5 queries — JSD should still compute (may be noisy)."""
        ref = generate_window(n=200)
        cur = generate_window(n=5)
        engine = CDSEngine(default_config)
        result = engine.compute(ref, cur)
        assert result.cds_value is not None
    
    def test_all_zero_descriptor(self):
        """If a descriptor is all zeros (e.g., no citations ever), handle gracefully."""
        ref = generate_window(n=200)
        ref.set_values("citation_count", np.zeros(200))
        cur = generate_window(n=200)
        cur.set_values("citation_count", np.zeros(200))
        
        engine = CDSEngine(default_config)
        result = engine.compute(ref, cur)
        assert result.per_descriptor["citation_count"].jsd < 1e-6
    
    def test_single_unique_value(self):
        """If all values are identical (zero variance), JSD computation should not crash."""
        ref = generate_window(n=100)
        ref.set_values("response_latency_ms", np.full(100, 200.0))
        cur = generate_window(n=100)
        cur.set_values("response_latency_ms", np.full(100, 200.0))
        
        engine = CDSEngine(default_config)
        result = engine.compute(ref, cur)
        # Should not raise; JSD should be ~0
    
    def test_disabled_descriptors(self):
        """When some descriptors are toggled off, CDS uses remaining weights."""
        config = default_config.copy()
        config.disabled_descriptors = ["response_sentiment", "hedge_word_count"]
        engine = CDSEngine(config)
        result = engine.compute(ref_window, cur_window)
        
        assert "response_sentiment" not in result.per_descriptor
        assert "hedge_word_count" not in result.per_descriptor
    
    def test_custom_weights_normalisation(self):
        """When weights are changed, they should be normalised."""
        config = default_config.copy()
        config.weights = {"response_length": 1.0}  # Only one descriptor
        engine = CDSEngine(config)
        result = engine.compute(ref_window, cur_window)
        # CDS should be exactly the JSD of response_length
```

### 7.4 FDS Test Cases

#### 7.4.1 Unit Tests

```python
class TestFDSUnit:
    """Unit tests for FDS computation."""
    
    def test_perfect_faithfulness(self):
        """All claims supported → faithfulness = 1.0."""
        claims = [
            ClaimVerdict(claim="X", verdict="supported", confidence=1.0, evidence_quote="..."),
            ClaimVerdict(claim="Y", verdict="supported", confidence=0.9, evidence_quote="..."),
        ]
        faithfulness = sum(1 for c in claims if c.verdict == "supported") / len(claims)
        assert faithfulness == 1.0
    
    def test_zero_faithfulness(self):
        """No claims supported → faithfulness = 0.0."""
        claims = [
            ClaimVerdict(claim="X", verdict="unsupported", confidence=0.8, evidence_quote="N/A"),
        ]
        faithfulness = sum(1 for c in claims if c.verdict == "supported") / len(claims)
        assert faithfulness == 0.0
    
    def test_signed_jsd_positive_when_improved(self):
        """FDS should be positive when current faithfulness is higher than reference."""
        ref = np.array([0.6, 0.7, 0.5, 0.65, 0.55])
        cur = np.array([0.8, 0.85, 0.9, 0.82, 0.78])
        fds = FDSEngine.compute_signed_jsd(ref, cur)
        assert fds > 0, "Improvement should give positive FDS"
    
    def test_signed_jsd_negative_when_decayed(self):
        """FDS should be negative when current faithfulness is lower than reference."""
        ref = np.array([0.8, 0.85, 0.9, 0.82, 0.78])
        cur = np.array([0.6, 0.7, 0.5, 0.65, 0.55])
        fds = FDSEngine.compute_signed_jsd(ref, cur)
        assert fds < 0, "Decay should give negative FDS"
    
    def test_no_claims_in_response(self):
        """If response has no extractable claims, faithfulness should be None/handled."""
        # This can happen for very short or refusal responses
        # System should assign faithfulness = None and exclude from FDS computation
```

#### 7.4.2 Scenario Tests

```python
class TestFDSScenarios:
    """Realistic scenarios for FDS."""
    
    def test_hallucination_injection(self):
        """
        Scenario: Model starts generating claims not in source documents.
        Expected: FDS becomes significantly negative.
        """
        ref_faithfulness = np.array([0.85] * 50)  # baseline: 85% faithful
        
        # Simulate: current responses have 50% unsupported claims
        cur_logs = generate_logs_with_hallucinations(n=50, hallucination_rate=0.5)
        
        engine = FDSEngine(default_config)
        result = engine.compute(cur_logs, ref_faithfulness)
        
        assert result.fds_value < -0.05, "Hallucinations should cause negative FDS"
        assert result.mean_faithfulness < 0.7
    
    def test_knowledge_base_update_improves_faithfulness(self):
        """
        Scenario: Knowledge base updated with more accurate information.
        Expected: FDS becomes positive (faithfulness improved).
        """
        ref_faithfulness = np.array([0.75] * 50)
        cur_logs = generate_logs_with_improved_context(n=50, faithfulness_mean=0.90)
        
        engine = FDSEngine(default_config)
        result = engine.compute(cur_logs, ref_faithfulness)
        
        assert result.fds_value > 0, "Better KB should improve FDS"
    
    def test_evaluator_drift_detected(self):
        """
        Scenario: LLM-as-Judge becomes lenient (meta-drift).
        Expected: Calibration ρ drops below threshold.
        """
        engine = FDSEngine(default_config)
        
        # Create a biased judge that always says "supported"
        biased_judge = AlwaysSupportedJudge()
        engine.judge = biased_judge
        
        calibration = engine.calibrate_with_ground_truth(test_queries)
        
        assert calibration.rho < 0.6, "Biased judge should fail calibration"
        assert calibration.status == "DRIFTED — RECALIBRATE"
    
    def test_numerical_threshold_drift(self):
        """
        Scenario: Model starts saying "£15,000" instead of "£16,000" for savings limit.
        Expected: Claim verification catches this specific factual error.
        """
        claim = "You cannot claim Housing Benefit if your savings are over £15,000."
        context = "Usually, you will not get Housing Benefit if your savings are over £16,000."
        
        judge = LLMJudge(default_config)
        verdict = judge.verify_claim(claim=claim, context=context)
        
        assert verdict.verdict == "unsupported", "£15,000 ≠ £16,000"
    
    def test_stable_system_fds_near_zero(self):
        """
        Scenario: No change in model or knowledge base.
        Expected: FDS ≈ 0, within noise tolerance.
        """
        ref_faithfulness = np.array([0.85] * 50)
        cur_logs = generate_logs_with_faithfulness(n=50, mean=0.85, std=0.05)
        
        engine = FDSEngine(default_config)
        result = engine.compute(cur_logs, ref_faithfulness)
        
        assert abs(result.fds_value) < 0.02, "Stable system should have FDS ≈ 0"
```

#### 7.4.3 Edge Cases

```python
class TestFDSEdgeCases:
    """Edge cases for FDS."""
    
    def test_all_refusal_responses(self):
        """If all sampled responses are refusals, FDS should handle gracefully."""
        # Refusal responses have no claims to verify
        # FDS should report "insufficient data" rather than crash
    
    def test_single_claim_response(self):
        """Response with exactly one claim — faithfulness is 0 or 1."""
        # Should work but produce high-variance FDS
    
    def test_claim_decomposition_failure(self):
        """If LLM fails to decompose claims (returns invalid JSON), handle gracefully."""
        # Implementation should retry or skip with logging
    
    def test_very_long_response_many_claims(self):
        """Response with 50+ claims — verify budget management."""
        # FDS should cap claims per response (configurable, default: 20)
    
    def test_ambiguous_claims_handling(self):
        """With toggle on/off for ambiguous claims, FDS should differ."""
        # When ambiguous counts as unsupported → lower faithfulness
        # When ambiguous excluded → potentially higher faithfulness
    
    def test_empty_context(self):
        """If retrieved context is empty, all claims should be 'unsupported'."""
        claim = "Housing Benefit helps pay rent."
        verdict = judge.verify_claim(claim=claim, context="")
        assert verdict.verdict == "unsupported"
```

### 7.5 DDI Test Cases

#### 7.5.1 Unit Tests

```python
class TestDDIUnit:
    """Unit tests for DDI computation."""
    
    def test_uniform_drift_gives_zero_ddi(self):
        """If all segments drift equally, DDI (std) should be 0."""
        drift_scores = {"a": 0.1, "b": 0.1, "c": 0.1}
        ddi = np.std(list(drift_scores.values()))
        assert abs(ddi) < 1e-10
    
    def test_non_uniform_drift_gives_positive_ddi(self):
        """If segments drift differently, DDI should be > 0."""
        drift_scores = {"a": 0.01, "b": 0.05, "c": 0.20}
        ddi = np.std(list(drift_scores.values()))
        assert ddi > 0
    
    def test_topic_classification(self):
        """Queries should be correctly classified into topic segments."""
        engine = DDIEngine(default_config)
        
        assert engine.classify_topic("How do I apply for Universal Credit?") == "universal_credit"
        assert engine.classify_topic("What's my council tax band?") == "council_tax"
        assert engine.classify_topic("Am I eligible for PIP?") == "disability_benefits"
    
    def test_quality_proxy_bounded(self):
        """Quality proxy should be in [0, 1]."""
        logs = generate_logs(n=100)
        engine = DDIEngine(default_config)
        quality = engine.compute_quality_proxy(logs)
        assert np.all(quality >= 0) and np.all(quality <= 1)
```

#### 7.5.2 Scenario Tests

```python
class TestDDIScenarios:
    """Realistic scenarios for DDI."""
    
    def test_disability_segment_degradation(self):
        """
        Scenario: Disability benefits queries get significantly worse responses.
        Expected: DDI elevated, disability segment is worst.
        
        This is the key Equality Act concern — degradation affecting
        a segment that serves as proxy for a protected characteristic.
        """
        ref_window = generate_balanced_window(n=200, segments=6)
        cur_window = generate_balanced_window(n=200, segments=6)
        
        # Inject: degrade disability segment quality
        cur_window.degrade_segment("disability_benefits", quality_drop=0.3)
        
        engine = DDIEngine(default_config)
        result = engine.compute(ref_window, cur_window)
        
        assert result.ddi_value > 0.05, "Non-uniform drift should be detected"
        assert result.worst_segment == "disability_benefits"
    
    def test_all_segments_degrade_equally(self):
        """
        Scenario: Overall quality drops, but uniformly across segments.
        Expected: DDI low (drift is uniform), but individual segment JSDs may be high.
        """
        ref_window = generate_balanced_window(n=200, quality_mean=0.8)
        cur_window = generate_balanced_window(n=200, quality_mean=0.5)
        
        engine = DDIEngine(default_config)
        result = engine.compute(ref_window, cur_window)
        
        assert result.ddi_value < 0.05, "Uniform degradation → low DDI"
        # But CDS should catch the overall degradation
    
    def test_imbalanced_segments(self):
        """
        Scenario: One segment has many queries, another has very few.
        Expected: Small segment flagged as insufficient data, not used in DDI.
        """
        ref_window = generate_imbalanced_window(
            n=200, segment_sizes={"universal_credit": 150, "homelessness": 3}
        )
        cur_window = generate_imbalanced_window(
            n=200, segment_sizes={"universal_credit": 160, "homelessness": 2}
        )
        
        engine = DDIEngine(default_config)  # min_segment_size=20
        result = engine.compute(ref_window, cur_window)
        
        assert result.per_segment["homelessness"].status == "INSUFFICIENT_DATA"
    
    def test_intersectional_escalation(self):
        """
        Scenario: One segment has JSD=0.30, another has JSD=0.01 → range=0.29.
        Expected: Intersectional flag triggered (range > 0.20).
        """
        engine = DDIEngine(default_config)
        # Manually set segment drift scores with extreme gap
        result = engine.compute(extreme_gap_ref, extreme_gap_cur)
        
        assert result.intersectional_flag, "Extreme gap should trigger intersectional flag"
    
    def test_new_topic_appears(self):
        """
        Scenario: Queries about a new benefit appear that don't match existing segments.
        Expected: Queries classified as "other" or unclassified, flagged for review.
        """
        # This tests the robustness of topic classification
        query = "How do I get the new Carer's Support Payment?"
        engine = DDIEngine(default_config)
        topic = engine.classify_topic(query)
        # Should either match a close topic or return "uncategorised"
```

#### 7.5.3 Edge Cases

```python
class TestDDIEdgeCases:
    """Edge cases for DDI."""
    
    def test_single_segment(self):
        """If only one segment has enough data, DDI should be 0 or undefined."""
        # std of a single value is 0
    
    def test_all_segments_insufficient(self):
        """If no segment has enough data, DDI should report as unavailable."""
    
    def test_quality_proxy_all_zeros(self):
        """If quality proxy is 0 for all (e.g., all refusals), handle gracefully."""
    
    def test_topic_overlap(self):
        """Query matches multiple topics — should be assigned to best match only."""
        query = "I'm disabled and need help with housing benefit"
        # Could match disability_benefits OR housing_benefit
        # Should assign to highest-confidence match
```

### 7.6 Cross-Metric Integration Tests

```python
class TestCrossMetricIntegration:
    """Tests that verify metrics work together correctly."""
    
    def test_silent_model_update_triggers_trci_and_cds(self):
        """
        When the LLM is silently updated, both TRCI and CDS should detect it.
        TRCI catches it via canary divergence; CDS via response distribution shift.
        """
        # Simulate model update
        orchestrator = MetricOrchestrator(default_config)
        orchestrator.inject_model_update(response_style_shift=0.3)
        
        report = orchestrator.run_daily()
        
        assert report.trci.trci_mean < 0.95
        assert report.cds.cds_value > 0.05
        assert len(report.alerts) >= 2
    
    def test_knowledge_base_update_triggers_trci_not_ddi(self):
        """
        When the knowledge base is updated (correctly), TRCI should detect
        the change, but DDI should remain stable (update affects all segments).
        """
        orchestrator = MetricOrchestrator(default_config)
        orchestrator.inject_kb_update(uniform=True)
        
        report = orchestrator.run_weekly()
        
        assert report.trci.trci_mean < 0.95  # Canaries detect KB change
        assert report.ddi.ddi_value < 0.05    # Uniform change → low DDI
    
    def test_targeted_degradation_triggers_ddi_and_fds(self):
        """
        When one topic segment gets worse responses, DDI should detect
        non-uniformity and FDS should detect faithfulness decay.
        """
        orchestrator = MetricOrchestrator(default_config)
        orchestrator.inject_segment_degradation(
            segment="disability_benefits", 
            hallucination_rate=0.3
        )
        
        report = orchestrator.run_weekly()
        
        assert report.ddi.ddi_value > 0.05    # Non-uniform drift
        assert report.fds.fds_value < -0.02   # Faithfulness decay
    
    def test_full_system_no_drift(self):
        """
        Baseline test: stable system should have all metrics in GREEN.
        """
        orchestrator = MetricOrchestrator(default_config)
        report = orchestrator.run_weekly()
        
        assert report.trci.status.startswith("GREEN")
        assert report.cds.status.startswith("GREEN")
        assert report.fds.status.startswith("GREEN") or "within" in report.fds.status.lower()
        assert report.ddi.status.startswith("GREEN")
        assert len(report.alerts) == 0
```

---

## 8. Descriptive Indicators & Charts

### 8.1 Dashboard Page: Overview (`01_overview.py`)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  DRIFT MONITORING DASHBOARD — UK Benefits & Housing Chatbot              │
│  Last updated: 2026-02-25 08:00 UTC                                     │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ TRCI     │  │ CDS      │  │ FDS      │  │ DDI      │                │
│  │ 🟢 0.972 │  │ 🟡 0.087 │  │ 🟢-0.011 │  │ 🟢 0.034 │                │
│  │ GREEN    │  │ AMBER    │  │ GREEN    │  │ GREEN    │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
│                                                                          │
│  Active Alerts: 1                                                        │
│  ⚠️  CDS AMBER — response_length descriptor shifted (JSD=0.089)         │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  90-Day Trend (all metrics)                                      │    │
│  │                                                                  │    │
│  │  1.0 ─ ─ ─ ─ ─ ─ ─ ─── TRCI                                    │    │
│  │  0.8 ─                                                           │    │
│  │  0.6 ─                                                           │    │
│  │  0.4 ─                                                           │    │
│  │  0.2 ─          ─── CDS                                          │    │
│  │  0.0 ─ ─── FDS ─── DDI                                          │    │
│  │      ├────┼────┼────┼────┤                                       │    │
│  │      W-12  W-8  W-4  Now                                         │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  System Stats: 1,247 queries this week | 98.2% uptime | Avg latency 423ms│
└──────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Chart Catalogue

| Chart | Metric | Type | Purpose |
|---|---|---|---|
| **TRCI Canary Heatmap** | TRCI | Heatmap (topic × time) | Shows which topics' canaries are drifting over time |
| **TRCI Per-Canary Scatter** | TRCI | Scatter plot | Individual canary similarity scores, colour-coded by status |
| **CDS Descriptor Waterfall** | CDS | Waterfall/bar chart | Shows each descriptor's weighted contribution to CDS |
| **CDS Distribution Comparison** | CDS | Overlaid histograms | Side-by-side distribution of each descriptor (ref vs current) |
| **CDS Time Series** | CDS | Line chart with threshold bands | CDS value over time with GREEN/AMBER/RED zones |
| **FDS Faithfulness Distribution** | FDS | Box plot or violin | Distribution of per-response faithfulness (ref vs current) |
| **FDS Claim-Level Breakdown** | FDS | Stacked bar | Proportion of supported/unsupported/ambiguous claims per window |
| **FDS Calibration Scatter** | FDS | Scatter (judge vs ground truth) | Shows evaluator accuracy against known verdicts |
| **DDI Segment Radar** | DDI | Radar/spider chart | Per-segment drift scores overlaid |
| **DDI Segment Heatmap** | DDI | Heatmap (segment × time) | Shows which segments are drifting over time |
| **DDI Quality Proxy Distribution** | DDI | Multi-violin plot | Quality distributions per segment, ref vs current |
| **Cross-Metric Correlation** | All | Correlation matrix | Pairwise correlation between metric time series |
| **Alert Timeline** | All | Gantt-style timeline | When alerts fired, severity, resolution status |
| **Governance Traffic Light** | All | Summary table | Compliance status per UK governance requirement |

### 8.3 Interactive Chart Details

#### 8.3.1 TRCI Canary Heatmap

```python
def render_trci_heatmap(trci_history: list[TRCIResult]):
    """
    Rows: canary topics (universal_credit, housing_benefit, ...)
    Columns: probe dates
    Cells: mean similarity for that topic on that date
    Colour: gradient from red (< 0.80) to green (≥ 0.95)
    """
    import plotly.express as px
    
    data = []
    for result in trci_history:
        for canary in result.per_canary_results:
            data.append({
                "date": result.timestamp.date(),
                "topic": canary.topic,
                "similarity": canary.similarity
            })
    
    df = pd.DataFrame(data)
    pivot = df.pivot_table(values="similarity", index="topic", columns="date", aggfunc="mean")
    
    fig = px.imshow(
        pivot,
        color_continuous_scale=["red", "yellow", "green"],
        zmin=0.70, zmax=1.0,
        title="TRCI Canary Heatmap — Similarity by Topic Over Time",
        labels={"color": "Cosine Similarity"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

#### 8.3.2 CDS Descriptor Waterfall

```python
def render_cds_waterfall(cds_result: CDSResult):
    """
    Shows each descriptor's weighted contribution to the CDS total.
    Enables visual identification of the primary drift drivers.
    """
    import plotly.graph_objects as go
    
    descriptors = sorted(
        cds_result.per_descriptor.values(),
        key=lambda x: x.weighted_contribution,
        reverse=True
    )
    
    fig = go.Figure(go.Waterfall(
        name="CDS Breakdown",
        orientation="v",
        x=[d.name for d in descriptors] + ["Total"],
        y=[d.weighted_contribution for d in descriptors] + [cds_result.cds_value],
        measure=["relative"] * len(descriptors) + ["total"],
        text=[f"JSD={d.jsd:.3f}" for d in descriptors] + [f"CDS={cds_result.cds_value:.3f}"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="CDS Descriptor Contributions (Waterfall)",
        yaxis_title="Weighted JSD Contribution"
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

#### 8.3.3 DDI Segment Radar

```python
def render_ddi_radar(ddi_result: DDIResult):
    """
    Radar chart showing drift intensity per topic segment.
    Quickly reveals if one segment is disproportionately affected.
    """
    import plotly.graph_objects as go
    
    segments = [s for s in ddi_result.per_segment.values() if s.drift_score is not None]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[s.drift_score for s in segments],
        theta=[s.topic for s in segments],
        fill='toself',
        name='Current Drift (JSD)',
        line_color='steelblue'
    ))
    
    # Add threshold ring
    fig.add_trace(go.Scatterpolar(
        r=[0.15] * len(segments),
        theta=[s.topic for s in segments],
        fill='none',
        name='RED Threshold',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 0.3])),
        title="DDI — Per-Segment Drift Intensity"
    )
    
    st.plotly_chart(fig, use_container_width=True)
```

---

## 9. Explainability & Export Framework

### 9.1 Explanation Chain

Every metric result carries a **full explanation chain** — from raw data to final verdict:

```
Explanation Chain (example for CDS = 0.087, AMBER):
┌─────────────────────────────────────────────────────────────────────┐
│ Level 1: VERDICT                                                     │
│   CDS = 0.087 (AMBER). Composite drift detected.                    │
│                                                                      │
│ Level 2: BREAKDOWN                                                   │
│   Top contributors:                                                  │
│   1. response_length (JSD=0.089, weight=0.15, contribution=0.013)   │
│   2. refusal_flag (JSD=0.071, weight=0.15, contribution=0.011)      │
│   3. citation_count (JSD=0.058, weight=0.15, contribution=0.009)    │
│                                                                      │
│ Level 3: STATISTICAL DETAIL                                          │
│   response_length: ref mean=142.3 words, cur mean=98.7 words        │
│   refusal_flag: ref rate=4.2%, cur rate=11.8%                        │
│   Persistence: window 2 of 3 (not yet persistent)                    │
│                                                                      │
│ Level 4: RAW DATA                                                    │
│   Reference window: 2026-01-01 to 2026-01-31 (843 queries)          │
│   Current window: 2026-02-18 to 2026-02-25 (127 queries)            │
│   Full descriptor distributions: [exportable as CSV]                  │
│                                                                      │
│ Level 5: GOVERNANCE MAPPING                                          │
│   Relevant requirements:                                             │
│   - ICO: Accuracy chapter mandates drift monitoring ✓                │
│   - ICO: Must have documented thresholds ✓ (0.05/0.15/0.30)         │
│   - DPA Art 5(1)(d): Data accuracy obligation ✓ (monitoring active)  │
│   - AI White Paper Principle 3: Transparency ✓ (full chain exposed)  │
│                                                                      │
│ Level 6: RECOMMENDED ACTION                                          │
│   - Review response_length shift: are responses becoming truncated?  │
│   - Monitor refusal_flag trend: is the model refusing more queries?  │
│   - If CDS remains AMBER for 1 more window → will escalate to RED    │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Export Formats

| Format | Use Case | Content |
|---|---|---|
| **JSON** | Programmatic consumption, API integration | Full metric results with all intermediate data |
| **CSV** | Data analysis, spreadsheet review | Per-query / per-canary / per-descriptor tabular data |
| **PDF** | Governance reporting, board presentations | Formatted report with charts, explanations, recommendations |
| **HTML** | Shareable dashboard snapshot | Interactive charts with Plotly, self-contained file |
| **Markdown** | Documentation, audit trail | Human-readable summary for inclusion in DPIA |

### 9.3 Quarterly Governance Report Structure (PDF Export)

```markdown
# Quarterly Drift Monitoring Report
## Q1 2026 — UK Benefits & Housing AI Chatbot

### Executive Summary
- System monitored continuously for 13 weeks
- [X] TRCI probes conducted, [Y] total queries processed
- Overall compliance status: [GREEN/AMBER/RED]

### 1. TRCI Summary
- Trend chart (13 weeks)
- Notable events (model updates detected, KB refreshes)
- Canary topic breakdown

### 2. CDS Summary
- Trend chart with descriptor breakdown
- Persistent drift episodes (if any)
- Root cause analysis for each episode

### 3. FDS Summary
- Faithfulness trend
- Evaluator calibration history
- Notable hallucination incidents

### 4. DDI Summary
- Segment drift comparison
- Equality Act compliance assessment
- Intersectional analysis

### 5. Alert Log
- All alerts fired this quarter
- Resolution status and actions taken
- Mean time to resolution

### 6. Governance Compliance Matrix
- Per-requirement compliance status
- Evidence references (specific log entries)
- DPIA update recommendations

### 7. Recommendations
- Threshold adjustments (if any)
- Canary set updates needed
- Model re-evaluation timeline

### Appendix A: Full Metric History (tables)
### Appendix B: Configuration Used (all parameter values)
### Appendix C: Data Sources and Lineage
```

### 9.4 Export Implementation

```python
class ExportEngine:
    """Handles all export formats."""
    
    def export_json(self, report: Union[DailyReport, WeeklyReport, QuarterlyReport], 
                    path: str):
        """Export full report as JSON with all intermediate data."""
        with open(path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
    
    def export_csv(self, report, path: str):
        """Export tabular data as CSV files (one per metric)."""
        # TRCI: per-canary results
        pd.DataFrame([c.__dict__ for c in report.trci.per_canary_results]).to_csv(
            f"{path}/trci_canaries.csv", index=False
        )
        # CDS: per-descriptor results
        pd.DataFrame([d.__dict__ for d in report.cds.per_descriptor.values()]).to_csv(
            f"{path}/cds_descriptors.csv", index=False
        )
        # FDS: per-query results
        pd.DataFrame([q.__dict__ for q in report.fds.per_query]).to_csv(
            f"{path}/fds_queries.csv", index=False
        )
        # DDI: per-segment results
        pd.DataFrame([s.__dict__ for s in report.ddi.per_segment.values()]).to_csv(
            f"{path}/ddi_segments.csv", index=False
        )
    
    def export_pdf(self, report: QuarterlyReport, path: str):
        """Generate formatted PDF governance report."""
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Image
        
        doc = SimpleDocTemplate(path, pagesize=A4)
        story = []
        
        # Title page
        story.append(Paragraph("Quarterly Drift Monitoring Report", title_style))
        story.append(Paragraph(f"Quarter: {report.quarter}", subtitle_style))
        
        # Executive summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Paragraph(report.executive_summary(), body_style))
        
        # Metric sections with embedded charts
        for metric_name, metric_result in [
            ("TRCI", report.trci_trend),
            ("CDS", report.cds_trend),
            ("FDS", report.fds_trend),
            ("DDI", report.ddi_trend)
        ]:
            story.append(Paragraph(f"{metric_name} Summary", heading_style))
            # Save chart as PNG and embed
            chart_path = self.render_trend_chart(metric_result)
            story.append(Image(chart_path, width=450, height=200))
            story.append(Paragraph(metric_result.explanation, body_style))
        
        # Compliance matrix
        story.append(Paragraph("Governance Compliance Matrix", heading_style))
        compliance_data = report.governance_compliance.to_table()
        story.append(Table(compliance_data, style=compliance_table_style))
        
        doc.build(story)
```

---

## 10. Implementation Plan & Dependencies

### 10.1 Python Dependencies

```toml
[project]
name = "rait"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # Core
    "httpx>=0.27",                    # GOV.UK API calls
    "beautifulsoup4>=4.12",           # HTML parsing
    
    # Vector Store (Pinecone)
    "pinecone[grpc]>=5.0",            # Pinecone client (with gRPC for performance)
    # NOTE: No local embedding model needed — Pinecone integrated embedding
    # handles llama-text-embed-v2 (1024-dim) server-side
    
    # LLM Client
    "litellm>=1.40",                  # Unified LLM API client (routes to Groq)
    "openai>=1.30",                   # OpenAI SDK (Groq is OpenAI-compatible)
    
    # NLP
    "spacy>=3.7",                     # Topic extraction, text processing
    "tiktoken>=0.7",                  # Token counting
    "textstat>=0.7",                  # Readability scores
    
    # Statistical Computing
    "numpy>=1.26",
    "scipy>=1.12",
    "pandas>=2.2",
    
    # Dashboard
    "streamlit>=1.35",
    "plotly>=5.22",
    "matplotlib>=3.9",
    
    # Export
    "reportlab>=4.2",                 # PDF generation
    
    # Testing
    "pytest>=8.2",
    "hypothesis>=6.100",              # Property-based testing
    
    # Data Storage
    # SQLite is included in Python stdlib
    
    # Environment
    "python-dotenv>=1.0",             # .env file loading for API keys
]
```

### 10.2 Project Structure

```
rait/
├── docs/
│   ├── Assignment.md
│   ├── Foundational_Drift_Knowledge.md
│   ├── Metric_Design_Document.md
│   └── System_Design_Document.md          ← This document
│
├── src/
│   ├── __init__.py
│   ├── production/                         # Part A: Production System
│   │   ├── __init__.py
│   │   ├── ingestion.py                   # GOV.UK document ingestion
│   │   ├── retrieval.py                   # Vector search + reranking
│   │   ├── generation.py                  # LLM response generation
│   │   ├── logging.py                     # Production log management
│   │   └── pipeline.py                    # End-to-end query pipeline
│   │
│   ├── monitoring/                         # Part B: Monitoring System
│   │   ├── __init__.py
│   │   ├── data_collection.py             # Log → metric dataset
│   │   ├── descriptors.py                 # Descriptor extraction
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── trci.py                    # TRCI engine
│   │   │   ├── cds.py                     # CDS engine
│   │   │   ├── fds.py                     # FDS engine
│   │   │   ├── ddi.py                     # DDI engine
│   │   │   └── base.py                    # Base metric class
│   │   ├── judge.py                       # LLM-as-Judge
│   │   ├── calibration.py                 # Evaluator calibration
│   │   ├── alerts.py                      # Alert engine
│   │   ├── orchestrator.py                # Metric scheduling
│   │   └── export.py                      # JSON/CSV/PDF/HTML export
│   │
│   ├── dashboard/                          # Streamlit Dashboard
│   │   ├── app.py
│   │   ├── pages/
│   │   │   ├── 01_overview.py
│   │   │   ├── 02_trci.py
│   │   │   ├── 03_cds.py
│   │   │   ├── 04_fds.py
│   │   │   ├── 05_ddi.py
│   │   │   ├── 06_experiment.py
│   │   │   ├── 07_governance.py
│   │   │   └── 08_export.py
│   │   └── components/
│   │       ├── metric_card.py
│   │       ├── threshold_editor.py
│   │       ├── time_series.py
│   │       ├── distribution_plot.py
│   │       └── explanation_panel.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── statistics.py                  # JSD, cosine similarity, etc.
│       ├── text_processing.py             # Sentiment, readability, etc.
│       └── config.py                      # Configuration management
│
├── config/
│   ├── canary_queries.json                # TRCI canary set
│   ├── calibration_set.json               # FDS calibration data
│   ├── defaults.yaml                      # Default metric parameters
│   └── govuk_pages.yaml                   # Pages to ingest
│
├── tests/
│   ├── unit/
│   │   ├── test_trci.py
│   │   ├── test_cds.py
│   │   ├── test_fds.py
│   │   ├── test_ddi.py
│   │   └── test_statistics.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   ├── test_metric_orchestrator.py
│   │   └── test_export.py
│   ├── scenario/
│   │   ├── test_drift_scenarios.py
│   │   └── test_cross_metric.py
│   └── conftest.py                        # Shared fixtures
│
├── data/
│   ├── raw/                               # Fetched GOV.UK JSON
│   ├── processed/                         # Chunk metadata (text stored in Pinecone)
│   ├── logs/                              # Production logs (SQLite)
│   └── results/                           # Metric computation results
│
├── exports/                               # Generated reports
│
├── .env.example                           # Template for API keys
├── main.py                                # CLI entry point
├── pyproject.toml
└── README.md
```

### 10.3 Deployment Strategy

#### 10.3.1 Platform: Hugging Face Spaces (Streamlit)

| Property | Value |
|---|---|
| **Platform** | HF Spaces — Free CPU Basic tier |
| **Compute** | 16 GB RAM, 2 vCPU |
| **URL** | `https://huggingface.co/spaces/{username}/rait-chatbot` |
| **Framework** | Streamlit (auto-detected from `app.py`) |
| **Secrets** | HF Spaces Secrets panel: `PINECONE_API_KEY`, `GROQ_API_KEY` |
| **Storage** | SQLite logs stored in Space repo (persistent across restarts) |
| **Vector DB** | Pinecone Serverless (external, cloud-managed — no local storage needed) |
| **LLM** | Groq API (external, free tier) |

**Why HF Spaces works for this project:**
- Pinecone is cloud-managed → no local vector DB persistence issue
- Groq API is external → no local GPU needed
- SQLite can be committed to the Space repo for persistence
- Provides a sharable URL for assessment submission
- Zero-cost, zero-config deployment

#### 10.3.2 Environment Variables

```bash
# .env (local development) — NOT committed to git
PINECONE_API_KEY=pcsk_...          # From Pinecone dashboard
GROQ_API_KEY=gsk_...               # From Groq console
GOVUK_BASE_URL=https://www.gov.uk  # GOV.UK Content API base
```

#### 10.3.3 LLM Provider Configuration

```python
# config/llm_config.py
LLM_CONFIG = {
    "primary": {
        "provider": "groq",
        "model": "groq/llama-3.3-70b-versatile",  # litellm format
        "temperature": 0.1,
        "max_tokens": 1024,
        "description": "Primary RAG generation — high quality, GPT-4 comparable",
        "rate_limits": {"rpm": 30, "rpd": 1000, "tpm": 12000, "tpd": 100000},
    },
    "judge": {
        "provider": "groq",
        "model": "groq/llama-3.1-8b-instant",  # litellm format
        "temperature": 0.0,
        "max_tokens": 512,
        "description": "LLM-as-Judge for FDS claim verification — fast, high throughput",
        "rate_limits": {"rpm": 30, "rpd": 14400, "tpm": 6000, "tpd": 500000},
    },
    "fallback": {
        "provider": "groq",
        "model": "groq/llama-3.1-8b-instant",
        "temperature": 0.1,
        "max_tokens": 1024,
        "description": "Fallback if primary rate-limited",
    },
}
```

**UK Government Alignment:** The GOV.UK Chat experiment used OpenAI API calls with a RAG pattern on Google Cloud. Our architecture mirrors this pattern — external LLM API + cloud vector store + RAG — but uses free-tier equivalents (Groq + Pinecone) suitable for an assessment project. The Llama 3.3 70B model provides comparable quality to GPT-4 for grounded, factual responses.

#### 10.3.4 Implementation Sequence

| Phase | Duration | Activities |
|---|---|---|
| **Phase 1: Foundation** | Day 1 | Install dependencies, set up project structure, implement `utils/statistics.py` (JSD, cosine similarity), create SQLite schema, configure Pinecone + Groq API keys |
| **Phase 2: Ingestion** | Day 1-2 | Implement GOV.UK Content API ingestion, chunking (1024 tokens), Pinecone upsert |
| **Phase 3: Production Pipeline** | Day 2-3 | Implement retrieval (Pinecone query), Groq generation, logging; test with sample queries |
| **Phase 4: Metric Engines** | Day 3-5 | Implement TRCI, CDS, FDS, DDI engines with unit tests |
| **Phase 5: LLM-as-Judge** | Day 5-6 | Implement claim decomposition, verification (Llama 3.1 8B via Groq), calibration |
| **Phase 6: Dashboard** | Day 6-8 | Implement Streamlit pages, toggle controls, charts |
| **Phase 7: Testing** | Day 8-9 | Full scenario tests, cross-metric integration tests, edge cases |
| **Phase 8: Export** | Day 9-10 | Implement JSON/CSV/PDF/HTML export, quarterly report generation |
| **Phase 9: Deploy & Docs** | Day 10 | Deploy to HF Spaces, final documentation, README |

---

## 11. Appendices

### Appendix A: Governance Compliance Mapping

| # | UK Requirement | System Component | Evidence |
|---|---|---|---|
| 1 | ICO: Drift detection mandate | All 4 metrics (TRCI, CDS, FDS, DDI) | Metric computation engine |
| 2 | ICO: Documented thresholds | Toggle dashboard with configurable thresholds | `config/defaults.yaml` |
| 3 | AI White Paper: Explainability | 6-level explanation chain per metric result | Explanation panel in dashboard |
| 4 | Equality Act: Fairness monitoring | DDI with topic-as-proxy-for-protected-characteristic | DDI engine + segment radar chart |
| 5 | AI White Paper: Audit trails | Full production logging + metric result history | SQLite database + JSON exports |
| 6 | GDPR Recital 71: Appropriate statistical procedures | JSD (bounded, symmetric), cosine similarity (standard) | `utils/statistics.py` |
| 7 | ICO: Proportional monitoring | Configurable cadence (daily/weekly/quarterly) | Metric orchestrator |
| 8 | NCSC 4.1: Silent model update detection | TRCI canary probing | TRCI engine |
| 9 | Ethics Framework Point 7: Quarterly reviews | Quarterly governance report with full trend analysis | Export engine + governance page |
| 10 | ICO: Live DPIA updates | DPIA update section in quarterly report | `export.py → generate_dpia_update()` |

### Appendix B: Metric Cross-Reference

| Drift Type | Primary Metric | Secondary Metric | Detection Mechanism |
|---|---|---|---|
| Silent model update | TRCI | CDS | Canary divergence + distribution shift |
| Knowledge base change | TRCI (topic-specific) | FDS | Topic canaries + faithfulness change |
| Gradual quality decay | CDS | FDS | Descriptor drift + faithfulness trend |
| Fairness divergence | DDI | CDS (per-segment) | Cross-segment drift comparison |
| Hallucination onset | FDS | CDS (refusal_flag) | Claim verification + refusal rate |
| Evaluator meta-drift | FDS (calibration) | — | Ground truth correlation check |
| Seasonal pattern shift | CDS (multiple descriptors) | DDI | Multi-descriptor trend analysis |

### Appendix C: Canary Set Specification

| ID Range | Topic | Count | Difficulty | Example |
|---|---|---|---|---|
| C001–C010 | Universal Credit | 10 | Mixed | "What's the UC standard allowance?" |
| C011–C020 | Housing Benefit | 10 | Mixed | "Can I get HB if I'm over State Pension age?" |
| C021–C028 | Council Tax | 8 | Mixed | "How do I appeal my council tax band?" |
| C029–C036 | Disability Benefits | 8 | Mixed | "What's the difference between PIP and DLA?" |
| C037–C042 | Homelessness | 6 | Mixed | "What help is there if I'm about to be evicted?" |
| C043–C048 | Pension/Retirement | 6 | Mixed | "When can I start getting my State Pension?" |
| C049–C050 | Cross-topic | 2 | Hard | "I'm disabled, homeless, and over 65 — what benefits can I get?" |

### Appendix D: Glossary of Abbreviations

| Abbreviation | Full Name |
|---|---|
| TRCI | Temporal Response Consistency Index |
| CDS | Composite Drift Signal |
| FDS | Faithfulness Decay Score |
| DDI | Differential Drift Index |
| JSD | Jensen-Shannon Divergence |
| KDE | Kernel Density Estimation |
| RAG | Retrieval-Augmented Generation |
| OGL | Open Government Licence |
| ICO | Information Commissioner's Office |
| NCSC | National Cyber Security Centre |
| DPIA | Data Protection Impact Assessment |
| PSED | Public Sector Equality Duty |
| DPA | Data Protection Act |
| DPO | Data Protection Officer |

---

*End of System Design Document.*
