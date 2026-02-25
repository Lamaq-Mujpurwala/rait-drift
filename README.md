---
title: RAIT Drift Monitor
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RAIT — Data & Model Drift Monitoring for RAG Systems

A production-grade drift monitoring system built around a UK government benefits chatbot (RAG pipeline). It detects when an AI system silently degrades — whether from model updates, data staleness, or uneven performance across user groups.

**Live Demo:** [lamaq-rait-drift-monitoring.hf.space](https://lamaq-rait-drift-monitoring.hf.space)

## What It Does

The chatbot answers questions about UK benefits (Universal Credit, Housing Benefit, PIP, Council Tax, Pensions, Homelessness) using GOV.UK as its knowledge base. On top of this, four custom metrics continuously monitor for drift:

| Metric | What It Detects | Method |
|--------|----------------|--------|
| **TRCI** — Temporal Response Consistency Index | Silent model updates by the LLM provider | Fires 50 canary queries with known answers, measures cosine similarity drift |
| **CDS** — Composite Descriptor Shift | Changes in response characteristics (length, tone, citations, latency) | Jensen-Shannon Divergence across 9 weighted descriptors with adaptive binning |
| **FDS** — Faithfulness Decay Score | Responses becoming less faithful to source documents | Atomic claim decomposition → LLM-as-Judge verification → signed JSD |
| **DDI** — Differential Drift Index | Unfair degradation for specific user groups | Per-segment quality proxy comparison (disability, elderly, vulnerable groups) |

Each metric outputs a traffic-light status: **GREEN** (stable) → **AMBER** (investigate) → **RED** (action required).

## Architecture

```
User Query → Pinecone Retrieval → Groq LLM → Response + Logging (SQLite)
                                                    ↓
                              Monitoring Orchestrator (Daily/Weekly)
                              ├── TRCI: Canary probes via Pinecone cosine sim
                              ├── CDS:  9-descriptor JSD with persistence filter
                              ├── FDS:  Claim decomposition + Judge verification
                              └── DDI:  Per-segment quality fairness check
```

**Stack:** Python 3.11 · Pinecone (llama-text-embed-v2, 1024d) · Groq (Llama 3.3 70B + 3.1 8B) · Streamlit · SQLite · litellm

## Quick Start

```bash
# Clone and install
git clone https://github.com/Lamaq-Mujpurwala/rait-drift.git
cd rait-drift
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your PINECONE_API_KEY and GROQ_API_KEY

# Run the dashboard
streamlit run app.py

# Or use the CLI
python main.py query "What is Universal Credit?"
python main.py monitor          # Run daily monitoring
python main.py test-stats       # Offline smoke tests (no API needed)
```

## Dashboard Pages

| Page | Purpose |
|------|---------|
| 💬 **Chatbot** | Ask questions, observe per-response metadata (latency, citations, readability, topic) |
| 📊 **Monitoring** | Run daily/weekly metric cycles, view traffic-light cards, per-descriptor bar charts, canary tables |
| 🧪 **Tests** | Run 192 unit tests + CLI smoke tests + integration tests from the GUI |
| ⚙️ **System** | Verify API keys, Pinecone health, ingestion status, production log stats |

## Tests

```bash
# Run all 192 tests (no API keys needed)
pytest tests/ -v

# Individual smoke tests via CLI
python main.py test-stats        # Statistical utilities (JSD, KS, cosine)
python main.py test-logging      # SQLite round-trip
python main.py test-descriptors  # CDS descriptor extraction
python main.py test-cds          # CDS metric engine
```

## Project Structure

```
├── app.py                 # Streamlit entry point
├── main.py                # CLI interface
├── src/
│   ├── production/        # RAG pipeline (retrieval, generation, logging)
│   ├── monitoring/        # Drift detection (orchestrator, judge, metrics/)
│   │   └── metrics/       # TRCI, CDS, FDS, DDI engines
│   ├── dashboard/         # Streamlit pages (chatbot, monitoring, tests, system)
│   └── utils/             # Config, statistics, rate limiter, text processing
├── config/                # YAML defaults, canary queries, calibration set
├── tests/unit/            # 192 unit tests across all metrics
└── docs/                  # Design documents and submission
```

## Key Design Decisions

- **Adaptive binning** (Freedman-Diaconis) for JSD to handle varying sample sizes
- **Persistence filter** (P=3) in CDS to suppress transient spikes
- **Cross-validation** via Cohen's kappa between two LLM judges for FDS reliability
- **Rate limiter** with exponential backoff for Groq free-tier compliance (30 RPM)
- **Topic-as-proxy** for protected characteristics in DDI (maps to Equality Act 2010 groups)

## Documentation

Detailed write-ups for every aspect of the system are in the [`docs/`](docs/) folder:

| Document | What It Covers |
|----------|---------------|
| [Final Submission](docs/Final_Submission.md) | The main deliverable — Connect, Contextualise, Design & Justify (with formulas and confidence ratings), Operationalise |
| [System Design](docs/System_Design_Document.md) | Full architecture, data flow, schema definitions, API contracts, and configuration reference |
| [Metric Design](docs/Metric_Design_Document.md) | Deep dive into each metric's mathematical foundation, thresholds, and edge cases |
| [Testing & Evaluation](docs/Testing_and_Evaluation.md) | Test strategy, coverage breakdown, known limitations, and evaluation results |
| [Detect & Respond](docs/Detect_and_Respond.md) | How each metric maps to detection → alerting → response workflows |
| [Foundational Drift Knowledge](docs/Foundational_Drift_Knowledge.md) | Background theory on data drift, model drift, and statistical methods used |
| [GUI Playground Guide](docs/GUI_Playground_Guide.md) | Hands-on walkthrough of the Streamlit dashboard with sample queries and expected results |

## License

Academic project — RAIT Intern Assessment 2026.
