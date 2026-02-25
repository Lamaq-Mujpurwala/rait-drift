# Data & Model Drift — Metric Design Document

**Candidate**: Lamaq  
**Compliance Question**: *"How does the organisation detect and respond to changes in data or model behaviour over time to ensure the system remains valid, unbiased, and fit for purpose?"*  
**System**: UK public sector AI chatbot (RAG-based) answering citizen queries on housing, benefits, and local services  

---

## Table of Contents

1. [Part 1: Invariant Drift Concepts — What Generalises Beyond Our System](#part-1-invariant-drift-concepts)
2. [Part 2: The Metric Design Process — From Concepts to Computation](#part-2-the-metric-design-process)
3. [Part 3: UK Governance Compliance Framework](#part-3-uk-governance-compliance-framework)
4. [Part 4: Four Metric Designs](#part-4-four-metric-designs)
5. [Part 5: Novel & Experimental Approaches](#part-5-novel--experimental-approaches)
6. [Part 6: Confidence Evaluation](#part-6-confidence-evaluation)

---

## Part 1: Invariant Drift Concepts

### 1.1 The Invariance Criterion

The question: which drift concepts remain valid regardless of whether the system has more or fewer components, different architecture, different scale, or different domain? A concept is **invariant** if it can be defined purely in terms of the abstract input-output mapping of the system, without referring to any specific internal component.

Mathematically, any system can be modelled as a mapping $f: X \rightarrow \hat{Y}$ where:
- $X$ = input space (queries)
- $\hat{Y}$ = output space (responses)
- $Y$ = ground truth space (correct responses, when they exist)
- $f$ = the entire system (a black box — everything between query arrival and response delivery)

The joint distribution $P(X, \hat{Y}, Y)$ can decompose in multiple useful ways:
- $P(X, Y) = P(X) \cdot P(Y|X)$ — input distribution and true input-output relationship
- $P(X, \hat{Y}) = P(X) \cdot P(\hat{Y}|X)$ — input distribution and system's learned mapping
- $P(\hat{Y}, Y | X)$ — the relationship between what the system says and what's correct, for a given input

**Drift occurs when any of these distributions change over time.** The question is: which components of this decomposition are invariant to system architecture?

### 1.2 Classification: Invariant vs. Architecture-Dependent

| Concept | Mathematical Form | Invariant? | Reasoning |
|---|---|---|---|
| **Input Distribution Shift** | $P_t(X) \neq P_{t-1}(X)$ | ✅ Yes | Every system has inputs. Input distributions always defined. |
| **Output Distribution Shift** | $P_t(\hat{Y}) \neq P_{t-1}(\hat{Y})$ | ✅ Yes | Every system produces outputs. Output characteristics always measurable. |
| **System Behaviour Drift** | $P_t(\hat{Y}|X) \neq P_{t-1}(\hat{Y}|X)$ | ✅ Yes | For same inputs, does the system respond differently over time? Always defined. |
| **Concept Drift** | $P_t(Y|X) \neq P_{t-1}(Y|X)$| ✅ Yes | The correct answer for a query changes. Independent of system architecture — it's a property of the *world*, not the system. |
| **Ground Truth Distribution Shift** | $P_t(Y) \neq P_{t-1}(Y)$ | ✅ Yes | Distribution of correct answers shifts. A world property. |
| **Performance Drift** | $\mathbb{E}[\ell(\hat{Y}, Y)]$ changes over time | ✅ Yes* | Any loss function between outputs and ground truth can degrade. *Requires partial ground truth. |
| **Retrieval Relevance Drift** | Distribution of retrieval scores shifts | ❌ No | Requires a retrieval component. Not all systems have one. |
| **Embedding Space Drift** | Query/doc embeddings shift in vector space | ❌ No | Requires access to specific embedding representations. |
| **Knowledge Base Staleness** | Source documents become outdated | ❌ No | Requires a document store. Not all systems have one. |
| **Pipeline Component Drift** | Specific preprocessing/postprocessing changes | ❌ No | Requires knowledge of pipeline components. |
| **Retrieval-Generation Misalignment** | Retrieved docs and generated response diverge | ❌ No | Requires RAG-specific logging. |

### 1.3 The Five Invariant Drift Primitives

From the above, we derive **five primitive drift types** that are system-invariant:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INVARIANT DRIFT PRIMITIVES                    │
│                                                                  │
│   These hold for ANY system mapping inputs to outputs            │
│                                                                  │
│   1. INPUT DRIFT          P(X) changes                          │
│      → Who is asking, what are they asking, how they ask it      │
│                                                                  │
│   2. OUTPUT DRIFT         P(Ŷ) changes                          │
│      → Response characteristics shift regardless of input        │
│                                                                  │
│   3. BEHAVIOUR DRIFT      P(Ŷ|X) changes                       │
│      → Same inputs get different outputs over time               │
│                                                                  │
│   4. CONCEPT DRIFT        P(Y|X) changes                        │
│      → The correct answer for a query changes (world change)     │
│                                                                  │
│   5. PERFORMANCE DRIFT    E[L(Ŷ, Y)] worsens over time          │
│      → System accuracy degrades (requires ground truth)          │
│                                                                  │
│   NOTE: Primitives 1-4 are fully observable from production.     │
│   Primitive 5 requires partial ground truth (which we have).     │
└─────────────────────────────────────────────────────────────────┘
```

**Critical insight**: The compliance question asks about the system remaining "valid, unbiased, and fit for purpose." These three requirements map onto our invariant primitives:

| Compliance Requirement | Primary Invariant Primitive | Secondary |
|---|---|---|
| **Valid** (factually correct) | Concept Drift + Performance Drift | Behaviour Drift |
| **Unbiased** (fair across groups) | Behaviour Drift (differential across segments) | Input Drift (demographic shift) |
| **Fit for purpose** (useful) | Output Drift + Behaviour Drift | Performance Drift |

### 1.4 Why Invariance Matters for Metric Design

If we design metrics anchored to these five invariant primitives, they have several properties:

1. **Transferability**: The same metric framework works for a different chatbot, a classification system, a recommendation engine — any input-output system.
2. **Robustness to architectural change**: If the RAG pipeline is swapped for a different architecture (e.g., fine-tuned model without retrieval), the metrics still apply.
3. **Auditability**: External auditors can evaluate the metrics without knowing system internals — they only need input/output data.
4. **UK Governance Alignment**: The ICO explicitly says monitoring should be based on observable system behaviour and outputs, not just internal model diagnostics.

### 1.5 Mapping Our System's Specific Drift Types to Invariant Primitives

From our foundational research (30+ drift types identified), here's how specific drifts map back to invariant primitives:

| Invariant Primitive | System-Specific Manifestations |
|---|---|
| **P(X) — Input Drift** | Topic distribution shift, vocabulary drift, query complexity change, demographic shift, intent distribution change, seasonal query patterns, adversarial input patterns |
| **P(Ŷ) — Output Drift** | Response length drift, confidence/hedging drift, refusal rate drift, citation pattern drift, tone/style drift, error pattern drift |
| **P(Ŷ\|X) — Behaviour Drift** | Silent LLM update effects (Chen et al.: GPT-4 accuracy 84%→51%), instruction-following degradation, hallucination rate change, retrieval-to-response alignment change |
| **P(Y\|X) — Concept Drift** | Policy change (eligibility thresholds), legislation update, procedural change, jurisdictional change, seasonal rule changes |
| **E[L(Ŷ,Y)] — Performance Drift** | Accuracy on golden test set degrades, faithfulness score drops, user satisfaction declines, escalation rate increases |

---

## Part 2: The Metric Design Process

### 2.1 Design Criteria for Drift Metrics

Before designing specific metrics, we establish what makes a good drift metric for this context. Each metric must satisfy **all** of the following:

| Criterion | Description | Source/Rationale |
|---|---|---|
| **Computability** | Can be computed solely from available data (queries, responses, API access, partial ground truth, system performance metrics) | Assignment constraint |
| **Scalarity** | Produces or can be reduced to a single scalar value with justified derivation | Assignment requirement |
| **Interpretability** | The number it produces can be explained to a non-technical stakeholder (regulator, ethics board member) | UK AI White Paper: transparency principle; ICO: explainability |
| **Sensitivity** | Detects meaningful drift early enough to act upon | ICO: "proportional to the impact an incorrect output may have on individuals" |
| **Specificity** | Low false alarm rate — doesn't trigger on benign variation (seasonal patterns, noise) | Operational necessity |
| **Actionability** | When it fires, the organisation knows what kind of drift is occurring and what to investigate | Compliance question: "detect AND respond" |
| **Temporal Awareness** | Can distinguish between abrupt, gradual, recurrent, and blip drift patterns | Foundational research: slide concepts |
| **Fairness Sensitivity** | Can reveal differential drift across user segments | Compliance question: "unbiased" |
| **Regulatory Defensibility** | Can be justified under UK legal frameworks (DPA 2018, Equality Act 2010, ICO guidance) | Legal requirement |

### 2.2 The Design Process: From Invariant Primitives to Computable Metrics

```
INVARIANT PRIMITIVE
        │
        ▼
OBSERVABLE PROXY — What can we actually measure from available data?
        │
        ▼
STATISTICAL FORMULATION — How do we quantify change in this proxy?
        │
        ▼
SCALAR DERIVATION — How do we collapse this into a single number?
        │
        ▼
THRESHOLD CALIBRATION — When does this number indicate a problem?
        │
        ▼
OPERATIONAL PROTOCOL — What happens when the threshold is breached?
```

**Step 1: Choose the invariant primitive(s) the metric addresses.**

**Step 2: Identify observable proxies.** Since we can't directly observe $P(Y|X)$ (concept drift) without full ground truth, we must identify proxy signals. For $P(\hat{Y}|X)$ (behaviour drift), we can directly observe query-response pairs. The key challenge is that most interesting drift types require proxy measurement.

**Step 3: Select statistical method.** Match the method to the data type:
- Categorical distributions (topic bins) → PSI, JSD, Chi-squared
- Continuous distributions (embedding dimensions, scores) → KS test, Wasserstein
- High-dimensional distributions (full embeddings) → Domain classifier, MMD
- Time series (streaming scalar metrics) → CUSUM, ADWIN, Page-Hinkley
- Structured comparison (reference vs. current) → JSD (symmetric, bounded, preferred)

**Step 4: Derive scalar.** If the method produces a distribution, vector, or set of per-dimension results:
- **Aggregation**: Mean, weighted mean, or max across sub-components
- **Proportion**: Fraction of components exceeding a threshold (e.g., "23% of embedding dimensions drifted")
- **Summary statistic**: A single test statistic (KS D-statistic, ROC AUC, etc.)

**Step 5: Set thresholds.** Two approaches:
- **Absolute thresholds**: Based on literature or domain knowledge (e.g., PSI > 0.2 = significant drift)
- **Relative thresholds**: Based on the metric's own historical distribution (e.g., > 2σ from rolling mean)
- **Recommended**: Use relative thresholds with absolute bounds as safety nets

**Step 6: Define response protocol.** Per ICO guidance, documented actions that follow threshold breach.

### 2.3 Available Data Inventory (Strict)

| Data Source | What We Can Extract | Relevant Primitives |
|---|---|---|
| **User queries** (text) | Embeddings (via model API), length, vocabulary, topic clusters, sentiment, OOV rate, format, language patterns | P(X) |
| **LLM responses** (text) | Embeddings, length, sentiment, hedging language, refusal indicators, entity mentions, claim density | P(Ŷ), P(Ŷ\|X) |
| **Model API** (inference access) | Can send probe/canary queries; can compute embeddings; can generate responses on demand | P(Ŷ\|X), active probing |
| **Partial ground truth** (human-annotated) | Correctness labels for a subset of queries; enables direct quality measurement on that subset | P(Y\|X), E[L(Ŷ,Y)] |
| **System performance metrics** | Response latency, throughput, error rates, uptime | Operational drift |

### 2.4 What We Explicitly Cannot Do

- **No RAG-specific logs**: Cannot measure retrieval scores, retrieved document IDs, retrieval latency separately, or document-level relevance.
- **No model internals**: Cannot inspect attention weights, token probabilities, hidden states, or gradient information.
- **No demographic data on users**: Cannot directly observe protected characteristics. Any fairness analysis must use topic-based or query-characteristic-based segmentation as proxies.
- **No full ground truth**: Cannot compute accuracy, precision, or recall on the full production set. Limited to the annotated subset.

---

## Part 3: UK Governance Compliance Framework

### 3.1 Legal and Regulatory Landscape

The following UK laws, policies, and technical guidelines are directly relevant to our drift monitoring metrics. Each is mapped to specific requirements our metrics must satisfy.

#### A. UK GDPR & Data Protection Act 2018

| Provision | Requirement | Metric Implication |
|---|---|---|
| **Article 5(1)(d)** — Accuracy Principle | Personal data must be accurate and kept up to date | System must not provide outdated or inaccurate personal data; drift monitoring must catch accuracy degradation |
| **Article 22** — Automated Decision-Making | Right not to be subject to solely automated decisions with legal/significant effects | If the chatbot's response affects benefits eligibility, the user has rights to contest; drift that causes wrong decisions must be detectable |
| **Article 35** — Data Protection Impact Assessment | Mandatory DPIA for high-risk processing | DPIAs must be "live documents" (ICO) — drift metrics feed DPIA updates |
| **Recital 71** — Appropriate Safeguards | "Appropriate mathematical or statistical procedures" must be used | Our statistical drift detection methods must be justifiable and appropriate |
| **Section 14 DPA 2018** — Automated Decision-Making Safeguards | Controller must notify data subject of automated decision and allow representations | Drift that silently degrades automated decisions must be caught before harm occurs |

#### B. Equality Act 2010

| Provision | Requirement | Metric Implication |
|---|---|---|
| **Section 149** — Public Sector Equality Duty | Must have due regard to: eliminate discrimination, advance equality, foster good relations | Drift monitoring must include fairness dimension — detect if drift differentially affects protected groups |
| **Protected Characteristics** | Age, disability, gender reassignment, marriage/civil partnership, pregnancy, race, religion, sex, sexual orientation | Cannot directly measure (no demographic data), but can use topic/query-characteristic proxies: queries about disability benefits, maternity services, etc. |

#### C. UK AI White Paper — Pro-Innovation Approach (2023)

Five cross-sectoral principles that all regulators are expected to apply:

| Principle | Requirement | Metric Implication |
|---|---|---|
| **1. Safety, Security, Robustness** | AI must function robustly and not pose unreasonable safety risks | Drift that degrades safety (wrong medical/legal advice) must be detectable; robustness includes temporal robustness |
| **2. Transparency & Explainability** | Appropriate transparency about when AI is being used and how it works | Drift metrics themselves must be explainable; cannot use opaque drift detection methods |
| **3. Fairness** | AI must not undermine legal rights or discriminate unfairly | Must monitor for differential drift across user segments — aligns with Equality Act |
| **4. Accountability & Governance** | Clear lines of responsibility; governance appropriate to level of risk | Documented thresholds, escalation procedures, audit trails for all drift metrics |
| **5. Contestability & Redress** | People must be able to challenge AI outcomes | If drift causes wrong answers, the error must be detectable and reversible; historical metric data enables after-the-fact review |

#### D. GOV.UK Generative AI Framework for HMG (10 Principles)

| Principle | Requirement | Direct Metric Implication |
|---|---|---|
| **Principle 5** — "Meaningful human control" + drift monitoring | *"Review the performance of generative AI tools against predefined criteria to detect for bias, drift and hallucinations"* | Explicit mandate for drift monitoring metrics |
| **Principle 5** (continued) | *"Record, maintain and monitor outputs (including prompts and responses)"* | Historical data must be retained for trend analysis |
| **Principle 5** (continued) | *"Obtain user feedback to understand the usefulness of the returned response"* | User feedback signals must be incorporated into drift detection |
| **Principle 8** — Testing | *"Beyond initial testing... AI models' performance in production must be monitored continuously"* | Continuous production monitoring required, not just pre-deployment testing |

#### E. GOV.UK Ethics, Transparency and Accountability Framework

| Point | Requirement | Metric Implication |
|---|---|---|
| **Point 7** — Quarterly Reviews | Mandates quarterly review of AI system performance | Minimum cadence for comprehensive drift assessment; metrics must support quarterly reporting |

#### F. ICO Guidance on AI and Data Protection

| Chapter | Specific Requirements | Metric Implication |
|---|---|---|
| **Accuracy (Ch. 4)** | *"You should regularly assess drift and retrain the model on new data where necessary"* — explicit mandate | All four metrics must detect drift that degrades accuracy |
| **Accuracy (Ch. 4)** | *"Decide and document appropriate thresholds for determining whether your model needs to be retrained"* | Each metric must have documented, justified thresholds |
| **Accuracy (Ch. 4)** | Monitoring frequency should be *"proportional to the impact an incorrect output may have on individuals"* | Higher-impact topics (benefits eligibility, housing) need more frequent monitoring |
| **Accuracy (Ch. 4)** | Distinguishes data protection "accuracy" (personal data correct) from "statistical accuracy" (AI system correctness) | Metrics must address both: system correctness AND ensuring personal data accuracy |
| **Accountability (Ch. 1)** | DPIA as a *"live document... reviewing the DPIA regularly... the demographics of the target population may shift"* | DPIA updates triggered by metric thresholds; concept drift explicitly named |
| **Accountability (Ch. 1)** | Must assess *"allocative harms AND representational harms"* | Allocative: wrong benefit amount. Representational: stereotyping in responses. Both must be monitored. |
| **Accountability (Ch. 1)** | Trade-off management: *"accuracy vs data minimisation; accuracy vs fairness; explainability vs accuracy"* | Metrics should not create perverse incentives; must be balanced |
| **Accountability (Ch. 1)** | Outsourced AI: *"agree regular updates and reviews of statistical accuracy to guard against changing population data and concept/model drift"* | Since we use an external LLM API, this applies directly — SLA must include drift monitoring |

#### G. NCSC Machine Learning Principles (v2.0, May 2024)

| Principle | Requirement | Metric Implication |
|---|---|---|
| **4.1** — Continual Learning Risks | *"Understand and mitigate risks of continual learning"* | LLM provider may update the model (continual learning from their side); our metrics must detect these silent updates |
| **4.2** — Input Sanitisation | Appropriately sanitise inputs to model in use | Monitor for adversarial input patterns as part of input drift |
| **4.3** — Incident Management | Develop incident and vulnerability management processes | Drift detection → incident response workflow must be defined |

#### H. ISO/IEC Standards (Referenced in UK AI White Paper)

| Standard | Relevance |
|---|---|
| **ISO/IEC 42001** — AI Management System | Framework for managing AI systems throughout lifecycle; drift monitoring is a lifecycle concern |
| **ISO/IEC 23894** — AI Risk Management | Risk-based approach to AI governance; drift is a risk to be managed |
| **ISO/IEC TS 6254** — AI Objectives and Approaches for Explainability | Drift metrics must be explainable per this standard |

### 3.2 Compliance Requirements Synthesis

From the above, we distil the **non-negotiable requirements** our metrics must satisfy:

1. **Must include drift detection** (ICO explicit mandate, GOV.UK GenAI Principle 5)
2. **Must have documented thresholds** (ICO explicit mandate)
3. **Must be explainable** (UK AI White Paper Principle 2, ISO/IEC TS 6254)
4. **Must monitor for fairness/bias drift** (Equality Act 2010, UK AI White Paper Principle 3, ICO allocative + representational harms)
5. **Must support audit trails** (UK AI White Paper Principle 4, GOV.UK Ethics Framework)
6. **Must use appropriate statistical procedures** (UK GDPR Recital 71)
7. **Must monitor proportionally to impact** (ICO guidance)
8. **Must detect silent model updates** (NCSC Principle 4.1 — continual learning risks)
9. **Must support quarterly review cadence** minimum (GOV.UK Ethics Framework Point 7)
10. **Must feed into live DPIA updates** (ICO accountability guidance)

---

## Part 4: Four Metric Designs

### Overview

The four metrics below are designed to cover the compliance question's three requirements — **valid**, **unbiased**, **fit for purpose** — while satisfying all UK governance requirements from Part 3.

| Metric | Primary Invariant Primitive | Compliance Dimension | Novelty |
|---|---|---|---|
| **Temporal Response Consistency Index (TRCI)** | P(Ŷ\|X) — Behaviour Drift | Valid + Fit for Purpose | Active canary probing adapted from software engineering |
| **Composite Drift Signal (CDS)** | P(X) + P(Ŷ) — Input + Output Drift | Fit for Purpose (holistic) | Information-theoretic fusion of heterogeneous signals |
| **Faithfulness Decay Score (FDS)** | P(Ŷ\|X) → E[L(Ŷ,Y)] | Valid (factual accuracy) | Atomic claim decomposition applied as temporal drift mechanism |
| **Differential Drift Index (DDI)** | P(Ŷ\|X) conditional on segment | Unbiased | Cross-segment drift comparison for bias detection |

Together they form a **complete monitoring framework**:

```
                        COMPLIANCE QUESTION
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
       VALID              UNBIASED          FIT FOR PURPOSE
          │                    │                    │
     ┌────┴────┐              │             ┌──────┴──────┐
     │         │              │             │             │
   TRCI      FDS            DDI           CDS          TRCI
  (proactive) (factual)   (fairness)   (holistic)   (consistency)
```

---

### Metric 1: Temporal Response Consistency Index (TRCI)

#### Definition

The TRCI measures **how much the system's responses to known, fixed queries change over time**. It operationalises the invariant primitive $P_t(\hat{Y}|X) \neq P_{t-1}(\hat{Y}|X)$ by actively probing the system with a maintained set of "canary queries" and measuring response divergence from established reference responses.

The name "canary" is borrowed from canary deployments in software engineering — where a small, controlled test is used to detect problems before they affect the full population.

#### What It Detects

- **Silent LLM model updates** (Chen et al. 2023: GPT-4 accuracy dropped 84%→51% between versions)
- **RAG pipeline changes** (retrieval degradation, index corruption)
- **Knowledge base updates** (new or modified source documents changing responses)
- **Prompt template modifications** (if system prompts are adjusted)
- Any change that causes the system to respond differently to the same input

#### Data Required

| Data | Source | Availability |
|---|---|---|
| Canary query set $Q_C = \{q_1, ..., q_n\}$ | Curated from partial ground truth + high-frequency production topics | Derived from available data |
| Reference responses $R_{ref}(q_i)$ | Generated during a validated baseline period | Generated via model API |
| Current responses $R_{cur}(q_i)$ | Generated by sending canary queries through the live system | Generated via model API |
| Embedding function $\phi$ | Model API (embedding endpoint) | Available (model client) |

#### Computation

**Step 1: Establish Canary Query Set**

Select $n$ queries (recommended: 50-200) covering:
- All major topic areas (housing, benefits, local services, etc.)
- Different query complexities (simple factual, multi-step procedural, eligibility determination)
- Edge cases captured from production (ambiguous queries, multi-topic queries)
- Queries with known ground truth (from the partial ground truth set)

For each canary query $q_i$, record a **reference response** $r_i^{ref}$ during a validated baseline period where the system was confirmed to be performing correctly.

**Step 2: Periodic Probing**

At regular intervals (recommended: daily for high-impact topics, weekly for comprehensive set), send each canary query through the live system and record the current response $r_i^{cur}$.

**Step 3: Compute Per-Query Consistency Score**

For each canary query $q_i$, compute the semantic similarity between the reference and current responses:

$$c_i = \text{cosine\_similarity}\left(\phi(r_i^{ref}),\ \phi(r_i^{cur})\right)$$

where $\phi$ is the embedding function (accessible via model API).

**Step 4: Aggregate to Scalar**

$$\text{TRCI} = \frac{1}{n} \sum_{i=1}^{n} c_i$$

The TRCI is a **stability score**: values near 1.0 indicate the system is responding consistently; values dropping toward 0 indicate responses are changing.

**Alternative aggregation** (more sensitive to outliers — catches cases where a few queries drift severely):

$$\text{TRCI}_{min} = \min_{i \in \{1,...,n\}} c_i$$

$$\text{TRCI}_{p10} = \text{10th percentile of } \{c_1, ..., c_n\}$$

**Recommended**: Report both $\text{TRCI}_{mean}$ and $\text{TRCI}_{p10}$, using $\text{TRCI}_{p10}$ for alerting (a severe drop in even a few queries warrants investigation).

#### Topic-Level Decomposition

To make the metric actionable, also compute per-topic TRCI:

$$\text{TRCI}_{topic_k} = \frac{1}{|Q_k|} \sum_{q_i \in Q_k} c_i$$

where $Q_k$ is the subset of canary queries in topic cluster $k$. This immediately tells the operator *which topic area* is drifting.

#### Thresholds

| TRCI Value | Interpretation | Action |
|---|---|---|
| ≥ 0.95 | No meaningful drift | Continue monitoring |
| 0.90 – 0.95 | Minor drift detected | Log for quarterly review; investigate if trend is downward |
| 0.80 – 0.90 | Moderate drift | Investigate root cause; compare against ground truth on affected topics; escalate to technical team |
| < 0.80 | Severe drift | Immediate investigation; consider pausing affected topic areas; mandatory DPIA review (ICO requirement) |

Thresholds should be calibrated against the system's own historical TRCI distribution during the first 30 days of operation. The absolute thresholds above serve as **safety bounds** per ICO guidance to "document appropriate thresholds."

#### Justification

1. **Addresses the compliance question directly**: Detects changes in model behaviour over time (the "detect" part) at a cadence that enables response (the "respond" part).
2. **Proactive, not reactive**: Unlike production-data-based drift detection, TRCI doesn't wait for user harm — it tests the system before users encounter problems. This is analogous to regression testing in software engineering.
3. **ICO compliant**: *"You should regularly assess drift"* — TRCI operationalises this with defined frequency. *"Document appropriate thresholds"* — thresholds are defined above.
4. **UK AI White Paper compliant**: Principle 1 (safety — catches dangerous drift early), Principle 2 (transparency — fully explainable metric), Principle 4 (accountability — creates audit trail).
5. **GOV.UK GenAI Framework compliant**: Principle 5 explicitly mandates *"review the performance of generative AI tools against predefined criteria to detect for bias, drift and hallucinations"* — TRCI is predefined criteria applied at regular intervals.
6. **NCSC compliant**: Principle 4.1 (continual learning risks) — TRCI detects when the LLM provider silently updates the model.
7. **Empirically motivated**: Chen et al. (2023) demonstrated that LLM behaviour can change drastically between API versions. TRCI is designed specifically to catch this.

#### Limitations

1. **Canary set coverage**: The canary queries may not cover all drift types. Novel query patterns not represented in the canary set will be missed.
2. **Maintenance burden**: The canary set must be updated as the system's domain evolves (new policies, new services). If canary queries become unrepresentative, the metric loses validity.
3. **Semantic similarity ≠ correctness**: Two responses can be semantically different but both correct (e.g., different phrasing of the same information). TRCI may flag "drift" that is actually benign rephrasing. Mitigation: use the ground-truth-labelled subset of canary queries to validate flagged drift.
4. **Legitimate change**: When policies actually change, TRCI *should* detect a change — but this is a true positive, not a false alarm. The reference responses need to be updated when legitimate changes occur. This creates an operational workflow requirement.
5. **Embedding quality**: The metric's sensitivity depends on the quality of the embedding model. If the embedding model itself changes (provider update), TRCI can produce spurious signals. Mitigation: use a locally-controlled embedding model for consistency scoring, separate from the production embedding pipeline.

#### Confidence Rating: **4 / 5**

**Why 4**: Strong theoretical basis (invariant primitive P(Ŷ|X), empirically motivated by Chen et al.), clearly computable from available data, well-aligned with UK governance requirements, interpretable and explainable. The canary testing concept is well-established in software engineering and translates naturally to LLM monitoring.

**Why not 5**: The canary set coverage problem is real — we cannot guarantee coverage of all drift types. The maintenance burden is non-trivial. The distinction between "the model is saying something different" and "the model is saying something wrong" requires additional investigation beyond this metric alone.

---

### Metric 2: Composite Drift Signal (CDS)

#### Definition

The CDS fuses **multiple heterogeneous drift signals** extracted from production data into a single, interpretable composite indicator. It operationalises the invariant primitives $P(X)$ (input drift) and $P(\hat{Y})$ (output drift) by monitoring several text properties of queries and responses simultaneously and combining their drift scores using Jensen-Shannon Divergence with information-theoretic weighting.

The core idea: individual weak signals (query length shifted slightly, response sentiment changed a little, follow-up rate ticked up) may each be within noise, but **their conjunction** is strong evidence of drift. CDS captures this.

#### What It Detects

- **Input distribution drift**: Topic shifts, vocabulary changes, query complexity changes, seasonal patterns
- **Output distribution drift**: Response length changes, tone shifts, hedging pattern changes, refusal rate changes
- **Combined input-output drift**: Cases where both input and output properties shift simultaneously (possibly correlated)
- **Gradual drift**: Small changes across many signals accumulate into a detectable composite signal even when individual signals are below threshold

#### Data Required

| Data | Source | Availability |
|---|---|---|
| Production query texts | User queries (logged) | Available |
| Production response texts | LLM responses (logged) | Available |
| User feedback signals | System performance metrics (if thumbs up/down, follow-up rate, escalation rate are tracked) | Available (system performance metrics) |
| Historical baseline distributions | Computed from a validated reference period | Derived |

#### Computation

**Step 1: Extract Text Descriptors**

For each query $q$ and response $r$ in a time window $W$, extract $K$ descriptors:

*Query-side descriptors (input drift signals):*

| Descriptor | Notation | Computation |
|---|---|---|
| Query length (words) | $d_1^q$ | Word count |
| Query vocabulary richness | $d_2^q$ | Type-token ratio |
| Query OOV (out-of-vocabulary) rate | $d_3^q$ | Fraction of words not in baseline vocabulary |
| Query sentiment score | $d_4^q$ | Sentiment model output (via model API) |
| Query topic distribution | $d_5^q$ | Topic cluster assignment probabilities |

*Response-side descriptors (output drift signals):*

| Descriptor | Notation | Computation |
|---|---|---|
| Response length (words) | $d_1^r$ | Word count |
| Response hedging rate | $d_2^r$ | Frequency of uncertainty markers ("might", "possibly", "I'm not sure") |
| Response refusal rate | $d_3^r$ | Binary: does response decline to answer? |
| Response sentiment score | $d_4^r$ | Sentiment model output |
| Response entity density | $d_5^r$ | Count of named entities (GOV.UK services, policy names, etc.) per sentence |

*Interaction-level descriptors (if available from system performance metrics):*

| Descriptor | Notation | Computation |
|---|---|---|
| Follow-up query rate | $d_1^{int}$ | Fraction of sessions with >1 query |
| Response latency | $d_2^{int}$ | API response time |

**Step 2: Compute Per-Descriptor Drift Scores**

For each descriptor $d_k$, compare its distribution in the current window $W_{cur}$ against a reference window $W_{ref}$ using Jensen-Shannon Divergence:

$$\delta_k = JSD\left(P_{ref}(d_k)\ \|\ P_{cur}(d_k)\right) = \frac{1}{2} D_{KL}(P_{ref} \| M) + \frac{1}{2} D_{KL}(P_{cur} \| M)$$

where $M = \frac{1}{2}(P_{ref} + P_{cur})$.

For continuous descriptors, discretise into bins (e.g., decile bins from the reference distribution) before computing JSD.

JSD is chosen because:
- **Symmetric**: Neither time window is privileged as "truth" — appropriate since we compare production windows
- **Bounded**: $JSD \in [0, \ln 2]$ (natural log) or $[0, 1]$ (log base 2) — enables comparison across descriptors
- **Defined everywhere**: Unlike KL divergence, JSD doesn't blow up when one bin has zero probability
- **UK GDPR Recital 71 compliant**: An established, appropriate statistical procedure

Normalise to $[0, 1]$ using log base 2: $\hat{\delta}_k = JSD_{base2}(P_{ref}(d_k) \| P_{cur}(d_k)) \in [0, 1]$.

**Step 3: Weight and Fuse**

Assign weights $w_k$ to each descriptor based on **signal reliability and relevance to the compliance question**:

| Signal Type | Weight Rationale | Recommended Weight Band |
|---|---|---|
| Ground-truth-validated signals (e.g., accuracy on golden set, if available as a descriptor) | Directly measures system correctness | $w = 0.25 - 0.35$ |
| Explicit user feedback (satisfaction, escalation) | Strong quality proxy | $w = 0.20 - 0.25$ |
| Response-side text properties (length, hedging, entities) | Observable output drift | $w = 0.15 - 0.20$ |
| Query-side text properties (length, OOV, sentiment) | Input drift (necessary context but doesn't directly indicate quality) | $w = 0.10 - 0.15$ |
| Operational metrics (latency) | Indirect quality signal | $w = 0.05 - 0.10$ |

Weights must sum to 1: $\sum_k w_k = 1$.

Compute the Composite Drift Signal:

$$\text{CDS} = \sum_{k=1}^{K} w_k \cdot \hat{\delta}_k$$

**The CDS is a single scalar in $[0, 1]$** where 0 = no drift detected across any signal, and 1 = maximum drift across all signals.

**Step 4: Contextualize with Per-Signal Breakdown**

While CDS produces a scalar, the per-signal breakdown $\{\hat{\delta}_1, \hat{\delta}_2, ..., \hat{\delta}_K\}$ is retained for diagnostic purposes. When CDS breaches a threshold, the operator inspects which signals contributed most.

#### Thresholds

| CDS Value | Interpretation | Action |
|---|---|---|
| < 0.05 | Nominal — system stable | Continue monitoring |
| 0.05 – 0.15 | Minor drift — likely noise or benign variation | Log; check if consistent across consecutive windows |
| 0.15 – 0.30 | Moderate drift — one or more signals deviating | Investigate top contributing signals; cross-reference with TRCI |
| > 0.30 | Significant drift — multiple signals converging | Trigger full investigation; run TRCI with expanded canary set; evaluate with FDS; consider pausing if DDI shows bias |

**Persistence requirement**: To avoid false alarms from transient blips, require CDS > threshold for **at least 2 consecutive windows** before escalating beyond "log" action.

#### Justification

1. **Holistic detection**: The compliance question asks about "changes in data or model behaviour" — CDS captures both input-side (data) and output-side (model behaviour) drift in a single indicator.
2. **Resilience to subtle drift**: Individual signals may be within noise, but if 8 out of 12 descriptors each shift slightly in the same direction, CDS aggregates these weak signals into a detectable composite. This is the "conjunction of weak signals" principle.
3. **Interpretable decomposition**: While the scalar CDS enables dashboard monitoring and threshold alerting, the per-signal breakdown enables root cause analysis. This satisfies UK AI White Paper Principle 2 (transparency) — the metric is not a black box.
4. **JSD is principled**: Symmetric, bounded, well-established in information theory — satisfies UK GDPR Recital 71 requirement for "appropriate statistical procedures." Also used in standard practice (EvidentlyAI, NannyML).
5. **Adapts to available data**: If some descriptors are unavailable (e.g., no explicit user feedback), the remaining descriptors still produce a valid CDS — just re-normalise weights. This makes the metric **robust to data availability changes**.
6. **ICO compliance**: Operationalises *"regularly assess drift"* with the defined windowing and frequency. The documented thresholds satisfy ICO's explicit requirement.
7. **GOV.UK GenAI Framework**: Principle 5 mentions monitoring *"prompts and responses"* — CDS does exactly this by extracting descriptors from both.

#### Limitations

1. **Weight selection is subjective**: The weight assignment reflects our judgement about signal reliability. Different weight choices will produce different CDS values and different sensitivity profiles. Mitigation: sensitivity analysis — report CDS under multiple weight configurations; validate weights against known drift events.
2. **JSD on discretised distributions is sensitive to binning**: The choice of bin boundaries affects the computed divergence. Mitigation: use quantile-based bins from the reference distribution (deciles recommended), which are robust to distribution shape.
3. **Reference window decay**: The reference window itself becomes outdated over time. If the reference is from 6 months ago, even legitimate evolution of the system will trigger drift signals. Mitigation: use a **rolling reference** (e.g., 30-day rolling window, compared against the prior 30 days) for routine monitoring, with a **fixed baseline** for quarterly reviews per Ethics Framework Point 7.
4. **Correlation between signals**: Descriptors may be correlated (e.g., longer queries often have higher OOV rates). Weighted sum may double-count correlated drift. Mitigation: could decorrelate signals (PCA on descriptor drift scores), but this reduces interpretability. The trade-off favours interpretability per our design criteria.
5. **Does not identify drift direction**: CDS tells you "how much" but not "better or worse." A system that improves will also show drift. Mitigation: pair with TRCI and FDS for directionality.

#### Confidence Rating: **3 / 5**

**Why 3**: The individual components (JSD on text descriptors) are well-established in the literature (EvidentlyAI text descriptor monitoring). The concept of multi-signal fusion is sound. However:
- The weighting scheme is not empirically validated — it's a reasoned proposal, not a proven approach
- The specific descriptors chosen may not be optimal — a real deployment would require iteration
- The threshold values need calibration against real system data, which we don't have

**What would raise it to 4**: Empirical validation on a real system showing that CDS detects known drift events with acceptable false alarm rate. Robust weight selection methodology (e.g., validated against historical drift incidents).

---

### Metric 3: Faithfulness Decay Score (FDS)

#### Definition

The FDS measures **the rate at which the system's responses become less grounded in verifiable knowledge over time**. It operationalises the transition from $P(\hat{Y}|X)$ (behaviour drift) to $E[L(\hat{Y}, Y)]$ (performance drift) by decomposing responses into atomic claims, using the model API to assess their verifiability, and tracking this faithfulness distribution's drift over time.

This metric specifically targets the "valid" dimension of the compliance question — it catches the most dangerous form of drift: **the system confidently giving answers that are no longer (or never were) supported by authoritative sources**.

**Theoretical inspiration**: Adapted from Min et al. (2023) FActScore framework and Zhang et al. (2023) hallucination taxonomy, applied as a *temporal drift detection mechanism* rather than a one-shot evaluation tool.

#### What It Detects

- **Hallucination rate drift**: LLM starts generating more unsupported claims over time
- **Knowledge staleness drift**: Responses cite or rely on outdated information that's no longer in the knowledge base
- **Grounding degradation**: Responses become more generic, less specific, less tied to authoritative sources
- **Silent model update effects**: A model update may cause the LLM to rely more on parametric knowledge vs. retrieved context

#### Data Required

| Data | Source | Availability |
|---|---|---|
| Production response texts (sample) | LLM responses | Available |
| Corresponding query texts | User queries | Available |
| Model API for claim evaluation | Model client | Available (API access for inference) |
| Partial ground truth | Human-annotated correct responses for subset | Available |

**Important**: We do NOT have RAG-specific logs (retrieved documents). FDS must work without knowing which documents were retrieved. It uses the model API to assess whether claims in the response are verifiable — essentially using the LLM-as-Judge paradigm.

#### Computation

**Step 1: Sample Selection**

From each monitoring window $W$, sample $m$ query-response pairs $(q_j, r_j)$. Sampling should be stratified by topic to ensure coverage. Recommended: $m = 100-200$ per window.

**Step 2: Atomic Claim Decomposition**

For each response $r_j$, use the model API (LLM) to decompose it into atomic claims:

*Prompt template*:
```
Given the following response to a citizen's query about government services, 
list each distinct factual claim made in the response as a separate item.
A claim is a single assertion that can be independently verified as true or false.

Query: {query}
Response: {response}

List each atomic claim:
```

This produces a set of atomic claims $A_j = \{a_{j,1}, a_{j,2}, ..., a_{j,p}\}$ for each response.

**Step 3: Claim Verifiability Assessment**

For each atomic claim $a_{j,l}$, use the model API to assess whether the claim is verifiable/faithful:

*Prompt template*:
```
You are evaluating the factual accuracy of a government chatbot response.

Original citizen query: {query}
Claim to verify: {claim}

Considering UK government policies and public services, assess this claim:
1. Is this claim specific enough to be verifiable? (Yes/No)
2. Is this claim consistent with current UK government policy and services? (Supported/Unsupported/Cannot Determine)
3. Confidence in your assessment (High/Medium/Low)

Respond in the format:
Verifiable: [Yes/No]
Status: [Supported/Unsupported/Cannot Determine]
Confidence: [High/Medium/Low]
```

**Step 4: Compute Per-Response Faithfulness Score**

For each response $r_j$:

$$f_j = \frac{|\{a \in A_j : \text{Status}(a) = \text{Supported}\}|}{|A_j|}$$

This is the proportion of atomic claims in the response that are assessed as supported. $f_j \in [0, 1]$.

**Step 5: Compute Window-Level Faithfulness Distribution**

For window $W_t$, the faithfulness distribution is $F_t = \{f_1, f_2, ..., f_m\}$.

**Step 6: Compute Drift**

Compare the faithfulness distribution of the current window against the reference:

$$\text{FDS}_t = JSD(F_{ref},\ F_t)$$

Additionally, compute the **directional component** — has mean faithfulness increased or decreased?

$$\Delta\bar{f}_t = \bar{f}_{ref} - \bar{f}_t$$

Positive $\Delta\bar{f}_t$ = faithfulness has decreased (concerning). Negative = increased (improving).

**Step 7: Final Scalar**

The FDS combines drift magnitude with direction:

$$\text{FDS}_{final} = \text{sign}(\Delta\bar{f}_t) \cdot JSD(F_{ref}, F_t)$$

- **Positive FDS**: Faithfulness is decreasing (drift toward less grounded responses — **danger signal**)
- **Negative FDS**: Faithfulness is increasing (drift toward more grounded responses — likely benign)
- **Near zero**: No meaningful drift in faithfulness

#### Validation Against Partial Ground Truth

For the subset of queries where human-annotated ground truth exists, compute a **validated faithfulness score**:

$$f_j^{val} = \frac{|\{a \in A_j : a \text{ is consistent with human ground truth}\}|}{|A_j|}$$

Compare $f_j^{val}$ (ground-truth-validated) against $f_j$ (LLM-assessed) to calibrate the LLM-as-Judge component. Track the correlation $\rho(f^{val}, f)$ over time — if this correlation itself drifts, the evaluator is losing calibration ("criteria drift" per Shankar et al. 2024).

#### Thresholds

| FDS Value | Interpretation | Action |
|---|---|---|
| FDS ∈ [-0.02, 0.02] | No meaningful faithfulness drift | Continue monitoring |
| FDS ∈ (0.02, 0.10] | Mild faithfulness decrease | Investigate which topics are affected; increase sampling rate |
| FDS > 0.10 | Significant faithfulness decrease | Full investigation; run TRCI to cross-validate; check if LLM provider announced model changes; consider escalating to human review of affected responses |
| FDS < -0.02 | Faithfulness increasing | Likely benign (system improving), but verify that improvement is real and not just the evaluator becoming less strict |

#### Justification

1. **Directly addresses "valid"**: The compliance question's "valid" requirement means the system gives factually correct responses. FDS measures exactly this by tracking whether responses remain grounded in verifiable claims over time.
2. **Catches the most dangerous drift type**: A system that hallucrinates more over time may continue to *appear* functional (response length, latency, format all look normal) while delivering increasingly incorrect information. FDS catches this silent degradation.
3. **ICO compliance**: ICO accuracy chapter explicitly states statistical accuracy of AI must be monitored, and distinguishes AI accuracy from data protection accuracy. FDS measures AI statistical accuracy in a temporal context.
4. **DPA 2018 / UK GDPR Article 5(1)(d)**: Accuracy principle requires information to be "accurate and, where necessary, kept up to date." FDS monitors this for AI-generated information.
5. **Recital 71 compliance**: The claim decomposition + verification approach constitutes an "appropriate statistical procedure" for assessing accuracy.
6. **Novel application**: While FActScore (Min et al. 2023) was designed as a one-shot evaluation, applying it as a temporal drift detection mechanism is novel. The contribution is not the claim verification methodology but its application as a distributional drift signal.
7. **Works without RAG-specific logs**: Despite not having access to retrieved documents, FDS uses the model API itself to perform verification — making it architecture-invariant (works whether the system uses RAG, fine-tuning, or any other approach).

#### Limitations

1. **LLM-as-Judge reliability**: The claim verification step uses an LLM to evaluate an LLM's output. This creates a dependency on the evaluator's accuracy. If the evaluating LLM itself is wrong, FDS will be miscalibrated.
   - **Mitigation**: Use the partial ground truth subset to continuously calibrate the evaluator. Track $\rho(f^{val}, f)$ and alert if correlation drops.
   - **Further mitigation**: Use a different model for evaluation than the production model (if possible within API access constraints).

2. **Criteria drift in the evaluator** (Shankar et al. 2024): The evaluation LLM's own behaviour may drift over time (since it's also accessed via API). This is a meta-drift problem.
   - **Mitigation**: The ground-truth calibration check catches this indirectly. If the evaluator becomes less strict, $\rho(f^{val}, f)$ will decrease, triggering an alert.

3. **Cost and latency**: Decomposing responses into atomic claims and verifying each requires multiple API calls per response. For 200 responses per window with an average of 5 claims each, that's ~1000 evaluation API calls per window.
   - **Mitigation**: This is a batch/periodic metric, not real-time. Run weekly or at quarterly review cadence. The cost is bounded and predictable.

4. **Cannot distinguish "never correct" from "became incorrect"**: FDS measures faithfulness at a point in time. A low faithfulness score could mean the system was always unfaithful (not drift — just poor quality) or that it became unfaithful (actual drift). The JSD comparison against the reference window handles this — drift is the change, not the absolute level.

5. **Sensitive to prompt engineering**: The quality of claim decomposition and verification depends heavily on the prompt templates used. Different prompts may yield different results.
   - **Mitigation**: Fix prompts during the baseline period and do not change them. The prompts themselves become part of the metric's documented specification.

#### Confidence Rating: **3 / 5**

**Why 3**: The theoretical foundation is strong (FActScore, hallucination research, ICO mandate for accuracy monitoring). The concept of tracking faithfulness as a temporal signal is sound and novel. However:
- LLM-as-Judge reliability is a known concern — the metric's validity depends on the evaluator being reasonably accurate
- The meta-drift problem (evaluator itself drifting) is acknowledged but not fully solved
- Ground truth validation partially addresses this, but the partial coverage means blind spots exist
- No empirical validation of threshold values

**What would raise it to 4**: Empirical demonstration that FDS detects known faithfulness degradation events (e.g., before/after a model update that increased hallucination). Strong correlation between $f^{val}$ and $f$ validated over multiple time periods.

---

### Metric 4: Differential Drift Index (DDI)

#### Definition

The DDI measures **whether drift affects different user segments or topic areas unequally**. It operationalises the invariant primitive $P(\hat{Y}|X)$ conditioned on segment membership: if the system's response quality degrades more for one user group than another, that constitutes **bias drift** — a shift from fairness to unfairness over time.

This metric specifically targets the "unbiased" dimension of the compliance question.

#### What It Detects

- **Bias emergence**: The system was equally good across segments but drift causes one segment to degrade faster
- **Bias amplification**: A pre-existing quality gap between segments widens over time
- **Bias reversal**: The system was better for segment A, but drift causes it to become better for segment B
- **Topic-specific degradation**: A policy change affects responses about disability benefits but not housing — DDI catches this unequal impact

#### Segmentation Strategy

Since we do **not** have demographic data on users (a strict data availability constraint), we cannot directly segment by protected characteristics. Instead, we use **topic-based segmentation as a proxy for population segments**:

| Topic Cluster | Proxy For | Rationale |
|---|---|---|
| Disability benefits queries | Disability (protected characteristic) | Queries about disability-related services predominantly come from or concern disabled individuals |
| Maternity/parental queries | Pregnancy/maternity (protected characteristic) | Directly related to protected characteristic |
| Housing/homelessness queries | Socioeconomic vulnerability | Disproportionately affects certain demographic groups |
| Pension/retirement queries | Age (protected characteristic) | Predominantly relevant to older adults |
| Immigration/visa queries | Race/nationality (related to protected characteristic) | Disproportionately relevant to certain racial/ethnic groups |
| General local services | Broad population | Baseline/comparison segment |

This proxy approach is explicitly justified under the Equality Act 2010's Public Sector Equality Duty, which requires *"due regard"* to fairness impacts — topic-based analysis is a reasonable approach when direct demographic data is unavailable.

#### Data Required

| Data | Source | Availability |
|---|---|---|
| Production query texts | User queries | Available — needed for topic clustering |
| Production response texts | LLM responses | Available — needed for quality assessment per segment |
| Topic cluster assignments | Derived from query embeddings (via model API) | Derived |
| Response quality proxy scores | Derived from response analysis | Derived |

#### Computation

**Step 1: Segment Assignment**

Assign each query $q$ to a topic segment $s \in S = \{s_1, s_2, ..., s_G\}$ using embedding-based clustering (via model API) or keyword-based rules:

$$s(q) = \arg\max_{s_g} P(s_g | \phi(q))$$

**Step 2: Compute Per-Segment Quality Proxy**

For each segment $s_g$ in time window $W_t$, compute a quality proxy score. Since we don't have full ground truth per segment, use a composite of available signals:

$$\text{Quality}_{s_g, t} = \alpha \cdot \bar{L}_{s_g, t} + \beta \cdot (1 - \bar{H}_{s_g, t}) + \gamma \cdot \bar{S}_{s_g, t}$$

Where:
- $\bar{L}_{s_g, t}$ = normalised mean response length for segment $s_g$ at time $t$ (very short responses often indicate failure; normalise against expected length for the topic)
- $\bar{H}_{s_g, t}$ = mean hedging rate for segment $s_g$ at time $t$ (high hedging = uncertainty = potential quality issue)
- $\bar{S}_{s_g, t}$ = mean user satisfaction signal for segment $s_g$ at time $t$ (if available from system performance metrics; otherwise omit and re-normalise)
- $\alpha + \beta + \gamma = 1$

**Alternative (simpler)**: Use the faithfulness score from FDS if available, computed per-segment.

**Step 3: Compute Per-Segment Drift**

For each segment, compute how much its quality has changed from the reference:

$$\Delta_{s_g} = \text{Quality}_{s_g, ref} - \text{Quality}_{s_g, t}$$

Positive $\Delta_{s_g}$ = quality decreased for segment $s_g$.

**Step 4: Compute Differential Drift**

The DDI measures the **dispersion of drift across segments** — the idea is that if all segments drift equally, there's no differential (bias) concern, even if there's overall drift. But if some segments drift much more than others, that's a fairness issue.

$$\text{DDI}_t = \text{std}\left(\{\Delta_{s_1}, \Delta_{s_2}, ..., \Delta_{s_G}\}\right)$$

Alternatively, use the **max-min gap** for interpretability:

$$\text{DDI}_{range} = \max_{g}(\Delta_{s_g}) - \min_{g}(\Delta_{s_g})$$

This directly answers: "What is the largest difference in quality drift between any two segments?"

**Step 5: Identify Most-Affected Segment**

$$s^* = \arg\max_{s_g} \Delta_{s_g}$$

Report which segment has experienced the most negative drift — this is immediately actionable.

#### Scalar Output

DDI produces a single scalar (standard deviation or range of cross-segment drift). Additionally report:
- The **worst-affected segment** $s^*$
- The **best-performing segment** relative to baseline
- A binary flag: **any segment with $\Delta_{s_g} > \tau$ while others have $\Delta_{s_g} < \tau$?** (This catches the specific case where one group is harmed while others are fine.)

#### Thresholds

| DDI (range) | Interpretation | Action |
|---|---|---|
| < 0.05 | Uniform drift — no differential fairness concern | Monitor; overall drift may still be an issue but it's not creating bias |
| 0.05 – 0.15 | Moderate differential — some segments affected more | Investigate which segments are diverging; check if protected characteristic proxies are involved |
| > 0.15 | Significant differential drift — potential bias concern | Immediate investigation; Equality Act 2010 compliance review; DPIA update required; consider pausing system for affected segments |

**Intersectional check**: If the worst-affected segment $s^*$ corresponds to a protected characteristic proxy (disability, maternity, age), escalate regardless of DDI magnitude.

#### Justification

1. **Directly addresses "unbiased"**: The compliance question requires the system to remain "unbiased." DDI specifically measures whether drift creates or amplifies disparities.
2. **Equality Act 2010 compliance**: Section 149 Public Sector Equality Duty requires "due regard" to eliminating discrimination and advancing equality. DDI operationalises this for an AI system — it monitors whether the system's quality is shifting unequally across groups that proxy for protected characteristics.
3. **ICO compliance**: ICO accountability guidance requires assessing both *"allocative harms AND representational harms."* DDI catches allocative harms (wrong benefit information for one group) by detecting quality drops per segment.
4. **UK AI White Paper Principle 3 (Fairness)**: *"AI must not undermine legal rights or discriminate unfairly."* DDI monitors for emerging unfair discrimination over time — catching bias that develops post-deployment, not just bias at launch.
5. **Fills a gap left by other metrics**: TRCI detects overall behaviour change, CDS detects holistic drift, FDS detects faithfulness degradation — but none of them examine whether the drift is *unequal*. DDI does.
6. **Novel contribution**: Most drift monitoring frameworks treat the population as homogeneous. DDI explicitly segments the population and compares drift *across* segments — asking not "is there drift?" but "is there *more* drift for some groups than others?"

#### Limitations

1. **Topic ≠ demographic**: Using topic clusters as proxies for protected characteristics is imperfect. A query about disability benefits could come from a carer, a policy researcher, or the disabled person themselves. The proxy relationship is noisy.
   - **Mitigation**: Acknowledge this limitation explicitly. The metric provides *"due regard"* evidence (Equality Act language) — it doesn't prove discrimination but it flags differential impact for investigation.

2. **Quality proxy imperfection**: The quality score used per segment is itself an approximation (composite of response length, hedging, satisfaction). It may not accurately reflect actual response quality for each segment.
   - **Mitigation**: For segments with partial ground truth available, validate the quality proxy against actual correctness. Prioritise ground truth annotation for segments corresponding to protected characteristic proxies.

3. **Small sample sizes per segment**: If a topic segment has few queries in a given window, the per-segment quality estimate will be noisy. DDI may produce false signals from small-sample variance.
   - **Mitigation**: Set minimum sample size per segment (e.g., $n_{min} = 30$). For segments below this threshold, exclude from DDI computation and flag for manual review instead.

4. **Legitimate differential drift**: Some drift may be legitimately differential — e.g., a policy change affects disability benefits but not housing. The system should respond differently to disability queries after the policy change. DDI would flag this as differential drift, which is a true positive in terms of detection but not necessarily a fairness issue.
   - **Mitigation**: When DDI flags differential drift, the response protocol must include investigation of whether the drift is policy-driven (legitimate) or system-driven (concerning).

5. **Does not detect bias that exists at baseline**: DDI measures *change* in cross-segment quality. If the system is already biased at the reference period, DDI won't catch that — it only catches drift from the biased baseline.
   - **Mitigation**: The baseline itself should be audited for fairness before being established as the reference. DDI then monitors for *additional* bias emergence over time.

#### Confidence Rating: **3 / 5**

**Why 3**: The concept is well-motivated by UK governance requirements (Equality Act, PSED, ICO) and fills a genuine gap in standard drift monitoring approaches. The topic-as-proxy-for-demographics approach is a reasonable workaround for the data constraint. However:
- The proxy relationship between topics and protected characteristics is imperfect and not validated
- The quality proxy used per segment is multi-layered approximation (proxy of a proxy)
- Small sample sizes per segment may limit statistical power
- The distinction between legitimate differential drift and bias-inducing drift requires human judgement

**What would raise it to 4**: Empirical validation that topic-based segments do indeed proxy for differential impact on protected groups. Larger sample sizes enabling statistically robust per-segment analysis. Integration with actual Equality Impact Assessment processes.

---

## Part 5: Novel & Experimental Approaches

### 5.1 Established Approaches Used in Our Metrics

Before discussing novel approaches, here's what our four metrics already incorporate:

| Approach | Where Used | Novelty Level |
|---|---|---|
| JSD for distributional drift detection | CDS, FDS | Established — standard in drift detection literature |
| Text descriptors for NLP monitoring | CDS | Established — EvidentlyAI recommends this |
| LLM-as-Judge for evaluation | FDS | Emerging — growing body of work, some concerns |
| Topic-based segmentation for fairness | DDI | Novel application — not standard in drift literature |
| Canary query probing for LLM monitoring | TRCI | Novel adaptation — borrowed from software engineering |
| Atomic claim decomposition for drift | FDS | Novel application — FActScore used as temporal signal |

### 5.2 Genuinely Novel Design Choices in Our Framework

#### A. Active Probing as Drift Detection (TRCI)

**Why this is novel**: The standard approach to drift detection is passive — monitor production data as it flows through the system. TRCI inverts this: it **actively queries** the system with controlled inputs to detect drift.

**Analogy**: In software engineering, canary deployments send a small fraction of traffic through a new release to detect problems. TRCI sends fixed "canary queries" through the live system to detect behaviour changes. This has been discussed for web services (A/B testing) but is underexplored for LLM monitoring specifically.

**Advantage over passive monitoring**: Active probing controls the input distribution — it eliminates input drift as a confound. If the responses to the same canary queries change, the change is necessarily in the system (model, retrieval, pipeline), not in the users.

**Why it doesn't break the rules**: We have API access for inference. Sending queries through the API is explicitly within our data access constraints.

#### B. Multi-Signal Fusion with Weighted JSD (CDS)

**Why this is novel**: Standard practice monitors individual signals independently and expects the human to mentally integrate them. CDS formally fuses heterogeneous signals into a single indicator with documented, justified weights.

**The information-theoretic framing is novel**: Each descriptor contributes a JSD value (bits of divergence from the reference), and CDS computes the weighted sum. This means CDS has an information-theoretic interpretation: "the total weighted information divergence from the reference state, measured in bits."

**Experimental extension** (not yet in our design): Use **mutual information** between each descriptor's drift score and the ground truth error rate (on the annotated subset) to learn optimal weights. This would replace the hand-tuned weights with data-driven weights:

$$w_k^* \propto I(\hat{\delta}_k;\  \mathbb{1}[\text{error on ground truth}])$$

This is genuinely experimental — it requires sufficient ground truth data and drift events to estimate mutual information, which we may not have initially.

#### C. Atomic Claim Drift Tracking (FDS)

**Why this is novel**: FActScore (Min et al. 2023) was designed as a one-shot evaluation tool — evaluate a response's factual precision at a single point in time. We repurpose this into a **temporal drift detection mechanism** by computing it over sliding windows and applying distributional tests (JSD) to the resulting score distributions.

The contribution is not the claim verification step (that's adapted from literature) but the insight that faithfulness is a **dynamic property that drifts** — and that its drift is the signal we should monitor, not its absolute level.

**Experimental extension**: Instead of binary claim assessment (supported/unsupported), compute a **claim confidence trajectory** — track how the verifier's confidence in specific recurring claims changes over time. This could detect subtle knowledge staleness:

- "Universal Credit rate is £X" → the verifier's confidence slowly decreases as the referenced rate becomes outdated
- This would constitute a leading indicator of concept drift, potentially detecting staleness before it causes harm

#### D. Cross-Segment Drift Comparison for Bias Detection (DDI)

**Why this is novel**: Most drift detection frameworks treat the user population as a monolith. DDI explicitly segments the population and asks: "is drift uniform or differential?" This reframes drift detection as a fairness monitoring problem — measuring not just *whether* drift occurs, but *whether it disproportionately affects certain groups*.

This is novel in the LLM monitoring space. Classical ML fairness metrics (demographic parity, equalised odds) are computed at a single point in time. DDI is a **temporal fairness metric** — it monitors how the fairness properties of the system evolve.

### 5.3 Unexplored Experimental Approaches (Beyond Current Design)

The following approaches are genuinely experimental — they represent promising directions that could complement our four metrics but have not been validated:

#### E1. Self-Consistency Probing for Stability Detection

**Idea**: Send the same query $q$ through the system $k$ times (e.g., $k=10$) and measure the variance of the responses. If the system is deterministic (temperature=0), all responses should be identical. If there's variance, it reveals stochasticity in the pipeline. Track this variance over time.

$$\text{Self-Consistency}(q, t) = \frac{1}{\binom{k}{2}} \sum_{i<j} \text{cosine\_sim}(\phi(r_i), \phi(r_j))$$

**Why experimental**: For deterministic systems, this is trivially 1.0. For stochastic systems (temperature > 0), the baseline variance is non-zero, and detecting meaningful *changes* in variance (vs. inherent stochasticity) is methodologically challenging.

**Risk**: High API cost (k calls per probe query). May not be informative if the system is deterministic.

#### E2. Temporal Embedding Trajectory Analysis

**Idea**: Track the centroid of query embeddings and response embeddings over time. Plot them as trajectories in a reduced-dimensional space (e.g., PCA to 2D). Analyse trajectory properties:
- **Velocity**: How fast is the centroid moving? (drift rate)
- **Direction**: Which direction is it moving? (toward what topic area?)
- **Curvature**: Is the trajectory changing direction? (drift pattern change)

$$\vec{v}_t = \bar{\phi}(Q_t) - \bar{\phi}(Q_{t-1})$$

$$\text{Drift Velocity}_t = \|\vec{v}_t\|_2$$

**Why experimental**: The interpretation of trajectory properties in embedding space is not well-established. PCA reduction may lose important information. The relationship between embedding trajectory velocity and actual system quality degradation is an open question.

**Potential**: Could provide intuitive visualisations for stakeholders ("the system is moving away from its intended operating region").

#### E3. Synthetic Ground Truth Generation for Continuous Evaluation

**Idea**: Since ground truth is partial, use the model API during the baseline period (when the system is verified to be working correctly) to generate synthetic ground truth for a large set of queries. Then use these synthetic labels for ongoing drift detection.

**Why experimental**: The synthetic ground truth inherits the biases and errors of the baseline model. If the baseline model was already wrong about something, the synthetic ground truth will encode that error. This creates a risk of entrenching existing biases.

**Mitigation**: Only use synthetic ground truth for *relative* comparisons (drift detection), never for absolute quality assessment. The synthetic labels don't need to be correct — they need to be *consistent* with the baseline system's behaviour, so that deviations from them indicate system change.

#### E4. Information-Theoretic Anomaly Score

**Idea**: Compute the conditional entropy $H(\hat{Y}|X)$ over time. If the system is behaving consistently, the mapping from queries to responses has stable entropy. If drift occurs, entropy may increase (responses become less predictable for given inputs) or decrease (responses become more templated/generic).

$$\hat{H}_t(\hat{Y}|X) = -\frac{1}{|W_t|} \sum_{(q,r) \in W_t} \log P(r|q)$$

Approximated using the LLM's own log-probability outputs (if available) or embedding-based density estimation.

**Why experimental**: Requires either log-probabilities from the LLM API (which may not be available for all providers) or density estimation in embedding space (which is computationally expensive and may not be well-calibrated). The interpretation of entropy changes is also ambiguous — increased entropy could mean either more diverse (good) or more random (bad) responses.

### 5.4 Approaches Explicitly Avoided

| Approach | Why Avoided |
|---|---|
| **Traditional accuracy/precision/recall on full data** | No full ground truth; assignment guidance says avoid classification metrics |
| **Model internal monitoring (attention, gradients)** | No access to model internals |
| **RAG-specific metrics (retrieval recall, NDCG)** | No access to RAG-specific logs |
| **Demographic-specific fairness metrics** | No access to user demographic data |
| **Real-time per-query drift scoring** | Drift is a distributional property — it only makes sense over populations, not individual queries |

---

## Part 6: Confidence Evaluation

### 6.1 Per-Metric Confidence Summary

| Metric | Rating | Reasoning Summary |
|---|---|---|
| **TRCI** — Temporal Response Consistency Index | **4/5** | Strong theoretical basis, clearly computable, well-aligned with governance requirements, empirically motivated by Chen et al. Minor gap: canary set coverage, maintenance burden. |
| **CDS** — Composite Drift Signal | **3/5** | Sound concept, individual components well-established. Gap: weighting scheme not empirically validated, threshold calibration requires real data. |
| **FDS** — Faithfulness Decay Score | **3/5** | Novel and important concept, targets most dangerous drift type. Gap: LLM-as-Judge reliability, meta-drift problem, cost. |
| **DDI** — Differential Drift Index | **3/5** | Well-motivated by Equality Act, fills genuine gap. Gap: topic-demographic proxy validity, quality proxy approximation, small sample sizes. |

### 6.2 Honest Calibration Discussion

**Why no metric gets a 5**: A rating of 5 means "I could defend this in a client meeting." To reach that level, each metric would need:
- Empirical validation on a real system demonstrating detection of known drift events
- Calibrated thresholds from production data
- Peer review of the statistical methodology
- Demonstrated false alarm rate
- Evidence that remedial actions triggered by the metric lead to improved outcomes

We have none of these — our metrics are well-reasoned proposals, not battle-tested systems. Claiming 5 would be overconfident.

**Why TRCI gets 4 and others get 3**: TRCI has the strongest connection between concept and computation.The "send the same query, see if the response changes" logic is immediately intuitive and mechanistically transparent. The link between Chen et al.'s empirical findings and our metric design is direct. The main uncertainty is operational (canary set maintenance), not conceptual.

The other three metrics involve more layers of approximation:
- CDS requires choosing weights (subjective), choosing descriptors (incomplete), and trusting JSD on discretised distributions (sensitive to binning)
- FDS requires trusting LLM-as-Judge (unresolved meta-drift), and cost is a practical concern
- DDI requires trusting topic-demographic proxies (imperfect) and per-segment quality proxies (noisy)

**What would raise each metric's confidence**:

| Metric | Current → Target | What's Needed |
|---|---|---|
| TRCI: 4 → 5 | Empirical validation; demonstrated resistance to false alarms; canary set coverage analysis |
| CDS: 3 → 4 | Data-driven weight selection; sensitivity analysis; threshold calibration on real system |
| FDS: 3 → 4 | Demonstrated evaluator stability; $\rho(f^{val}, f)$ > 0.8 sustained over multiple periods |
| DDI: 3 → 4 | Validated proxy relationships; sufficient per-segment sample sizes; integration with EIA process |

### 6.3 Cross-Metric Confidence

The **combined confidence** of the four-metric framework is higher than any individual metric. The metrics complement each other:

| Failure Mode | Which Metric Catches It | Backup |
|---|---|---|
| Silent LLM update | TRCI (canary responses change) | CDS (output descriptors shift) |
| Knowledge base staleness | FDS (faithfulness decreases) | TRCI (responses to canary queries change) |
| Bias emergence | DDI (differential drift) | CDS (segment-level signal shift if decomposed by topic) |
| Gradual quality degradation | CDS (slow multi-signal accumulation) | FDS (faithfulness trend) |
| Abrupt policy change | TRCI (sudden canary response change) | CDS (JSD spike) |
| Input distribution shift (e.g., demographic change) | CDS (query-side descriptors) | DDI (segment proportions change) |
| Adversarial input patterns | CDS (OOV rate, query format descriptors) | - (TRCI won't catch this since canary queries are controlled) |

**Combined framework confidence: 4/5** — the redundancy and complementarity of the four metrics significantly reduces the risk of missing important drift types.

### 6.4 Governance Compliance Confidence

| Requirement | Addressed By | Confidence |
|---|---|---|
| ICO: Regular drift assessment | All four metrics with defined cadences | **5/5** — explicitly operationalised |
| ICO: Documented thresholds | All four metrics with documented thresholds | **5/5** — explicitly defined |
| ICO: Proportional monitoring | Tiered monitoring (daily TRCI, weekly CDS/FDS, periodic DDI) | **4/5** — cadence proposed but not all topic-specific |
| Equality Act 2010: PSED | DDI with topic-demographic proxies | **3/5** — proxy approach is reasonable but imperfect |
| UK AI White Paper: Transparency | All metrics are explainable and interpretable | **4/5** — CDS weight rationale could be stronger |
| UK AI White Paper: Accountability | Audit trail of all metric computations | **4/5** — specified but implementation details TBD |
| DPA 2018 Art 35: DPIA | Metrics feed DPIA updates | **4/5** — connection specified but workflow needs operationalisation |
| GOV.UK GenAI Principle 5 | All four metrics | **5/5** — directly implements principle |
| Ethics Framework Point 7: Quarterly reviews | All metrics support quarterly cadence | **5/5** — explicitly designed for this |
| NCSC 4.1: Continual learning risks | TRCI detects silent model updates | **4/5** — TRCI is well-suited but canary coverage limits apply |
| Recital 71: Appropriate statistical procedures | JSD, cosine similarity, standard deviation | **5/5** — established, well-understood methods |

---

## Appendix A: Complete UK Governance Source Map

| Source | Document | Key Sections | URL |
|---|---|---|---|
| UK GDPR / DPA 2018 | Data Protection Act 2018 | Art 5(1)(d), Art 22, Art 35, Recital 71, Section 14 | legislation.gov.uk |
| Equality Act 2010 | Equality Act 2010 | Section 149, Schedule 1 (protected characteristics) | legislation.gov.uk |
| UK Government | AI Regulation White Paper (2023) | 5 cross-sectoral principles, Box 3.1 (central functions) | gov.uk/government/publications/ai-regulation-a-pro-innovation-approach |
| UK Government | Generative AI Framework for HMG | 10 principles, especially Principle 5, 8 | gov.uk/government/publications/generative-ai-framework-for-hmg |
| UK Government | Ethics, Transparency and Accountability Framework | 7 points, especially Point 7 (quarterly reviews) | gov.uk |
| ICO | Guidance on AI and Data Protection | Chapters on Accuracy, Accountability, Fairness | ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/ |
| NCSC | Principles for the Security of Machine Learning v2.0 | Part 4: Secure Operation (4.1, 4.2, 4.3) | ncsc.gov.uk/collection/machine-learning |
| ISO/IEC | 42001, 23894, TS 6254 | Referenced in UK AI White Paper | iso.org |

## Appendix B: Research Paper References

| Paper | Key Finding for Our Metrics | Citation |
|---|---|---|
| Chen, Zaharia & Zou (2023) | LLM behaviour drift is real: GPT-4 accuracy 84%→51% | arXiv:2307.09009 |
| Lu, Liu et al. (2020) | Concept drift taxonomy and detection methods (DDM, ADWIN) | arXiv:2004.05785 |
| EvidentlyAI (2023) | Domain classifier recommended for embedding drift detection | evidently.ai blog |
| Min et al. (2023) | FActScore: atomic claim decomposition for factual precision | arXiv:2310.07521 |
| Shankar et al. (2024) | Criteria drift in LLM evaluation (meta-drift problem) | arXiv:2404.12272 |
| Zhang et al. (2023) | Hallucination taxonomy: input-conflicting, context-conflicting, fact-conflicting | arXiv:2309.01219 |
| Gao et al. (2024) | RAG survey: Naive → Advanced → Modular RAG paradigms | arXiv:2312.10997 |
| Yu et al. (2023) | Chain-of-Note: RAG robustness to noisy retrieval (+7.9 EM improvement) | arXiv:2311.09210 |

## Appendix C: Metric Quick Reference Card

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    DATA & MODEL DRIFT — METRIC FRAMEWORK                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  METRIC 1: TRCI (Temporal Response Consistency Index)     Confidence: 4  │
│  ─────────────────────────────────────────────────────────────────────── │
│  Type: Active probing    Primitive: P(Ŷ|X)    Targets: Valid + Fit      │
│  Method: Canary queries → cosine similarity of responses over time       │
│  Scalar: Mean cosine similarity ∈ [0,1]   Alert: TRCI_p10 < 0.90       │
│  Cadence: Daily (high-impact topics), Weekly (comprehensive)             │
│                                                                          │
│  METRIC 2: CDS (Composite Drift Signal)                  Confidence: 3  │
│  ─────────────────────────────────────────────────────────────────────── │
│  Type: Passive monitoring  Primitive: P(X)+P(Ŷ)  Targets: Fit          │
│  Method: K text descriptors → JSD per descriptor → weighted sum          │
│  Scalar: Weighted JSD composite ∈ [0,1]   Alert: CDS > 0.15 (×2 win)  │
│  Cadence: Daily (operational), Weekly (comprehensive)                    │
│                                                                          │
│  METRIC 3: FDS (Faithfulness Decay Score)                Confidence: 3  │
│  ─────────────────────────────────────────────────────────────────────── │
│  Type: Batch evaluation  Primitive: P(Ŷ|X)→E[L]  Targets: Valid        │
│  Method: Atomic claims → LLM verification → JSD of faithfulness dist     │
│  Scalar: Signed JSD ∈ [-1,1]   Alert: FDS > 0.10                       │
│  Cadence: Weekly (sample), Monthly (comprehensive)                       │
│                                                                          │
│  METRIC 4: DDI (Differential Drift Index)                Confidence: 3  │
│  ─────────────────────────────────────────────────────────────────────── │
│  Type: Segment comparison  Primitive: P(Ŷ|X)|segment  Targets: Unbiased│
│  Method: Per-segment quality drift → std/range across segments           │
│  Scalar: Std of cross-segment drift ∈ [0,∞)  Alert: DDI > 0.15         │
│  Cadence: Weekly (routine), Quarterly (comprehensive — Ethics P7)        │
│                                                                          │
│  COMBINED FRAMEWORK                                      Confidence: 4  │
│  Coverage: Model updates (TRCI) + Holistic (CDS) + Factual (FDS)        │
│           + Fairness (DDI)                                               │
│  Compliance: ICO ✓  Equality Act ✓  UK AI White Paper ✓  DPA ✓          │
│              GOV.UK GenAI ✓  Ethics Framework ✓  NCSC ✓                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

*Document version: 1.0 — Metric Design Phase*  
*Next step: Integrate into final assignment deliverable (Connect → Contextualise → Design & Justify → Operationalise)*
