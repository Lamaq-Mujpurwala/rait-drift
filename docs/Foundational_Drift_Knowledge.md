# Foundational Knowledge: Data & Model Drift for NLQ/RAG Systems

> **Context**: UK Government public-sector AI chatbot (RAG-based) answering citizen queries about housing, benefits, local services, governance, and policies. This document builds our foundational understanding before designing metrics.

---

## Question 1: All Possible Data and Model Drifts in Our System

### 1.1 Taxonomy of Drift — General Framework

In classical ML, drift is formally defined in terms of the joint probability distribution $P(X, Y)$ where $X$ is the input and $Y$ is the output. The joint can be decomposed as:

$$P(X, Y) = P(Y|X) \cdot P(X) = P(X|Y) \cdot P(Y)$$

Any change in any component constitutes some form of drift:

| Drift Type | What Changes | Formal Definition |
|---|---|---|
| **Concept Drift** | The input-output relationship | $P_t(Y\|X) \neq P_{t-1}(Y\|X)$ |
| **Data / Feature Drift** | The input distribution | $P_t(X) \neq P_{t-1}(X)$ |
| **Label / Target Drift** | The output distribution | $P_t(Y) \neq P_{t-1}(Y)$ |
| **Virtual Drift** | Input distribution changes but the decision boundary still holds | $P_t(X) \neq P_{t-1}(X)$ but $P_t(Y\|X) = P_{t-1}(Y\|X)$ |
| **Prediction Drift** | The model's output distribution shifts | $P_t(\hat{Y}) \neq P_{t-1}(\hat{Y})$ |

**Key insight from slides**: If label drift and data drift happen simultaneously and cancel each other out, there is no concept drift. The boundary is unchanged even though both sides shifted.

### 1.2 Translating This to Our NLQ/RAG System

Our system architecture is fundamentally different from classical supervised ML. There is no single trained model we control — instead:

```
User Query (X) → [Embedding Model] → [Vector Search / Retrieval] → [Retrieved Docs] → [LLM API] → Response (Ŷ)
```

This means drift can occur at **multiple independent points** in the pipeline. Below is a comprehensive enumeration:

---

#### A. Input Drift (Query-Side) — $P(X)$ changes

These are shifts in **what citizens are asking**.

| Drift | Description | Example |
|---|---|---|
| **Topic Distribution Drift** | The mix of topics citizens ask about changes | Post-election surge in policy questions; seasonal spike in benefits queries near tax deadlines |
| **Vocabulary / Lexical Drift** | New words, phrases, slang, or jargon appear in queries | Citizens start using "UC" for Universal Credit; new policy names like "Renters Reform Bill" |
| **Query Complexity Drift** | Queries become longer, more multi-part, or more specific over time | Users learn the system and ask compound questions: "Can I get housing benefit AND council tax reduction if I'm on PIP?" |
| **Language & Demographic Drift** | Shift in user demographics — language, dialect, literacy level | Users from a newly onboarded region with different dialects; increase in non-native English speakers |
| **Intent Distribution Drift** | Shift in the type of intent — informational vs. transactional vs. complaint | Users shift from "What is council tax?" to "I want to complain about my council tax bill" |
| **Query Volume Drift** | Sudden changes in query volume (not distribution) | A news article about a policy change drives 10x normal traffic |
| **Adversarial / Abuse Drift** | Bad actors start probing, testing, or attacking the system | Prompt injection attempts; users trying to extract training data |
| **Out-of-Vocabulary (OOV) Drift** | Increase in words outside the system's known vocabulary | Typos, non-English text, code/HTML injection |
| **Sentiment Drift** | Shift in the emotional tone of queries | Citizens become more frustrated/angry during a service outage |
| **Query Format Drift** | Users change how they phrase questions (keyword-style vs. natural sentences) | Shift from "housing benefit eligibility" to "am I able to get help with my rent?" |

---

#### B. Knowledge / Document Drift (RAG Context-Side)

These are shifts in the **retrieval corpus** — the documents the RAG system retrieves from.

| Drift | Description | Example |
|---|---|---|
| **Policy Update Drift** | Government policies, regulations, or legislation change | Universal Credit eligibility criteria change; new housing regulations enacted |
| **Document Staleness Drift** | Retrieved documents contain outdated information | The system retrieves a 2023 benefits guide when 2025 rates apply |
| **Corpus Composition Drift** | New documents added or old ones removed from the knowledge base | A new department publishes guidance; legacy pages are decommissioned |
| **Document Quality Drift** | Quality of source documents degrades | Formatting changes, broken links, incomplete policy pages |
| **Retrieval Relevance Drift** | The retrieval model's ability to find relevant documents degrades | Embedding model's representations no longer align well with the evolving query distribution |
| **Coverage Gap Drift** | New policy areas or topics emerge that have no documents in the corpus | A new government scheme is announced but not yet documented in the knowledge base |

---

#### C. Model Behaviour Drift (LLM-Side)

Since we use the LLM via API (no access to model internals), the model itself can drift **without our knowledge**.

| Drift | Description | Example |
|---|---|---|
| **Silent Model Update Drift** | The LLM provider updates the model behind the API | OpenAI/Anthropic pushes a new version; behaviour changes on the same prompts (documented in Chen et al., 2023 — GPT-4 accuracy on prime numbers dropped from 84% to 51% between March and June 2023) |
| **Instruction-Following Drift** | The model's adherence to system prompts/instructions degrades or changes | Model becomes more/less verbose, more/less willing to say "I don't know" |
| **Hallucination Rate Drift** | The model's propensity to fabricate information changes | After an update, the model confabulates more plausible-sounding but incorrect policy details |
| **Tone / Style Drift** | The model's writing style, formality, or register shifts | Model responses become more casual, less appropriate for GOV.UK tone |
| **Safety Filter Drift** | The model's safety/refusal behaviour changes | Model starts refusing legitimate policy questions it previously answered; or becomes less cautious |
| **Latency / Performance Drift** | Response times or token throughput change | API provider degrades performance during high-load periods |

---

#### D. Concept Drift — $P(Y|X)$ changes

In our RAG system, "concept drift" means **the correct answer to a given query changes over time**, even though the query itself hasn't changed. This is the most critical and unique form of drift for our system.

| Drift | Description | Example |
|---|---|---|
| **Policy-Driven Concept Drift** | Government policy changes make previously correct answers wrong | "What is the state pension age?" — answer changes when legislation is enacted |
| **Eligibility Criteria Drift** | Thresholds, rules, or conditions for services change | Income threshold for housing benefit changes from £16,000 to £18,000 |
| **Procedural Drift** | The process for accessing services changes | "How do I apply for Universal Credit?" — shifts from paper to wholly digital |
| **Jurisdictional Drift** | Regional variations emerge or change | A devolved government introduces different rules from England |
| **Temporal/Seasonal Concept Drift** | Some answers are inherently time-dependent | "When is the council tax due?" varies by billing cycle and local authority |

---

#### E. Output / Response Drift — $P(\hat{Y})$ changes

Changes in what the system produces, regardless of input.

| Drift | Description | Example |
|---|---|---|
| **Response Length Drift** | Average response length increases or decreases | System starts generating much longer/shorter answers |
| **Confidence Drift** | Model's expressed confidence or hedging language changes | More/fewer "I'm not sure" qualifiers |
| **Citation/Grounding Drift** | Proportion of responses grounded in retrieved documents changes | Model starts relying more on parametric knowledge vs. retrieved context |
| **Refusal Rate Drift** | Rate at which system declines to answer changes | After LLM update, more questions are rejected |
| **Error Pattern Drift** | Types of errors change — from wrong facts to wrong format to wrong scope | Shifts from factual errors to procedural errors |

---

#### F. Upstream / Data Integrity Drift

Not "real" drift but engineering failures that **appear as drift** (the "blip" and "data integrity" categories from the slides).

| Drift | Description | Example |
|---|---|---|
| **Pipeline Bug Drift** | Preprocessing, tokenization, or formatting breaks | An update to the query preprocessor strips punctuation, changing meaning |
| **Encoding / Schema Drift** | Data format or schema changes upstream | API returns queries in a different character encoding |
| **Integration Drift** | Third-party service changes its interface | The document retrieval API changes response format |
| **Infrastructure Drift** | Hardware/software environment changes | Migration to a new server affects embedding computation precision |

---

### 1.3 Summary: Drift Taxonomy for NLQ/RAG

```
                    ┌─────────────────────────────────┐
                    │     DRIFT IN NLQ/RAG SYSTEM      │
                    └────────────────┬────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
    ┌─────▼─────┐            ┌──────▼──────┐           ┌──────▼──────┐
    │  INPUT (X) │            │ KNOWLEDGE   │           │  MODEL (f)  │
    │  Query     │            │ Documents   │           │  LLM API    │
    │  Drift     │            │ Drift       │           │  Drift      │
    └─────┬─────┘            └──────┬──────┘           └──────┬──────┘
          │                          │                          │
    • Topic dist.             • Policy updates          • Silent updates
    • Vocabulary              • Staleness               • Instruction-follow
    • Complexity              • Corpus changes          • Hallucination rate
    • Demographics            • Coverage gaps           • Tone/style
    • Intent dist.            • Retrieval relevance     • Safety filters
    • Sentiment               • Quality degradation     • Latency
    • Adversarial                                       
    • OOV words                                         
    • Format                                            
          │                          │                          │
          └──────────────────────────┼──────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │        CONCEPT DRIFT             │
                    │   P(Y|X) — correct answer for    │
                    │   a query changes over time      │
                    │   (policy, eligibility, process)  │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │        OUTPUT DRIFT              │
                    │   P(Ŷ) — system response         │
                    │   characteristics shift           │
                    │   (length, confidence, refusals)  │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   DATA INTEGRITY / ENGINEERING   │
                    │   Pipeline bugs, schema changes   │
                    │   (Appears as drift but isn't)    │
                    └─────────────────────────────────┘
```

---

## Question 2: Mapping Slide Concepts to Our NLQ/RAG System

### 2.1 Image 1 — "How We Experience ML Drift" (Temporal Patterns)

The slides show 5 temporal patterns of drift. Here's how each manifests in our system:

| Pattern | Classical Description | NLQ/RAG Parallel | Detection Approach |
|---|---|---|---|
| **Abrupt** | Sudden complete shift in distribution | **A major policy change** (e.g., Brexit transition rules come into effect overnight); **LLM provider pushes a major model update** (GPT-4 → GPT-4o); **Knowledge base wholesale refresh** | Step-change detection in output metrics; sequential analysis with CUSUM or Page-Hinkley test |
| **Gradual** | Slow mixing of old and new distributions | **Evolving citizen language** — users slowly adopt new terminology; **Policy being phased in** (e.g., Universal Credit gradually replacing legacy benefits); **LLM response style slowly shifting across minor API updates** | Sliding window comparison; Wasserstein distance tracked over time; ADWIN (Adaptive Windowing) |
| **Incremental** | Progressive transition through intermediate states | **Seasonal changes in query topics** — gradual shift from tax queries (Jan-Apr) to benefits queries (summer); **Progressive knowledge base aging** as documents become incrementally more outdated | Trend analysis on topic distributions; linear regression on drift scores over time |
| **Recurrent** | Cyclic alternation between distributions | **Annual cycles** — council tax queries peak every April, school admissions every September; **Weekly patterns** — Monday spikes in employment/benefits queries | Seasonal decomposition (STL); autocorrelation analysis of drift metrics; track period-over-period rather than absolute drift |
| **Blip** | Temporary anomaly, not real drift | **A viral social media post** sends unusual traffic for 48 hours; **A temporary API outage** causes error responses; **A news article** about a policy generates a burst of identical queries | Anomaly detection (Z-score, isolation forests); distinguish from sustained drift using window size analysis; require persistence threshold before flagging |

**Critical insight for our system**: In a UK government chatbot, **recurrent drift is expected and should not trigger alerts**. Annual cycles in query topics (tax season, benefits payments, school admissions) are predictable. The system should be calibrated against the same period last year, not just the previous window.

### 2.2 Image 2 — "Key Types of Drift" (Concept, Data, Label, Virtual)

| Slide Concept | Classical Definition | NLQ/RAG Translation |
|---|---|---|
| **Concept Drift** — $P(Y\|X)$ changes | The relationship between input and output changes; the decision boundary shifts | **The correct answer to a citizen's query changes.** E.g., "What benefits am I entitled to?" — the correct answer shifts when eligibility criteria are updated by legislation. The query (X) is identical, but the ground truth (Y) has changed. This is the most dangerous drift for us because the system may continue to confidently give answers that are now wrong. |
| **Label Drift** — $P(Y)$ changes | The distribution of correct outputs shifts | **The distribution of "correct" responses shifts.** E.g., a new government scheme means many more queries should now reference that scheme in the response. If 30% of queries used to need "Universal Credit" in the answer and now 50% do, that's label drift. We can track this via keyword/entity prevalence in responses. |
| **Feature Drift** — $P(X)$ changes | The input data distribution shifts | **The distribution of user queries shifts.** E.g., queries shift from predominantly housing-related to predominantly benefits-related; or the average query length increases; or queries start including more specific postcodes. This is what we can most easily measure. |
| **Virtual Drift** | Data distribution changes but the decision boundary still works | **Query distribution shifts but answers remain valid.** E.g., a new demographic starts using the chatbot with different phrasing but asking about the same topics. The correct answers haven't changed — just the way questions are asked. The system should handle this gracefully. Virtual drift is benign in terms of correctness but may signal upcoming real drift. |

**Key parallel from the slides**: The slide notes that if label drift and feature drift happen simultaneously and cancel each other out, there is no concept drift. For our system: if both the queries and the correct answers shift together (e.g., a new policy generates new types of questions with new types of answers), the *relationship* between query and answer may remain stable. The system may still perform well. This is why monitoring concept drift (not just data drift) is critical.

### 2.3 Image 3 — "Drift Examples for Loan Application Model" → Mapped to Our System

| Loan Model Example | Our Chatbot Parallel |
|---|---|
| **Concept Drift**: An income level that was previously creditworthy is now risky | **Concept Drift**: A benefits eligibility threshold that was previously sufficient now disqualifies the applicant. E.g., the income threshold for Housing Benefit drops from £16,000 to £14,000 — the same query about "I earn £15,000, am I eligible?" now has the opposite correct answer. |
| **Label Drift**: More creditworthy applications showing up | **Label Drift**: More queries about a specific service spike — e.g., universal credit queries increase from 20% to 40% of all queries, shifting the expected distribution of response types. |
| **Feature Drift**: Incomes increase/decrease; applications from new region | **Feature Drift**: Citizens from a newly included geographic area start querying the system; average query length increases; new terminology appears. |

### 2.4 Image 4 — "Triggers of ML Model Drift" (Real Change vs. Data Integrity)

| Trigger Category | Classical Example | NLQ/RAG System Parallel |
|---|---|---|
| **Real Change — Label/Feature distribution** | Product launch in new market shifts customer demographics | **New government policy or service launch** — creates new query types and changes correct answers; **Seasonal patterns** — tax season, benefits payment dates; **Demographic shift** — new populations using the chatbot |
| **Real Change — Concept change** | Competitor launches new service; fundamentally changes what "good" looks like | **Legislation change** — fundamentally alters eligibility rules; **LLM provider model update** — changes what the model considers a "good" response; **Regulatory framework change** — new compliance requirements alter what the system should/shouldn't say |
| **Data Integrity — Faulty engineering** | Correct data enters system but processing swaps values | **RAG pipeline bug** — retrieval returns wrong documents due to index corruption; **Embedding model update** — new embedding model changes similarity scores; **Preprocessing error** — query normalization strips important information |
| **Data Integrity — Incorrect data at source** | Website form leaves field blank | **Knowledge base corruption** — documents uploaded with wrong metadata or formatting; **Stale API cache** — system returns cached responses instead of fresh retrieval; **Source document error** — the GOV.UK page itself contains an error |

**Critical insight**: The slides emphasise that **real change may require a new model** while data integrity issues require **fixing the pipeline**. For our system: real policy change → update the knowledge base and potentially retune retrieval. Data integrity issues → debug and fix the engineering.

### 2.5 Image 5 — "Data Drift Monitoring & Unsupervised Learning" (Statistical Methods)

| Statistical Method | Classical Application | NLQ/RAG Application |
|---|---|---|
| **Population Stability Index (PSI)** | Compare categorical feature distributions between training and production | **Compare query topic distributions** between a reference period and current production. Bin queries by topic/intent and compute PSI. Also applicable to response length distributions, retrieval score distributions. |
| **KL Divergence** $D_{KL}(P \|\| Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$ | Measure how one distribution diverges from another (asymmetric) | **Compare embedding distributions** of queries between time windows. Asymmetry is a problem: use the reference period as P and current as Q. Also useful for comparing response token distributions. |
| **Jensen-Shannon Divergence** $JSD(P \|\| Q) = \frac{1}{2}D_{KL}(P \|\| M) + \frac{1}{2}D_{KL}(Q \|\| M)$ where $M = \frac{1}{2}(P+Q)$ | Symmetric, bounded version of KL. Preferred when neither distribution is "truth". | **Compare any two time periods** without assuming one is "correct". Ideal for our system since we don't have a fixed training distribution — we're comparing production windows. Bounded [0, ln2] for natural log, [0, 1] for log base 2. |
| **Kolmogorov-Smirnov Test** $D = \sup_x |F_1(x) - F_2(x)|$ | Non-parametric test comparing cumulative distributions | **Compare continuous feature distributions** — e.g., embedding dimensions, response lengths, retrieval similarity scores, response times. Gives a p-value for statistical significance. Good for smaller samples. |

**Additional methods relevant to our NLQ/RAG system** (from research):

| Method | Application to Our System |
|---|---|
| **Cosine Distance** on mean embeddings | Compare average query embedding between reference and current windows. Simple, fast. |
| **Domain Classifier (Model-based drift detection)** | Train a binary classifier to distinguish reference vs. current query embeddings. ROC AUC > 0.55 indicates detectable drift. **Recommended as default** by EvidentlyAI research. |
| **Maximum Mean Discrepancy (MMD)** | Kernel-based comparison of embedding distributions. More principled but slower. |
| **Share of Drifted Embedding Components** | Treat each embedding dimension as a feature, test each for drift, report the fraction that drifted. E.g., "23% of embedding dimensions show significant drift." |
| **Wasserstein (Earth-Mover) Distance** | Quantifies the "cost" of transforming one distribution into another. Good for continuous distributions like response lengths or similarity scores. |
| **CUSUM / Page-Hinkley Test** | Sequential change-point detection. Good for detecting abrupt drift in a streaming metric. |
| **ADWIN (Adaptive Windowing)** | Automatically adjusts window size to detect drift. Good for gradual drift. |

---

## Question 3: Ground Truth, Continuous Monitoring & Extracting User Feedback

### 3.1 What "Ground Truth" Means for Our System

In classical ML, ground truth is the known correct label $Y$ for each input $X$. In supervised learning, you compare predictions $\hat{Y}$ against $Y$ to compute metrics like accuracy, precision, recall.

**For our NLQ/RAG chatbot, ground truth is fundamentally problematic.**

There is no automatic way to know if the chatbot's response was "correct." The assignment explicitly states we have access to **"Ground truth (partial)"** — meaning:

#### Types of Ground Truth Available

| Ground Truth Type | Description | Availability | Reliability |
|---|---|---|---|
| **Expert-annotated QA pairs** | Human experts write correct answers for a subset of real queries | Small curated set; expensive to maintain | High, but covers tiny fraction of queries |
| **GOV.UK source documents** | The authoritative source documents that should inform answers | Always available | High, but doesn't directly map to specific query-answer pairs |
| **Historical user feedback** | Thumbs up/down, ratings, escalation data | Accumulates over time | Low-medium: biased toward engaged users; satisfaction ≠ correctness |
| **LLM-as-Judge evaluations** | Using a separate LLM to evaluate response quality | Scalable, automatable | Medium: inherits LLM biases; requires validation against human judgement |
| **Retrieval relevance labels** | Whether the retrieved documents were actually relevant to the query | Can be labeled for a subset | Medium-high for the retrieval stage |

#### The "Delayed/Absent Ground Truth" Problem

For most production queries, we will **never** get a definitive ground truth label. This is why drift monitoring is so critical — it serves as a **proxy signal** when ground truth is unavailable.

As EvidentlyAI notes: *"Getting the ground truth can be a challenge. An ML model predicts something, but you do not immediately know how well it works. In scenarios like text classification, you might need to label the data to evaluate the model quality. Otherwise, you are flying blind."*

For our system, the feedback delay is often **infinite** — we may never know if a citizen got the right answer unless they come back to complain, escalate, or ask a follow-up.

### 3.2 Continuous Monitoring Architecture

A production monitoring system for our chatbot should operate on **multiple time horizons**:

#### Real-Time Monitoring (Per Query)

| What to Monitor | How | Alert Condition |
|---|---|---|
| Response latency | Track API response time | > 2x baseline |
| Empty/error responses | Check for null, error, or refusal responses | Rate > threshold |
| Response length anomalies | Track character/word count | Outside 2σ of baseline |
| Retrieval score | Track similarity score of top-k retrieved docs | Mean score drops below threshold |
| Prompt injection detection | Pattern matching / classifier on inputs | Any detection triggers review |

#### Batch Monitoring (Hourly/Daily Windows)

| What to Monitor | Method | Reference |
|---|---|---|
| Query topic distribution | Cluster queries, compare cluster proportions via PSI/JSD | Same day last week, or rolling 30-day baseline |
| Query embedding drift | Domain classifier or cosine distance on mean embeddings | Reference window (e.g., last month's validated period) |
| Response embedding drift | Same methods applied to response embeddings | Same reference |
| Text descriptor drift | Track sentiment, length, OOV %, trigger words, formality | Rolling baseline |
| Retrieval relevance distribution | Distribution of top-k similarity scores | Training/validation period |
| User feedback distribution | Distribution of thumbs up/down | Previous period |
| Topic coverage | % of queries matched to known topic clusters vs. "unknown" | Historical baseline |

#### Periodic Evaluation (Weekly/Monthly)

| What to Evaluate | Method | Purpose |
|---|---|---|
| Expert evaluation on golden set | Run curated QA pairs through system, compare to known correct answers | Direct quality measurement |
| LLM-as-Judge evaluation | Use a second LLM to evaluate a sample of production responses for relevance, accuracy, completeness, tone | Scalable quality proxy |
| Knowledge base freshness audit | Check when source documents were last updated | Ensure currency |
| Retrieval drift analysis | Full embedding drift report on query-document pairs | Deep diagnostic |
| Comparative analysis | Compare current period vs. same period last year (for seasonal patterns) | Recurrent drift calibration |

### 3.3 Extracting User Information to Materialise Drift

**The central challenge**: We must extract as much signal as possible from user interactions without being invasive or violating data protection (ICO/GDPR compliance is mandatory for UK government systems).

#### Explicit Feedback Mechanisms

| Mechanism | What It Captures | Implementation |
|---|---|---|
| **Thumbs Up/Down** | Binary satisfaction signal | Button after each response. GOV.UK Generative AI Framework recommends this explicitly: *"Obtain user feedback to understand the usefulness of the returned response. This could be a simple thumbs-up indicator."* |
| **"Was this helpful?" with reason** | Categorised dissatisfaction | Drop-down: "Not relevant", "Outdated information", "Too vague", "Incorrect", "Didn't understand my question" |
| **Escalation to human** | System failure signal | Track rate and categorise reasons for escalation |
| **Contact form / complaint** | Detailed failure description | Mine text for drift indicators |
| **Rating scale (1-5)** | Granular satisfaction | Useful for tracking distribution shifts over time |

#### Implicit Feedback Signals (Behavioural)

These don't require active user participation:

| Signal | What It Indicates | How to Compute |
|---|---|---|
| **Follow-up queries** | User didn't get what they needed | Track % of sessions with >1 query; high follow-up rate → potential answer quality issue |
| **Query reformulation** | User is trying different phrasing | Detect semantic similarity between consecutive queries in a session (embedding cosine sim > threshold + different surface form) |
| **Session abandonment** | User gave up | Track sessions with no explicit positive signal and no follow-up |
| **Time-to-next-query** | Thinking time / confusion | Very short → quick follow-up (dissatisfied). Very long → reading (possibly satisfied). |
| **Query repetition** | System gave unsatisfactory answer | Detect near-identical queries repeated within a session |
| **Bounce rate** | User left immediately after response | One query, no feedback, no further interaction |
| **Copy/paste behaviour** | User found response useful | Track clipboard events (if available) — indicates actionable response |
| **Click-through on links** | User following up on referenced resources | Track if user clicks GOV.UK links provided in responses |

#### Materialising Drift from These Signals

The key insight is that **individual feedback signals are noisy, but distributional shifts in these signals over time materialise drift**.

**Example workflow**:
1. **Compute daily aggregates**: mean satisfaction rating, follow-up rate, escalation rate, abandonment rate, response length distribution
2. **Compare against baseline**: Use JSD or PSI to compare each day's aggregate distribution against the reference period
3. **Track per-topic**: Break down by query topic cluster — a topic-specific drift is more actionable than a global one
4. **Correlate signals**: If follow-up rate increases AND satisfaction decreases AND for a specific topic cluster → strong evidence of concept drift in that topic area
5. **Alert thresholds**: Set multi-signal alert thresholds — e.g., alert when ≥2 of 4 proxy signals drift simultaneously for the same topic

**Formalising the feedback loop**:

$$\text{Drift Signal}(t) = \sum_{i} w_i \cdot \mathbb{1}[\text{metric}_i(t) \text{ drifted from baseline}]$$

Where $w_i$ are weights reflecting the reliability of each signal (explicit feedback > implicit signals > volume metrics), and $\mathbb{1}$ is an indicator function based on the chosen statistical test.

---

## Question 4: Research on Drift in NLQ/Language Modelling

### 4.1 Key Research Papers & Findings

#### A. LLM Behaviour Drift — Empirical Evidence

**Chen, Zaharia & Zou (2023). "How is ChatGPT's behavior changing over time?"** arXiv:2307.09009

This is the **foundational empirical paper** on LLM drift. Key findings:
- Evaluated GPT-3.5 and GPT-4 across March 2023 and June 2023 on 7 diverse tasks
- **GPT-4 accuracy on identifying prime numbers dropped from 84% → 51%** between versions
- GPT-4 became less willing to answer sensitive/opinion questions
- Both models showed more formatting mistakes in code generation over time
- **GPT-4's ability to follow user instructions decreased over time** — identified as a common factor behind multiple observed drifts
- Conclusion: *"the behavior of the 'same' LLM service can change substantially in a relatively short amount of time, highlighting the need for continuous monitoring"*

**Relevance to our system**: Since we use an LLM via API, we are subject to silent model updates. This paper provides empirical evidence that LLM drift is real, measurable, and can be severe. We must include LLM behaviour monitoring in our drift detection system.

---

#### B. Concept Drift — Theoretical Framework

**Lu, Liu, Dong, Gu, Gama & Zhang (2020). "Learning under Concept Drift: A Review."** arXiv:2004.05785 (IEEE TKDE 2018)

Comprehensive review of 130+ publications establishing the concept drift framework:
- **Taxonomy**: sudden, gradual, incremental, recurrent drift (maps directly to our slides)
- **Three-component framework**: Detection → Understanding → Adaptation
- **Detection methods**: Error-rate based (DDM, EDDM, HDDM), distribution-based (ADWIN, Kolmogorov-Smirnov), multiple hypothesis testing
- **Adaptation strategies**: Instance selection, instance weighting, ensemble methods

**Key mathematical formalisms**:
- Drift at time $t$: $\exists t: P_t(X, Y) \neq P_{t-1}(X, Y)$
- DDM (Drift Detection Method): monitors error rate $p_t$ and standard deviation $s_t = \sqrt{p_t(1-p_t)/n}$; flags warning when $p_t + s_t > p_{min} + 2s_{min}$ and drift when $p_t + s_t > p_{min} + 3s_{min}$
- ADWIN: maintains a variable-length window, splits it into two subwindows, tests if their means differ significantly

**Relevance**: Provides the mathematical foundations for our drift detection metrics. The DDM/ADWIN methods can be adapted for streaming monitoring of our proxy metrics (satisfaction rate, follow-up rate, etc.).

---

#### C. Embedding Drift Detection for NLP

**EvidentlyAI (2023). "Shift happens: we compared 5 methods to detect drift in ML embeddings."**

Empirical comparison of 5 embedding drift detection methods on text datasets (Wikipedia comments, news categories, food reviews) using BERT and FastText embeddings:

| Method | Drift Score | Interpretability | Speed | PCA Stability | Embedding Stability |
|---|---|---|---|---|---|
| **Euclidean Distance** | 0 to ∞ | Low (absolute scale) | Fast | Consistent | Inconsistent |
| **Cosine Distance** | 0 to 2 | Low | Fast | **Fails with PCA** | Inconsistent |
| **Domain Classifier** (ROC AUC) | 0 to 1 | **High** | Medium | Consistent | **Consistent** |
| **Share of Drifted Components** | 0 to 1 | Fairly high | Medium | Less sensitive | **Consistent** |
| **MMD** | ≥ 0 | Low | **Very slow** | Slightly more sensitive | Consistent |

**Recommendation**: **Domain Classifier (model-based drift detection)** is the best default method — interpretable threshold (ROC AUC), consistent across different embeddings and with/without PCA, reasonable computation speed.

**Relevance**: Directly applicable to monitoring query embedding drift and response embedding drift in our system. We should use domain classifier as our primary method and supplement with Euclidean distance for trend tracking.

---

#### D. Text Descriptor Monitoring for NLP/LLM

**EvidentlyAI (2023/2025). "Monitoring unstructured data for LLM and NLP with text descriptors."**

Proposes monitoring **interpretable text properties** instead of (or alongside) raw embeddings:

**Recommended default descriptors**:
1. **Text length** (words, characters, sentences)
2. **Out-of-vocabulary (OOV) word share** — "crude measure of data quality"
3. **Non-letter character share** — detects code injection, HTML leaks, formatting issues
4. **Sentiment** — tracks emotional tone shifts
5. **Trigger words** — custom word lists for domain-specific monitoring
6. **RegExp matches** — pattern-based checks
7. **Custom descriptors** — using external models (emotion classification, toxicity, topic)

**Relevance**: This is highly practical for our system. We can monitor these descriptors for both queries AND responses:
- Query-side: OOV drift detects language shifts; sentiment drift detects frustrated users; trigger words detect topic shifts
- Response-side: Length drift detects verbosity changes; sentiment drift detects tone changes; trigger words ensure policy terms appear appropriately

---

#### E. RAG Systems — Survey and Challenges

**Gao, Xiong et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey."** arXiv:2312.10997

Comprehensive survey of RAG paradigms (Naive RAG → Advanced RAG → Modular RAG):
- Identifies hallucination and outdated knowledge as core challenges RAG addresses
- RAG enables "continuous knowledge updates and integration of domain-specific information"
- Covers retrieval, generation, and augmentation techniques
- Introduces evaluation frameworks for RAG systems

**Relevance**: Establishes that our RAG system design is the current best practice for knowledge-grounded QA. The challenge of **knowledge freshness** (document staleness drift) is a recognised open problem. RAG evaluation metrics (faithfulness, answer relevance, context relevance) can serve as drift detection signals.

---

#### F. Hallucination as a Drift-Related Phenomenon

**Zhang, Li et al. (2023). "Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models."** arXiv:2309.01219

Surveys hallucination detection, explanation, and mitigation:
- Taxonomy: input-conflicting, context-conflicting, fact-conflicting hallucinations
- Fact-conflicting hallucinations = the model generates information that contradicts world knowledge
- Hallucination rate is not static — it can change with model updates, prompt changes, and context changes

**Relevance**: Hallucination rate drift is a key metric for our system. If the LLM starts hallucinating more (or differently), that's model behaviour drift. We can monitor for this via:
- Comparing response claims against retrieved documents (faithfulness checking)
- Tracking the distribution of "groundedness" scores over time
- Detecting shifts in the rate of unsupported claims

---

#### G. RAG Robustness to Noisy Retrieval

**Yu, Zhang et al. (2023). "Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models."** arXiv:2311.09210 (EMNLP 2024)

Addresses a critical RAG failure mode: what happens when retrieved documents are irrelevant or noisy?
- Standard RALMs can be misguided by irrelevant retrieved data
- Can cause model to overlook its own knowledge
- Chain-of-Noting (CoN) generates sequential reading notes for documents, evaluating relevance
- +7.9 EM improvement with entirely noisy documents; +10.5 rejection rate improvement for out-of-scope questions

**Relevance**: When our knowledge base drifts (staleness, coverage gaps), the retrieval may return irrelevant documents. This paper shows the impact is measurable and significant. Monitoring retrieval relevance scores (and their distribution over time) is essential for detecting knowledge drift.

---

#### H. Criteria Drift in LLM Evaluation

**Shankar, Zamfirescu-Pereira et al. (2024). "Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences."** arXiv:2404.12272

Introduces the concept of **"criteria drift"** in LLM evaluation:
- Users need criteria to grade outputs, but grading outputs helps users define criteria
- Some evaluation criteria are **dependent on the specific LLM outputs observed** rather than being definable a priori
- This raises questions about the independence of evaluation from observation

**Relevance**: When we use LLM-as-Judge for monitoring, the evaluation criteria themselves may drift. This is a meta-drift problem: our drift detection system may itself be subject to drift. We must establish fixed, well-defined evaluation rubrics and periodically validate them against human judgement.

---

#### I. Factuality in LLMs

**Min et al. (2023). "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation."** (related: Survey on Factuality in Large Language Models, arXiv:2310.07521)

Key framework: decompose long-form text into **atomic facts** and verify each against a knowledge source.
- Factual precision = proportion of atomic facts that are supported
- Can be automated using retrieval against a reference corpus

**Relevance**: Directly applicable to measuring "faithfulness drift" in our system. We can:
1. Decompose chatbot responses into atomic claims
2. Verify each against the RAG knowledge base
3. Track the "faithfulness score" distribution over time
4. Alert when the distribution shifts (more unsupported claims = hallucination drift)

### 4.2 Summary of Mathematical Methods from Research

| Method | Formula / Description | Best For |
|---|---|---|
| **PSI** | $PSI = \sum_{i=1}^{k}(p_i - q_i) \cdot \ln\frac{p_i}{q_i}$ | Categorical/binned distributions; topic distributions |
| **KL Divergence** | $D_{KL}(P \|\| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$ | Comparing against a fixed reference |
| **Jensen-Shannon** | $JSD(P \|\| Q) = \frac{1}{2}D_{KL}(P \|\| M) + \frac{1}{2}D_{KL}(Q \|\| M)$, $M = \frac{P+Q}{2}$ | Comparing any two periods (symmetric, bounded) |
| **KS Test** | $D = \sup_x \|F_1(x) - F_2(x)\|$; p-value from $\sqrt{n}D$ | Continuous features; small samples |
| **Wasserstein** | $W(P,Q) = \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$ | Continuous distributions; response properties |
| **Domain Classifier** | Train binary classifier (ref vs. current); ROC AUC as drift score | Embedding drift (recommended default) |
| **Cosine Distance** | $d = 1 - \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}$ on mean embeddings | Quick embedding comparison |
| **Euclidean Distance** | $d = \|\mu_{ref} - \mu_{cur}\|_2$ on mean embeddings | Trend tracking over time |
| **MMD** | $MMD^2 = \mathbb{E}[k(x,x')] + \mathbb{E}[k(y,y')] - 2\mathbb{E}[k(x,y)]$ | Principled distribution comparison |
| **CUSUM** | $S_t = \max(0, S_{t-1} + (x_t - \mu_0) - \delta)$; alarm if $S_t > h$ | Sequential change-point detection |
| **ADWIN** | Adaptive windowing; tests $\|\hat{\mu}_{W_1} - \hat{\mu}_{W_2}\| \geq \epsilon_{cut}$ | Gradual drift in streaming metrics |
| **DDM** | Monitors $p_t + s_t$ where $s_t = \sqrt{p_t(1-p_t)/n}$; warning at $+2s_{min}$, drift at $+3s_{min}$ | Error-rate based sequential detection |
| **Chi-Square Test** | $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ | Categorical distributions |
| **Fisher's Exact Test** | Exact p-value for 2×2 contingency tables | Small sample categorical comparison |

### 4.3 Research Gaps and Opportunities

Based on the literature review, several important gaps are relevant to our assignment:

1. **No established framework for drift monitoring in RAG systems specifically**. Most drift research focuses on classical ML or standalone LLMs. Our assignment has an opportunity to propose a novel framework.

2. **LLM-as-Judge reliability for drift detection is underexplored**. The "criteria drift" paper (Shankar et al.) raises concerns but doesn't resolve them.

3. **Multi-component drift attribution in pipelines** is an open problem. When the RAG system produces a wrong answer, was it a query understanding failure, a retrieval failure, an LLM generation failure, or a knowledge base issue? Attributing drift to the right component is critical for remediation.

4. **Partial/proxy ground truth methods for government chatbots** need more work. The intersection of delayed feedback, high stakes, and regulatory compliance creates unique constraints.

5. **Temporal knowledge management in RAG systems** — how to detect when retrieved knowledge is outdated and should no longer be served — is practiced but not well formalised.

---

## Key Takeaways for Our Metric Design

1. **We should design metrics at multiple pipeline stages**: input (queries), retrieval, generation (responses), and feedback (user signals).

2. **Embedding drift detection (domain classifier method)** should be our primary unsupervised drift detection approach for both queries and responses.

3. **Text descriptors** (length, sentiment, OOV, trigger words, topic distribution) provide interpretable, fast monitoring signals.

4. **Statistical methods should be chosen based on our data characteristics**: JSD for comparing production windows (symmetric), PSI for categorical topic distributions, KS test for continuous features, domain classifier for embedding spaces.

5. **Ground truth is partial** — we must rely heavily on proxy metrics and multi-signal correlation to materialise drift.

6. **LLM behaviour drift (Chen et al.)** is empirically proven and must be explicitly monitored through periodic evaluation against a golden test set.

7. **Concept drift is our highest-risk drift type** — when correct answers change due to policy updates, our system may confidently give wrong answers with no automatic signal. This requires a combination of knowledge base freshness monitoring, periodic expert evaluation, and response-source consistency checking.

---

*Sources consulted: EvidentlyAI (embedding drift, data drift, text descriptors, concept drift), Neptune.ai (concept drift), GOV.UK Generative AI Framework (Principle 5), Chen et al. 2023 (LLM drift), Lu et al. 2020 (concept drift review), Gao et al. 2024 (RAG survey), Zhang et al. 2023 (hallucination survey), Yu et al. 2023 (Chain-of-Note), Shankar et al. 2024 (criteria drift), Min et al. 2023 (FActScore).*
