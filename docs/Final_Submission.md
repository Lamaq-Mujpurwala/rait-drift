# Data and Model Drift: Detecting and Responding to Behavioural Change in a Public Sector AI System

**Candidate:** Lamaq  
**Ethical Dimension:** Data & Model Drift  
**Compliance Question:** How does the organisation detect and respond to changes in data or model behaviour over time to ensure the system remains valid, unbiased, and fit for purpose?  
**Date:** 25 February 2026  

---

## Introduction

This document presents a monitoring framework for detecting and responding to behavioural drift in a Retrieval-Augmented Generation (RAG) chatbot that answers public queries about UK government services. The work was carried out over approximately three days: the first spent understanding the problem space and studying how data and model drift manifest in production AI systems, the second building the RAG pipeline and the four monitoring metrics described in this document, and the third writing and validating a 192-test suite and preparing this submission.

The assignment rules shaped the work in important ways. I was told to assume access to user queries, LLM responses, a model client via API (with no access to model internals), limited ground truth, and system performance metrics. I was told not to assume any additional data beyond this. This constraint is central to the design: every metric in this document operates exclusively on the data specified, and no metric requires access to model weights, training data, or RAG-specific internal logs. I was also told not to rely on standard classification metrics such as accuracy, precision, or recall, and to design my own metrics rather than simply naming existing ones. The four metrics presented here are purpose-built for this scenario.

The system I built uses the GOV.UK Content API as the knowledge base (33 government pages covering Universal Credit, Housing Benefit, PIP, Council Tax, Homelessness, and Pensions), Pinecone as the vector store with Meta's Llama Text Embed v2 for embeddings, and Groq-hosted Llama 3.3 70B as the generation model. All inference runs through API calls; I host no models myself. This is deliberate, because it is precisely the kind of deployment where drift is most dangerous: the model provider can update the model at any time, and the deploying organisation has no visibility into when or why that happens.

Throughout this document, I make specific claims and provide specific reasoning. Where I am uncertain, I say so. Where my approach has weaknesses, I identify them. The goal is not to present a flawless system but to demonstrate that I understand the problem, have thought carefully about how to measure it, and can reason honestly about where my measurements fall short.

---

## 1. Connect

The compliance question asks how an organisation detects and responds to changes in data or model behaviour over time to ensure the system remains valid, unbiased, and fit for purpose. The ethical dimension assigned to this question is Data and Model Drift. To connect these, I need to explain what drift actually is in the context of a deployed language model, and why the ethical dimension is not merely a technical concern but a governance one.

Data drift, in the traditional machine learning sense, refers to a change in the statistical distribution of inputs to a system. In a chatbot that handles government service queries, this could mean that the kinds of questions people ask change over time. Perhaps a policy announcement causes a sudden increase in questions about Universal Credit sanctions, or a winter energy crisis shifts the balance of queries towards heating-related benefits. The system was designed and tested against one distribution of queries, and it is now facing a different one. This matters because the system's retrieval index, its prompt engineering, and the assumptions embedded in its design may not hold under the new distribution.

Model drift is different and, for API-hosted models, more insidious. It refers to a change in the model's behaviour on the same inputs. This can happen because the API provider updates the model (a new fine-tune, a quantisation change, a version bump), or because the embedding model shifts how it represents text, or because the retrieval index becomes stale as government policies change but the knowledge base does not. The critical property of model drift in API-hosted systems is that it is invisible to the deploying organisation. The model endpoint URL does not change, the API contract does not change, and there is no release note. The system simply starts behaving differently.

The ethical dimension of drift is not just about accuracy. It is about accountability. When a human civil servant gives wrong advice, there is a paper trail: the person can be identified, the advice can be corrected, and the individual can be retrained. When an AI system silently changes its behaviour, there may be no equivalent trail. Users who received incorrect information last week have no way of knowing that the system was behaving differently then than it does now. The organisation deploying the system may not even know that the change occurred. This is the core governance failure that the compliance question targets: not that drift happens, but that it goes undetected.

Translating this into something measurable requires deciding what aspects of behaviour to monitor. A single aggregate measure would be insufficient because drift can manifest in many ways: the model might become less accurate overall, or it might become less accurate for one topic while improving on another, or it might maintain accuracy while changing tone and style in ways that affect user trust. My approach is to decompose the problem into four distinct failure modes, each of which maps to a measurable signal, and each of which addresses a different part of the compliance question. The first two (TRCI and CDS) address the "detect changes" part. The third (FDS) addresses the "remains valid" part. The fourth (DDI) addresses the "unbiased and fit for purpose" part.

---

## 2. Contextualise

The scenario is a UK public sector organisation using an AI system to handle queries from members of the public about government services, policies, and support. To contextualise the drift problem, consider what actually happens when this system is deployed and used.

A person visits the chatbot and asks: "I have £14,000 in savings. Am I eligible for Universal Credit?" The system retrieves the relevant GOV.UK page, which states that the savings threshold is £16,000, and generates a response: "Yes, you may be eligible. Universal Credit has a savings limit of £16,000, and since your savings are below this threshold, they should not prevent you from claiming." This is correct, helpful, and grounded in the source document. This is what good looks like: accurate information, clearly stated, sourced from official government content.

Now consider what bad looks like. Suppose the API provider silently updates the language model three months into deployment. The new model is generally more capable, but it handles financial thresholds differently due to changes in its training data. The same question now produces: "You may be eligible for Universal Credit. Generally, you need savings below £15,000 to qualify." The number is wrong. The response sounds confident. The user has no reason to doubt it. The organisation has no mechanism to detect that the answer changed. Someone may make a financial decision based on incorrect information, and the error could go unnoticed for weeks or months until a complaint surfaces through unrelated channels.

This is not a hypothetical risk. API providers routinely update models, and the deploying organisation has no control over the timing or nature of those updates. On free-tier access (which is what I used for development), there is not even a mechanism to pin a specific model version. The model I am building monitoring for is, in a real sense, a moving target.

The consequences of undetected drift in this specific scenario are more severe than in many other AI applications. Government benefits are a critical safety net. People asking about Universal Credit, PIP, or Housing Benefit are often in financially precarious situations. An incorrect answer about eligibility criteria, payment amounts, or application deadlines can lead to missed claims, financial hardship, or incorrect expectations about entitlements. The information asymmetry is significant: the user is coming to the chatbot precisely because they do not know the answer themselves, and they are trusting the system to give them correct, current, and complete information.

There is also a fairness dimension that is easy to overlook. If drift affects all topics equally, the harm is distributed uniformly across users. But if drift affects some topics more than others, the harm falls disproportionately on the groups of people who ask about those topics. Questions about disability benefits are disproportionately asked by disabled people. Questions about pension credit are disproportionately asked by elderly people. If the system degrades more severely on disability-related queries than on council tax queries, then disabled users are receiving a worse service than non-disabled users, not because anyone decided to treat them differently, but because an invisible model change happened to affect that topic segment more. Under Section 149 of the Equality Act 2010, public bodies have a Public Sector Equality Duty (PSED) to have due regard to the need to eliminate discrimination and advance equality of opportunity. A monitoring system that only checks overall quality, without checking whether quality is distributed fairly across user groups, cannot satisfy this duty.

What good looks like in a well-functioning system, then, is not just "the chatbot gives correct answers." It is a system where the organisation knows, on an ongoing basis, that the answers are correct, that they remain correct over time, that the correctness has not degraded for any particular group more than others, and that when any of these properties change, there is a clear signal and a clear process for responding. This is the operational standard that the four metrics in the next section are designed to achieve.

---

## 3. Design and Justify

The four metrics described here were designed to cover four distinct failure modes: silent model replacement (TRCI), distributional response shift (CDS), declining answer accuracy (FDS), and unequal drift across user groups (DDI). Each metric produces a single scalar value that can be tracked over time, classified into a traffic-light status (GREEN, AMBER, RED), and used to trigger operational responses. I chose to use four metrics rather than a single composite score because a single score obscures which dimension is failing, and an operator who sees "overall health: 72%" cannot act on that without knowing whether the problem is accuracy, fairness, or system stability.

All four metrics are built on Jensen-Shannon Divergence (JSD) as the core statistical measure for comparing distributions. JSD was chosen over alternatives (KL divergence, Kolmogorov-Smirnov test, Wasserstein distance) for specific reasons: it is symmetric, meaning JSD(P, Q) = JSD(Q, P); it is bounded between 0 and ln(2), making threshold-setting meaningful; and it is always defined even when the two distributions have different supports, unlike KL divergence which goes to infinity when one distribution assigns zero probability to an event the other considers possible (Lin, 1991). Formally, for two probability distributions $P$ and $Q$:

$$\text{JSD}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M), \quad M = \frac{P + Q}{2}$$

where $D_{\text{KL}}$ is the Kullback-Leibler divergence. The output lies in $[0, \ln 2] \approx [0, 0.693]$. In my implementation, continuous data is discretised via histograms, discrete data uses count-based distributions over the union of observed values, and binary data (e.g., refusal flag) uses two-element Bernoulli distributions. A small $\varepsilon = 10^{-10}$ is added to all bin counts before normalisation to prevent $\log(0)$. These properties make JSD particularly suitable for production monitoring where robustness and interpretability matter more than statistical power.

One significant technical challenge I encountered and resolved during development concerns JSD estimation on small samples. JSD is computed via histograms for continuous data, and the number of histogram bins directly affects the estimate. The theoretical bias of histogram-based JSD is approximately $(k - 1) / 2N$, where $k$ is the number of bins and $N$ is the sample size. For N=30 and a fixed 50 bins, this gives a bias of approximately 0.82, which is larger than any of my RED thresholds and would produce false positives on every monitoring run. I addressed this by implementing adaptive binning using the Freedman-Diaconis rule (Freedman and Diaconis, 1981), which computes the optimal bin width $h$ as:

$$h = 2 \cdot \text{IQR} \cdot n^{-1/3}, \quad \text{bins} = \left\lceil \frac{\max - \min}{h} \right\rceil$$

clamped to the range $[10, 50]$. When the IQR is zero (degenerate distribution), the implementation falls back to Sturges' rule: $\text{bins} = \lceil \log_2(n) + 1 \rceil$. For N=30, the Freedman-Diaconis rule typically selects 12 to 15 bins, reducing the bias from approximately 0.82 to approximately 0.22 — a 3.7-fold reduction. I validated this improvement with a Monte Carlo simulation in my test suite: drawing 200 pairs from identical distributions with N=30, fixed binning produced a median JSD of 0.82 while adaptive binning produced 0.22.


### 3.1 Temporal Response Consistency Index (TRCI)

TRCI is designed to answer a specific question: has the underlying model or system changed since I last checked? It does this through active canary probing, a technique borrowed from infrastructure monitoring, where known inputs are periodically submitted to a system and the outputs are compared against stored references. In my implementation, 50 canary queries are pre-written to cover all six topic areas (Universal Credit, Housing Benefit, Council Tax, Disability Benefits, Homelessness, and Pensions). Before monitoring begins, each canary is submitted to the production pipeline and the response is saved as a reference. On each subsequent monitoring cycle, the canaries are re-submitted and the new responses are compared to their references using cosine similarity in the embedding space.

The computation proceeds as follows. Each canary query is run through the full production pipeline (retrieval from Pinecone, generation via Groq). The new response is embedded using Pinecone's integrated Llama Text Embed v2 model and compared against the stored reference embedding via cosine similarity. This produces one similarity score per canary. The TRCI scalar is defined as:

$$\text{TRCI} = \frac{1}{N} \sum_{i=1}^{N} \text{cosine\_sim}\left(\mathbf{e}_{i}^{\text{ref}},\; \mathbf{e}_{i}^{\text{new}}\right)$$

where $\mathbf{e}_{i}^{\text{ref}}$ is the stored reference embedding and $\mathbf{e}_{i}^{\text{new}}$ is the fresh response embedding for canary $i$, and $N = 50$. I also track the 10th percentile ($p_{10}$) of the per-canary similarity distribution as a tail-risk indicator: even if most canaries are stable, a small number of severely drifted canaries warrant attention.

The data required is: a set of canary queries with stored reference responses (provided), the production pipeline (user queries, LLM responses via API), and the embedding model (accessed via Pinecone). No data beyond what the assignment specifies is needed; the canaries are simply queries submitted through the same interface any user would use.

Classification uses a two-stage conjunction rule:

| Status | Condition |
|---|---|
| GREEN | $\text{TRCI} \ge 0.95$ **and** $p_{10} \ge 0.80$ |
| AMBER | $0.90 \le \text{TRCI} < 0.95$, or $p_{10} < 0.80$ with $\text{TRCI} \ge 0.90$ |
| RED | $\text{TRCI} < 0.90$ **or** $p_{10} < 0.80$ |

The GREEN threshold of 0.95 was set conservatively: natural variation in language model output (different word choices, sentence structures) means that perfectly equivalent responses will typically score between 0.95 and 0.99 in cosine similarity, not 1.0. A drop below 0.95 indicates that the semantic content of the response has changed, not just the phrasing.

The justification for TRCI is straightforward. In a system where the model is accessed via API and the provider can update the model at any time without notification, canary probing is one of the few mechanisms available to detect such changes. I cannot inspect model weights, request version identifiers (on the free tier), or subscribe to update notifications. What I can do is observe the model's behaviour on known inputs. If the behaviour changes, something in the system changed. TRCI makes this observable. It directly addresses the "detect changes in model behaviour" part of the compliance question.

The limitations of TRCI are worth stating clearly. First, it only detects changes that affect the canary queries. If the model changes its behaviour on a topic that no canary covers, TRCI will not catch it. I mitigate this by distributing canaries across all six topics, but the coverage is bounded by the canary set size. Second, TRCI measures similarity in the embedding space, which is itself subject to drift: if the embedding model changes, TRCI similarity scores would drop even if the language model's behaviour is unchanged. This is technically a false positive, but operationally it is still a signal worth investigating, since the retrieval system depends on those same embeddings. Third, a model update that produces better answers would also trigger TRCI, because the responses are different from the stored references. TRCI detects change, not degradation. Distinguishing the two requires FDS.

**Confidence: 4/5.** The mechanism is sound, well-tested, and directly addresses the compliance question. The main uncertainty is whether 50 canaries provide sufficient coverage of the query space, and whether cosine similarity in the embedding space is sensitive enough to catch all meaningful changes. I have not yet observed a real model update to validate TRCI's detection capability in practice.


### 3.2 Composite Drift Signal (CDS)

CDS monitors whether the statistical properties of the system's responses are shifting over time. Where TRCI probes the model directly, CDS observes the model passively through the responses it produces to real user queries. It does this by tracking nine measurable descriptors of each response and comparing their distributions between a reference window (the last 30 days of production data) and a current window (the last 7 days).

The nine descriptors are: response length (word count), response sentiment (polarity score), response readability (Flesch-Kincaid grade level), citation count (number of GOV.UK references), hedge word count (words like "might", "possibly", "generally"), refusal flag (whether the system refused to answer), response latency (milliseconds), mean retrieval distance (how far the retrieved documents are from the query in embedding space), and context token ratio (what fraction of the response is derived from retrieved context versus generated freely). Each descriptor captures a different facet of system behaviour. For example, an increase in hedge word count might indicate the model is becoming less certain, while a drop in citation count might indicate degraded retrieval.

For each descriptor, I compute the JSD between the reference window distribution and the current window distribution. The type of JSD depends on the descriptor's nature: continuous descriptors (response length, sentiment, readability, latency, retrieval distance, context ratio) use histogram-based JSD with adaptive Freedman-Diaconis binning; discrete descriptors (citation count, hedge word count) use count-based JSD over the union of observed values; and the binary descriptor (refusal flag) uses the JSD between two Bernoulli distributions derived from the proportion of refusals in each window.

The CDS scalar is the weighted average of the nine per-descriptor JSD values:

$$\text{CDS} = \sum_{k=1}^{9} w_k \cdot \text{JSD}_k$$

where $\sum w_k = 1.0$ and the weights are:

| Descriptor | Weight | JSD Type |
|---|---|---|
| Response length (word count) | 0.15 | continuous |
| Citation count (GOV.UK references) | 0.15 | discrete |
| Refusal flag (refused to answer) | 0.15 | binary |
| Response sentiment (polarity) | 0.10 | continuous |
| Response readability (Flesch-Kincaid) | 0.10 | continuous |
| Hedge word count | 0.10 | discrete |
| Response latency (ms) | 0.10 | continuous |
| Mean retrieval distance | 0.10 | continuous |
| Context token ratio | 0.05 | continuous |

The weights reflect domain-informed importance: citation count (0.15), refusal flag (0.15), and response length (0.15) are weighted highest because they most directly affect information quality; context token ratio (0.05) is weighted lowest because it is an indirect signal. The specific weight values were chosen through reasoning about the scenario rather than empirical optimisation, which is a limitation I acknowledge.

A deliberate design choice in CDS is the persistence filter. The metric does not escalate to RED the first time the score exceeds the threshold. Instead, the score must remain above the AMBER threshold for $P = 3$ consecutive monitoring windows before RED is triggered. Formally, the classification is:

| Status | Condition |
|---|---|
| GREEN | $\text{CDS} < 0.05$ |
| AMBER | $0.05 \le \text{CDS} < 0.15$, or $\text{CDS} \ge 0.15$ but persistence not met |
| RED | $\text{CDS} \ge 0.15$ **and** $\text{CDS}_{t-k} \ge 0.15$ for all $k \in \{0, 1, 2\}$ |

The reasoning is that production monitoring will encounter transient spikes caused by random variation, unusual query patterns (public holidays, news events), or temporary infrastructure issues. If the system alarmed on every spike, operators would quickly learn to ignore the alerts, which is worse than having no alerts. The persistence filter distinguishes sustained drift (a genuine concern) from transient noise (an expected occurrence). The threshold of three consecutive windows was chosen as a balance between responsiveness (detecting genuine drift within three weeks) and false-alarm tolerance.

The data required is: user queries and LLM responses (specified in the assignment as available), response latency (a standard system performance metric, specified as available), and retrieval distances (computed during the retrieval step of the RAG pipeline from data that is available). All nine descriptors can be computed from the available data without requiring any additional access.

The justification for CDS is that it provides broad, passive monitoring across multiple dimensions simultaneously. TRCI tells us whether the model changed; CDS tells us whether the observable properties of the system's output changed. These are complementary signals. A model update might not affect canary queris much but could significantly shift the distribution of response lengths or citation counts across the full query volume. CDS catches distributional shifts that TRCI, with its fixed canary set, might miss.

The limitations are as follows. CDS monitors the shape of outputs but not their correctness. A response can be the same length, same readability, and include the same number of citations, but contain entirely wrong information. CDS would show GREEN while the system is actively harming users. This is why FDS exists as a complementary metric. Additionally, the descriptor weights are hand-selected. I believe citation count and refusal rate are more important than context token ratio, and the weights reflect this belief, but I have not validated this empirically. In production, the weights should be informed by data on which descriptors best predict user-reported issues.

**Confidence: 4/5.** The multi-descriptor JSD approach is mathematically sound and well-tested. The persistence filter is a well-motivated design choice. The main gap is that the descriptor weights are not empirically validated, and I have not tested CDS on real production traffic to confirm that the thresholds produce useful signals.


### 3.3 Faithfulness Decay Score (FDS)

FDS is the metric most directly relevant to the "remains valid" part of the compliance question. It measures whether the chatbot's answers are faithful to the official source documents, and whether that faithfulness is changing over time. A system can be stable (GREEN on TRCI) and have consistent response properties (GREEN on CDS) while quietly becoming less accurate in its content. FDS is designed to catch exactly this failure mode.

The computation uses a three-step pipeline. First, each response is decomposed into atomic, independently verifiable claims using an LLM. For example, the response "Universal Credit is paid monthly. You must be 18 or over and have less than £16,000 in savings" would be decomposed into three claims: (1) Universal Credit is paid monthly, (2) you must be 18 or over for Universal Credit, and (3) you must have less than £16,000 in savings for Universal Credit. Second, each claim is verified by a separate LLM instance (the "judge", Llama 3.1 8B via Groq) against the retrieved GOV.UK source documents. The judge assigns one of three verdicts: supported (the source document explicitly or logically entails the claim), unsupported (the claim contradicts or goes beyond the source), or ambiguous (the source is relevant but neither clearly supports nor contradicts). Third, the per-response faithfulness score is computed as:

$$f_i = \frac{n_{\text{supported}}}{n_{\text{claims}}}$$

Ambiguous claims are excluded from the numerator by default (configurable via `include_ambiguous`), erring on the side of strictness. If enabled, the numerator becomes $n_{\text{supported}} + n_{\text{ambiguous}}$.

The FDS scalar is then computed by comparing the distribution of per-response faithfulness scores in the current window against a reference baseline using signed JSD:

$$\text{FDS} = \text{sign}\!\left(\bar{f}_{\text{cur}} - \bar{f}_{\text{ref}}\right) \cdot \text{JSD}(F_{\text{ref}},\; F_{\text{cur}})$$

The sign is determined by the direction of the mean shift. If the mean faithfulness in the current window is lower than the reference, the FDS is negative, indicating decay. If it is higher, the FDS is positive, indicating improvement. This directional information is important because I do not want to raise an alarm when faithfulness improves; only decay is a governance concern. Classification uses the absolute value:

| Status | Condition |
|---|---|
| GREEN | $|\text{FDS}| < 0.02$ |
| AMBER | $0.02 \le |\text{FDS}| < 0.10$ |
| RED | $|\text{FDS}| \ge 0.10$ |

Because this metric relies on one AI model judging another, the question of judge reliability is unavoidable. I address it through two mechanisms. The first is a calibration set of 55 hand-crafted entries, each containing a query, the expected claims, and expected faithfulness. These entries cover all six topic areas, four difficulty levels (easy, medium, hard, adversarial), and include several adversarial queries (prompt injection attempts, off-topic requests). The calibration set allows me to measure how far the judge's assessments deviate from known-good answers. With 55 entries, the standard error of the estimated mean faithfulness is approximately 0.04, assuming a standard deviation of 0.3. This is not ideal but is a meaningful improvement over the original set of 5 entries. The second mechanism is cross-validation using a second, independent LLM judge (Llama 3.3 70B). Both judges evaluate the same claims (up to 10 per cross-validation run) against the same context, and I compute Cohen's kappa (Cohen, 1960) to measure inter-rater agreement:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is the observed proportion of agreement between the two judges, and $p_e$ is the expected agreement by chance, computed as:

$$p_e = \sum_{c \in \{\text{supported, unsupported, ambiguous}\}} P_A(c) \cdot P_B(c)$$

Kappa adjusts for chance agreement, which raw agreement rate does not. Following the Landis and Koch (1977) interpretation scale, $\kappa \ge 0.6$ indicates substantial agreement and gives reasonable confidence in the verdicts; $0.4 \le \kappa < 0.6$ indicates moderate agreement; $\kappa < 0.4$ indicates poor agreement, and the system appends an explicit warning to the FDS output: the verdicts may be unreliable.

The data required is: LLM responses (available), the retrieved source contexts used to generate those responses (available from the RAG pipeline), and API access to a judge model for claim verification (available as API inference). The claim decomposition and verification steps are both performed via the model client specified in the assignment. The calibration set and canary queries are part of the system configuration, not additional data sources.

The justification for FDS is that it directly measures the property the compliance question cares about most: whether the system's outputs are faithful to authoritative source information. In a public sector context, faithfulness is not a nice-to-have; it is the fundamental requirement. A chatbot that sounds good but says wrong things is worse than a chatbot that refuses to answer, because wrong information leads to misguided decisions while a refusal at least signals that the user should seek information elsewhere. FDS makes faithfulness measurable and trackable over time.

The limitations of FDS are the most significant of any metric in this system, and I want to be explicit about them. The LLM-as-Judge approach is inherently circular: I am using a language model to evaluate a language model. The judge may have its own biases, may struggle with nuanced claims, and may produce inconsistent verdicts across runs (even at temperature 0.0, due to quantisation effects). Cross-validation with Cohen's kappa mitigates but does not eliminate this concern, particularly because both judges are from the Meta Llama family and may share systematic biases. A production deployment should include periodic human review of judge verdicts to calibrate and validate the automated assessments. Additionally, the calibration set is self-authored, not independently annotated, which means it reflects my understanding of what the correct answers should be rather than a verified ground truth. I investigated whether any publicly available benchmarks could serve this purpose (TruthfulQA, FEVER, HaluEval, MS MARCO, SQuAD 2.0, BEIR, RAGAS) and found that none align with my domain (UK government benefits), evaluation format (claim-level verification against retrieved context), or monitoring focus (temporal faithfulness tracking). This is a limitation of the field, not just of my system, but it is a limitation nonetheless. Finally, FDS measures faithfulness to the source documents, not the correctness of the source documents themselves. If GOV.UK publishes incorrect or outdated information, the chatbot would faithfully reproduce it, and FDS would show GREEN.

**Confidence: 3/5.** The conceptual design is sound, the signed JSD approach is mathematically clean, and the cross-validation mechanism adds genuine value. However, the LLM-as-Judge reliability concern is real, the calibration set is self-authored and not externally validated, and I have not tested the full pipeline on real production data. I am aware of these gaps and believe they can be addressed with human annotation and real-traffic testing, but within the scope of this project, they remain open.


### 3.4 Differential Drift Index (DDI)

DDI addresses the "unbiased and fit for purpose" part of the compliance question. Its focus is not on whether drift is happening (CDS covers that) but on whether it is happening equally across different user groups. The specific concern is that a model change or data shift could degrade the system's quality for one group of users while leaving others unaffected, creating an invisible fairness problem.

The computation begins by segmenting all queries into six topic categories: Universal Credit, Housing Benefit, Disability Benefits, Council Tax, Homelessness, and Pensions. Each query is assigned to a topic based on the topic classification stored in the production log, with keyword-based fallback matching for unclassified queries. For each topic segment, I compute a quality proxy per response:

$$Q_i = 0.4 \cdot \text{completeness}_i + 0.3 \cdot \text{citation}_i + 0.2 \cdot \text{non\_refusal}_i + 0.1 \cdot \text{latency}_i$$

where the four normalised components are defined as:

| Component | Formula | Range |
|---|---|---|
| Completeness | $\min\!\left(\dfrac{\text{completion\_tokens}}{200},\; 1.0\right)$ | [0, 1] |
| Citation | $\mathbb{1}[\text{citation\_count} > 0]$ | {0, 1} |
| Non-refusal | $1 - \mathbb{1}[\text{refusal\_flag}]$ | {0, 1} |
| Latency | $\max\!\left(0,\; 1 - \dfrac{\text{latency\_ms}}{5000}\right)$ | [0, 1] |

The weights reflect my judgment that completeness and citations are the strongest indicators of response quality in a public-sector information context. For each topic segment $s$ with at least 20 responses in both windows, I compute a per-segment drift score:

$$\text{drift}_s = \text{JSD}(Q_s^{\text{ref}},\; Q_s^{\text{cur}})$$

using histogram-based JSD with adaptive Freedman-Diaconis binning. The DDI scalar is the standard deviation of the per-segment drift scores:

$$\text{DDI} = \sigma\!\left(\{\text{drift}_s\}_{s=1}^{S}\right)$$

This is the key design decision and the mathematical insight behind the metric. If all segments drift by the same amount (say, all have JSD = 0.08), the standard deviation is zero and DDI is GREEN, even though overall drift may be significant. Conversely, if most segments have JSD near 0.02 but one segment has JSD near 0.15, the standard deviation is high and DDI will flag it. DDI measures differential drift, not absolute drift. This is the right thing to measure for fairness, because uniform degradation is an overall quality issue (which CDS detects), while non-uniform degradation is a fairness issue (which only DDI detects).

The link to protected characteristics is made through a topic-to-demographic proxy. I do not have demographic data about users, and collecting it would raise its own ethical concerns. Instead, I reason about the likely demographics of people asking about each topic. Questions about disability benefits (PIP, DLA, Attendance Allowance, ESA) are disproportionately asked by disabled persons, who are protected under the Equality Act 2010. Questions about pension credit and state pension are disproportionately asked by elderly persons. Questions about homelessness are disproportionately from vulnerable persons. This proxy is imperfect: a non-disabled person may ask about PIP, and a young person may ask about pensions. But at a population level, the correlation is strong enough to serve as a meaningful fairness signal. The alternative of collecting actual demographic data would be more accurate but introduces privacy risks and consent requirements that may not be appropriate for a government chatbot.

A minimum segment size of 20 queries per window is enforced. Segments with fewer queries are excluded from the DDI calculation and flagged as having insufficient data. This prevents small-sample JSD estimates from dominating the standard deviation calculation. I also track an intersectional flag:

$$\text{intersectional\_flag} = \mathbb{1}\!\left[\max(\text{drift}_s) - \min(\text{drift}_s) > 0.20\right]$$

This triggers when the range of per-segment drifts exceeds 0.20, indicating a particularly severe disparity between the best-served and worst-served segments. Classification follows the same pattern as CDS:

| Status | Condition |
|---|---|
| GREEN | $\text{DDI} < 0.05$ |
| AMBER | $0.05 \le \text{DDI} < 0.15$ |
| RED | $\text{DDI} \ge 0.15$ |

The data required is: user queries with topic classification (derivable from the queries themselves, which are specified as available), LLM responses with the four quality proxy factors (all derivable from available system performance metrics and LLM responses), and the same reference/current window comparison used by CDS. No demographic data or additional access is required.

The justification for DDI rests on the legal and ethical framework of the Equality Act 2010. Section 149 imposes a Public Sector Equality Duty on public authorities to have due regard to the need to eliminate discrimination and advance equality of opportunity between persons who share a protected characteristic and those who do not. If an AI system serving the public degrades its quality disproportionately for queries related to protected characteristic groups, the deploying organisation is potentially in breach of this duty, even if the degradation was unintended and caused by an external model update. DDI provides a quantitative signal that can be used to demonstrate due regard: the organisation is actively monitoring for differential impact and has a process for responding when it is detected.

The limitations of DDI centre on the proxy assumption. The mapping from topic to protected characteristic is a heuristic, not a demographic truth. It is strongest for disability benefits (where the connection between the topic and the protected characteristic of disability is direct and legal) and weakest for council tax (where "all citizens" is not a meaningful protected group). In my six-segment design, only three segments have a clear protected-characteristic mapping (disability, pension/age, homelessness/vulnerability). The other three (Universal Credit, Housing Benefit, Council Tax) serve broader populations. A more sophisticated approach would use query-level features to infer demographic likelihood, but this risks discrimination in itself and was beyond the scope of this project. Additionally, six segments may not capture all intersectional effects: a query about disability benefits for elderly carers spans multiple protected characteristics, and the segment-based approach treats it as a single topic.

**Confidence: 3/5.** The mathematical framework (standard deviation of per-segment JSDs) is clean and defensible, and the Equality Act justification is well-grounded in UK law. However, the topic-to-protected-characteristic proxy is a heuristic that I have not validated against real user demographics, and six topic segments may be too coarse to capture the full range of fairness concerns. I believe the approach is reasonable for a proof-of-concept but would need validation with real data before deployment.

---

## 4. Operationalise

In a production deployment, these four metrics would run on a defined schedule aligned with their data requirements and intended detection speed. TRCI, as an active probing metric, is designed to run daily. Each morning, the system submits the 50 canary queries through the production pipeline, computes cosine similarity against the stored references, and generates a TRCI report. This daily cadence means that a silent model update would be detected within 24 hours. CDS, FDS, and DDI are designed to run weekly, comparing the most recent 7-day window against the rolling 30-day reference baseline. These metrics require sufficient data volume to produce statistically meaningful distributions, and a 7-day current window ensures enough observations per descriptor and per topic segment (assuming moderate traffic volume).

Each metric run produces a structured result containing the scalar value, the traffic-light status, and a human-readable explanation. The explanation is generated programmatically and identifies the specific driver of any alert. For TRCI, it names the worst-performing canary query and its topic. For CDS, it lists the top three contributing descriptors by weighted JSD contribution. For FDS, it identifies the query with the lowest faithfulness score and the specific unsupported claims. For DDI, it names the most-drifted and least-drifted topic segments and their associated protected characteristic proxies. This level of detail is important because a traffic light alone is not actionable; the operator needs to know what to investigate.

The thresholds for each metric are as follows. TRCI uses GREEN at mean similarity 0.95 or above (with p10 at 0.80 or above), AMBER between 0.90 and 0.95, and RED below 0.90. CDS uses GREEN below 0.05, AMBER between 0.05 and 0.15, and RED at or above 0.15 after persisting for three consecutive windows. FDS uses GREEN when the absolute FDS is below 0.02, AMBER between 0.02 and 0.10, and RED at or above 0.10. DDI uses GREEN below 0.05, AMBER between 0.05 and 0.15, and RED at or above 0.15. These thresholds were chosen based on the mathematical properties of the measures and the domain context, but they are starting points. In a real deployment, the first two to four weeks of operation would be used to establish baselines: run all metrics on production data, observe the natural variation in each metric under stable conditions, and adjust thresholds so that GREEN encompasses normal variation while AMBER and RED correspond to genuine anomalies.

The four metrics complement each other in diagnosis. If TRCI goes RED while the other three remain GREEN, the model likely changed but the change did not materially affect response quality or fairness. This would be logged and monitored but would not require immediate intervention. If FDS goes RED while TRCI and CDS remain GREEN, the system's observable properties look normal but the content accuracy has declined. This is the most dangerous scenario because the system appears healthy while providing wrong information, and it warrants immediate investigation: reviewing the lowest-faithfulness queries, checking whether the retrieved documents are still current, and potentially adding a disclaimer to chatbot responses until the issue is resolved. If DDI goes RED while CDS remains GREEN, the overall response quality is stable but is distributed unfairly across user groups. This is a fairness concern that should be escalated to the organisation's equality impact assessment process. If all four metrics go RED simultaneously, a major system change has occurred and a full review is needed, potentially including rollback to a previous known-good configuration if available.

The response process follows a graduated escalation. GREEN requires no action; the monitoring results are logged for audit purposes. AMBER triggers an investigation: the responsible team reviews the metric explanation, examines the specific queries or descriptors driving the signal, and prepares a preliminary assessment.  RED triggers an action: for TRCI and CDS, this means comparing the current canary or descriptor outputs against their historical baselines to identify what changed; for FDS, this means reviewing the specific unsupported claims and determining whether they represent a model issue, a retrieval issue, or a data freshness issue; for DDI, this means identifying the affected segment, documenting the disparity, and assessing whether the differential impact is harmful. Every metric result, regardless of status, is logged with a timestamp, scalar value, status, and full explanation. This creates the audit trail that the compliance question implicitly requires: the organisation can demonstrate, at any point, what it was monitoring, what it detected, and what it did in response.

There are practical constraints that affect how this system operates. The Groq free tier imposes a rate limit of approximately 12,000 tokens per minute, which means a full TRCI probe of 50 canaries takes several minutes including mandatory delays between requests. FDS, which requires multiple API calls per sampled query (one for claim decomposition, one per claim for verification, plus cross-validation), is even more resource-intensive. In a production environment with paid API access, these constraints would be significantly relaxed, but for the current proof-of-concept, monitoring runs are batched and rate-limited accordingly. The system includes retry logic with exponential backoff for rate limit errors, and canary initialisation (the one-time setup that creates reference responses) uses a 20-second delay between queries to stay within the free tier's throughput limits.

All monitoring configuration is centralised in a YAML configuration file that specifies thresholds, weights, window sizes, and sampling parameters for every metric. This allows the operator to adjust the system's sensitivity without modifying code. For example, tightening the FDS GREEN threshold from 0.02 to 0.01 would make the system more sensitive to small faithfulness changes, while increasing the CDS persistence threshold from 3 to 5 windows would make it more tolerant of transient distribution shifts. This configurability is an important operational consideration because the optimal settings depend on the specific deployment context, and the values that work for a benefits chatbot with moderate traffic may not be appropriate for a high-volume service handling thousands of queries per day.

Finally, the system was validated with a test suite of 192 unit and integration tests covering all four metrics, the JSD statistical functions, the adaptive binning mechanism, the cross-validation engine, the production logging system, and the text processing descriptors. All 192 tests pass. The tests validate mathematical properties (JSD symmetry, boundedness, identity of indiscernibles), classification boundary conditions (at exact threshold values and at offsets of 0.001), edge cases (no data, single data point, all canaries failing), and integration between components (FDS with cross-validation enabled, CDS with persistence tracking). While passing tests do not guarantee correct behaviour on real data, they provide confidence that the code implements the intended mathematical specifications faithfully.

---

## References and Data Sources

### Academic References

1. Lin, J. (1991). Divergence Measures Based on the Shannon Entropy. *IEEE Transactions on Information Theory*, 37(1), 145-151. [https://doi.org/10.1109/18.61115](https://doi.org/10.1109/18.61115)

2. Freedman, D. and Diaconis, P. (1981). On the histogram as a density estimator: L₂ theory. *Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete*, 57(4), 453-476. [https://doi.org/10.1007/BF01025868](https://doi.org/10.1007/BF01025868)

3. Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37-46. [https://doi.org/10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104)

4. Landis, J.R. and Koch, G.G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159-174. [https://doi.org/10.2307/2529310](https://doi.org/10.2307/2529310)

### Legal References

5. Equality Act 2010, Section 149 (Public Sector Equality Duty). [https://www.legislation.gov.uk/ukpga/2010/15/section/149](https://www.legislation.gov.uk/ukpga/2010/15/section/149)

### Data Sources

6. GOV.UK Content API. 33 pages ingested covering Universal Credit, Housing Benefit, Council Tax, Disability Benefits, Homelessness, and Pensions. Licensed under the Open Government Licence v3.0. [https://content-api.publishing.service.gov.uk/](https://content-api.publishing.service.gov.uk/), [https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

7. Canary Query Set. 50 hand-authored queries across 6 topic areas, with approximately 35 initialised reference responses. Synthetic; not from any external benchmark.

8. FDS Calibration Set. 55 hand-authored entries with expected claims and faithfulness scores, covering all topics, four difficulty levels, and adversarial queries. Synthetic; not independently annotated.

### External Services

9. Pinecone Vector Database with Llama Text Embed v2 (1024 dimensions, cosine similarity). [https://www.pinecone.io/](https://www.pinecone.io/)

10. Groq LLM Inference. Primary model: Llama 3.3 70B Versatile. Judge model: Llama 3.1 8B Instant. [https://groq.com/](https://groq.com/)

11. Meta Llama Model Family. [https://llama.meta.com/](https://llama.meta.com/)

12. LiteLLM (multi-provider LLM gateway). [https://github.com/BerriAI/litellm](https://github.com/BerriAI/litellm)

### Tools and Frameworks

13. Python 3.11, NumPy, SciPy (statistical functions), Streamlit (dashboard), pytest (test suite).

14. Full project source code and test suite available in the accompanying repository.

### Video Resources

15. Jensen-Shannon Divergence — Explained. [https://youtu.be/bqjZK9tkWdk?si=BaVvt8qgxHFA6w0b](https://youtu.be/bqjZK9tkWdk?si=BaVvt8qgxHFA6w0b)

16. ML Drift Explained. [https://youtu.be/uOG685WFO00?si=LiXW41MSxm4vWhOH](https://youtu.be/uOG685WFO00?si=LiXW41MSxm4vWhOH)

17. Drift Monitoring Techniques. [https://youtu.be/eQ6cGzDUtMU?si=KSJ5VzZ2zh9VVoqQ](https://youtu.be/eQ6cGzDUtMU?si=KSJ5VzZ2zh9VVoqQ)

18. Building Trustworthy AI — IBM. [https://youtu.be/4gC3oueK9Gc?si=fPHouDopgbGuZ0vK](https://youtu.be/4gC3oueK9Gc?si=fPHouDopgbGuZ0vK)

---

## Appendix: Foundational Drift Concepts

The following figures are drawn from foundational material on data and model drift that informed the design of the four monitoring metrics. Each figure is accompanied by an explanation of how the concept applies to the NLQ/RAG chatbot scenario addressed in this document.

### Figure 1 — Temporal Patterns of ML Drift

![Temporal Patterns of ML Drift](images/MLDrift.png)

This diagram illustrates five temporal patterns through which drift manifests in production ML systems: abrupt, gradual, incremental, recurrent, and blip. In the context of this chatbot, each pattern maps to a distinct operational scenario. An **abrupt** shift corresponds to a silent LLM provider model update or a wholesale knowledge base refresh — the system's behaviour changes overnight. **Gradual** drift arises when citizen language evolves slowly (e.g., increasing use of abbreviations like "UC" for Universal Credit) or when a policy is phased in over months. **Incremental** drift captures seasonal transitions, such as the progressive shift from tax-related queries in January–April to benefits-related queries in summer. **Recurrent** drift describes annual cycles — council tax questions peaking every April, school admissions every September — which are expected and should not trigger alerts. A **blip** is a temporary anomaly, such as a viral social media post driving unusual traffic for 48 hours. The CDS persistence filter (requiring three consecutive windows above threshold before escalating to RED) is specifically designed to distinguish sustained drift from transient blips.

### Figure 2 — Key Types of Drift in Machine Learning

![Key Types of Drift](images/KeyDriftsinML.png)

This figure presents the four fundamental drift types defined over the joint probability distribution $P(X, Y)$: concept drift ($P(Y|X)$ changes), data/feature drift ($P(X)$ changes), label drift ($P(Y)$ changes), and virtual drift (where $P(X)$ changes but $P(Y|X)$ remains stable). For the benefits chatbot, **concept drift** is the most dangerous form — it means the correct answer to a given query has changed (e.g., a savings threshold for Universal Credit is revised by legislation), but the system continues to give the old answer with full confidence. **Data drift** manifests as shifts in the kinds of questions citizens ask, which CDS monitors through its nine response descriptors. **Label drift** appears as shifts in the distribution of correct response types, which FDS tracks through faithfulness scoring. **Virtual drift** occurs when a new demographic begins using the chatbot with different phrasing but the same underlying questions — the system should handle this gracefully, and CDS would register it as a distributional shift without FDS necessarily deteriorating. A key insight is that simultaneous data drift and label drift can cancel out, leaving concept drift unchanged; this is why monitoring multiple dimensions independently (as the four metrics do) is essential.

### Figure 3 — Drift Examples Mapped to a Chatbot Scenario

![Drift Examples for a Loan Application Model](images/ExampleofDrift.png)

This figure originally illustrates drift in a loan application model. The parallels to a government benefits chatbot are direct. **Concept drift** in the loan model (an income level that was previously creditworthy becomes risky) maps to an eligibility threshold change in the chatbot: if the income limit for Housing Benefit drops from £16,000 to £14,000, the same query ("I earn £15,000 — am I eligible?") now has the opposite correct answer. **Label drift** (more creditworthy applications appearing) maps to a spike in Universal Credit queries shifting from 20% to 40% of all traffic, altering the expected distribution of response types. **Feature drift** (applications coming from a new region) maps to citizens from a newly onboarded geographic area using the chatbot, introducing new vocabulary and query patterns. This mapping demonstrates why the four metrics were designed as separate instruments: TRCI detects model-level changes, CDS detects distributional shifts (feature/label drift), FDS detects accuracy degradation (concept drift consequences), and DDI detects when these drifts affect user groups unequally.

### Figure 4 — Triggers of ML Model Drift

![Triggers of ML Model Drift](images/TriggersofDrift.png)

This diagram distinguishes two fundamental categories of drift triggers: **real change** in the underlying world, and **data integrity** failures in the system pipeline. For the chatbot, real change includes new legislation altering eligibility rules, LLM provider model updates changing response behaviour, and seasonal or demographic shifts in the user population. Data integrity issues include RAG pipeline bugs (retrieval returning wrong documents due to index corruption), embedding model updates silently changing similarity scores, preprocessing errors stripping important information from queries, and knowledge base corruption (documents uploaded with wrong metadata). The distinction matters operationally because the response is different: real change requires updating the knowledge base, recomputing baselines, and potentially re-initialising canary references, while data integrity issues require debugging and fixing the engineering pipeline. In my system, TRCI is particularly sensitive to data integrity issues (a corrupted index would immediately change canary response similarity), while CDS and DDI are better at surfacing real-world changes that shift response distributions gradually.

### Figure 5 — Statistical Methods for Data Drift Monitoring

![Statistical Methods for Data Drift Monitoring](images/StatisticalMethods.png)

This figure surveys statistical methods for drift detection, including Population Stability Index (PSI), KL Divergence, Jensen-Shannon Divergence (JSD), and the Kolmogorov-Smirnov (KS) test. I chose JSD as the core statistical measure for all four metrics for specific reasons visible in this comparison. **PSI** is commonly used for categorical features but lacks a principled bounded range. **KL Divergence** is asymmetric ($D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$) and goes to infinity when one distribution assigns zero probability to an event the other considers possible — a frequent occurrence with small sample sizes in production monitoring. **JSD** resolves both problems: it is symmetric, bounded between 0 and $\ln(2)$, and always defined even with mismatched supports. The **KS test** is non-parametric and useful for continuous distributions, but produces a p-value rather than a magnitude, making it less suitable for threshold-based traffic-light monitoring. The bounded nature of JSD is particularly important for my system because it allows meaningful threshold-setting: GREEN < 0.05, AMBER < 0.15, and RED ≥ 0.15 have consistent interpretations regardless of the data distribution. The adaptive binning (Freedman-Diaconis rule) I implemented addresses the histogram estimation bias that would otherwise make JSD unreliable on the small sample sizes typical of weekly monitoring windows.
