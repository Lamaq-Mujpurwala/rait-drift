# Detecting and Responding to Data & Model Drift

**Author:** Lamaq  
**Date:** 25 February 2026  

---

## What This Document Is

This document explains the full reasoning behind the RAIT monitoring system — what it does, why it was designed this way, and how it would work in practice. It is written to be read by someone who is not deeply technical but wants to understand the logic.

The ethical question we were given:

> *"How does the organisation detect and respond to changes in data or model behaviour over time to ensure the system remains valid, unbiased, and fit for purpose?"*

The scenario: a UK public-sector AI chatbot answers questions about government services like Universal Credit, Housing Benefit, PIP, and Council Tax. The chatbot uses a technique called Retrieval-Augmented Generation (RAG) — it fetches relevant information from official GOV.UK pages, then uses a language model to write an answer grounded in that information.

The problem is: what happens when something changes silently? What if the model starts giving different answers tomorrow than it gave today, and nobody notices?

That is what drift is. And that is what this system is built to catch.

---

## Part 1: What Is Drift, and Why Should We Care?

### The Simple Explanation

Imagine a customer service desk at a council office. Every day, the same staff answer the same kinds of questions: "How do I apply for Universal Credit?", "What council tax band am I in?", "Can I get help with my rent?"

Now imagine that one morning, without telling anyone, the council replaces the entire staff with new people. The new staff are polite, they answer quickly, but some of them give slightly wrong information. Some quote the wrong savings threshold. Some forget to mention the 5-week waiting period for Universal Credit. Some are perfectly fine.

No one announced the change. The building looks the same. The phones still ring. But the *quality of information* has silently shifted.

That is exactly what happens when an AI model updates. The API provider (in our case, Groq) can change the underlying language model at any time. There is no notification. The chatbot still responds. But the distribution of its answers — how long they are, how accurate they are, which topics they handle well — may have changed.

### Why This Matters for Real People

This is not an abstract concern. When someone asks a government chatbot "Am I eligible for Universal Credit?", they may be making real financial decisions based on the answer. If the system quietly starts:

- Omitting the £16,000 savings threshold
- Giving less accurate disability benefits information than Universal Credit information
- Refusing to answer questions it used to answer confidently

...then real people get wrong information. And the organisation responsible has no idea it is happening.

That is the governance problem drift creates. Not that the system breaks — but that it *degrades invisibly*.

### The Two Types of Drift We Care About

**Data drift** — the nature of incoming questions changes. Maybe more people start asking about pension deferral, or a policy change causes a surge in disability-related queries. The system was designed for one distribution of queries and is now facing a different one.

**Model drift** — the model itself changes behaviour. This can happen because the API provider updates the model, or because the retrieval index gets stale while GOV.UK pages are updated, or because the embedding model shifts how it represents text. The input is the same, but the output is different.

Both are problems. Both are invisible without monitoring. Our system is designed to catch both.

---

## Part 2: The Design Thinking — From Question to Metrics

### Starting From the Compliance Question

The question asks us to detect AND respond. That means we need:

1. **Detection** — numbers that tell us something has changed
2. **Diagnosis** — enough detail to figure out *what* changed
3. **Response** — a clear signal that tells an operator what to do

A single "overall score" would not be enough. If we just computed one number and said "the system is 73% good", that tells no one anything useful. An operator needs to know: Is the model itself different? Are the answers less accurate? Is the problem affecting everyone equally, or is one group getting worse service?

That reasoning led us to design four separate metrics, each watching a different dimension of the problem.

### Why Four Metrics?

Think of it like a health check-up. You would not go to the doctor and ask for a single number that represents your entire health. You would get your blood pressure measured, your cholesterol checked, your heart rate recorded, and maybe an X-ray if something hurt. Each test looks at a different system, and together they build a picture.

Our four metrics work the same way:

| Metric | What It Watches | Analogy |
|--------|----------------|---------|
| **TRCI** | Has the model itself changed? | Taking the same patient's temperature every morning — if it suddenly changes, something happened |
| **CDS** | Has the overall shape of responses changed? | Comparing blood test distributions across a population — are the numbers shifting? |
| **FDS** | Are the answers still faithful to the source documents? | Checking whether a translator is still translating accurately, or starting to paraphrase freely |
| **DDI** | Is drift affecting all user groups equally? | Checking whether a hospital's care quality dropped only in one department, while others stayed the same |

Each metric answers a distinct part of the compliance question. Together, they provide comprehensive coverage.

---

## Part 3: The Four Metrics — How They Work

### Metric 1: TRCI — Temporal Response Consistency Index

**The idea in one sentence:** Ask the same questions every day and see if you get the same answers.

**How it works:**

We have a set of 50 "canary" questions — like the canary in a coal mine. These are questions we wrote ourselves, spanning all six topics (Universal Credit, Housing Benefit, Council Tax, Disability Benefits, Homelessness, and Pensions). Before we start monitoring, we run each canary through the system and save the response. That saved response becomes the "reference" — what the system said when we believed it was working correctly.

Every monitoring cycle (e.g. daily), we re-submit the same canary questions. The system produces new responses. We then compare each new response to its reference using cosine similarity — a mathematical way of measuring how similar two pieces of text are in meaning, not just word-for-word.

If the new response to "What is Universal Credit?" is semantically identical to the original reference, the similarity will be very close to 1.0. If the model has changed and now gives a different (or worse) answer, the similarity drops.

We then look at the distribution of all 50 similarities:

- **Mean similarity** — the average across all canaries. If the mean drops, something systematic changed.
- **10th percentile (p10)** — the worst 10% of canaries. This catches the scenario where most canaries are fine but a few have drifted badly.

**Classification:**

| Status | Condition | What It Means |
|--------|-----------|---------------|
| GREEN | Mean ≥ 0.95 AND p10 ≥ 0.80 | No detectable change. System is behaving as expected. |
| AMBER | Mean ≥ 0.90 but below GREEN | Something may have shifted. Investigate. |
| RED | Mean < 0.90 OR p10 < 0.80 | Significant change. Likely a model update or system issue. |

**Why these thresholds?** A mean of 0.95 is deliberately conservative. Natural variation in language model outputs means you will never get exactly 1.0 — the same question asked twice will produce slightly different wording. But if the meaning stays the same, cosine similarity stays above 0.95. A drop below 0.90 means the answers are substantively different.

**What it catches:** Silent model updates, embedding model changes, retrieval system failures.

**What it does NOT catch:** A model that is consistently wrong from day one, or gradual drift that never drops below the threshold in a single day.

**How we respond:** RED on TRCI means "stop and investigate." The report identifies which specific canaries drifted most, so the operator can see which topics are affected. The immediate action is to review the worst canaries, compare old and new responses, and determine whether the change improved or degraded quality.

---

### Metric 2: CDS — Composite Drift Signal

**The idea in one sentence:** Track nine different properties of the system's responses and see if any of them shift over time.

**How it works:**

Every response the chatbot produces has measurable properties:

- How long is the response? (word count)
- How positive or negative is the tone? (sentiment)
- How readable is the text? (readability score)
- How many citations or GOV.UK references does it include?
- Does it use hedging language like "might" or "possibly"?
- Did it refuse to answer?
- How long did it take to respond? (latency)
- How relevant were the retrieved documents? (retrieval distance)
- What proportion of the response was based on retrieved context vs. generated?

We call these "descriptors." We measure all nine for every query over a reference window (typically the last 30 days) and a current window (the last 7 days). For each descriptor, we compare the two windows using Jensen-Shannon Divergence (JSD) — a measure of how different two distributions are.

Think of it this way: if response lengths last month ranged from 100 to 300 words with an average of 180, and this week they range from 200 to 500 words with an average of 350, that is a significant shift in behaviour. JSD quantifies exactly how big that shift is.

Each descriptor gets a weight reflecting its importance (e.g., citation count and refusal rate are weighted higher at 0.15 because they directly affect information quality). The final CDS score is the weighted average of all nine JSD values.

**A key design choice — the persistence filter:**

We deliberately do not raise a RED alarm the first time CDS crosses a threshold. Why? Because drift monitoring on real data is noisy. Random variation in queries, a public holiday causing different usage patterns, or a brief API outage can all cause a temporary blip. If we alarmed on every blip, operators would quickly learn to ignore the alerts — which is worse than having no alerts at all.

Instead, CDS requires the score to stay above the threshold for **3 consecutive monitoring windows** before escalating to RED. This is the persistence filter. It means: "We are not worried about a bad day. We are worried about a bad week."

**Classification:**

| Status | Condition |
|--------|-----------|
| GREEN | CDS < 0.05 |
| AMBER | 0.05 ≤ CDS < 0.15 |
| RED | CDS ≥ 0.15 AND persisted for 3+ windows |

**What it catches:** Gradual changes in response characteristics, retrieval degradation, shifts in tone or style, changes in refusal behaviour.

**What it does NOT catch:** Whether the content is actually correct. A response can be the same length, same tone, and same readability — but contain wrong information. That is what FDS is for.

**How we respond:** CDS tells us *what* changed through the per-descriptor breakdown. If citation count dropped, the retrieval system may be failing. If refusal rate increased, the model may have become more conservative. If latency spiked, there may be an infrastructure issue. The response depends on which descriptors are driving the drift.

---

### Metric 3: FDS — Faithfulness Decay Score

**The idea in one sentence:** Check whether the chatbot's answers actually match what the official documents say.

**This is the most important metric for a public-sector chatbot.** A response can sound confident, be well-written, and arrive quickly — but if it says the savings threshold for Universal Credit is £15,000 when GOV.UK says £16,000, it is actively harmful.

**How it works:**

For a sample of recent queries, we run a three-step pipeline:

**Step 1 — Claim decomposition.** Break each response into individual factual claims. For example, the response "Universal Credit is a monthly payment for people on low income. You must be 18 or over to apply." becomes two claims:
1. "Universal Credit is a monthly payment for people on low income."
2. "You must be 18 or over to apply for Universal Credit."

**Step 2 — Claim verification.** For each claim, we ask a second language model (the "judge"): "Is this claim supported by the official source documents?" The judge reads the original GOV.UK text and gives a verdict:
- **Supported** — the claim matches the source
- **Unsupported** — the claim contradicts or goes beyond the source
- **Ambiguous** — the source is relevant but neither clearly supports nor contradicts

**Step 3 — Faithfulness scoring.** For each response: faithfulness = (number of supported claims) / (total claims). A response with 4 supported claims out of 5 total gets a faithfulness of 0.80.

We then compare the distribution of faithfulness scores in the current window against a reference baseline using Signed JSD. The "signed" part is crucial — it tells us not just that the distribution changed, but *in which direction*:

- Negative FDS → faithfulness is declining (responses are getting LESS accurate)
- Positive FDS → faithfulness is improving
- Near zero → no meaningful change

**Why signed JSD matters:** Regular JSD would just say "the distributions are different." But we do not want to alarm when faithfulness *improves*. A model update that makes answers more accurate is good news. We only care about decay.

**Cross-validation — checking the checker:**

There is an obvious weakness in this approach: we are using an AI model to judge another AI model. How do we know the judge is reliable?

We address this in two ways:

1. **Calibration set** — 55 hand-crafted queries with expected faithfulness scores. We know what the judge *should* say for these, so we can measure how far off it is.

2. **Cross-validation** — we run a second, different model as an independent judge on the same claims and compute Cohen's kappa (a standard measure of inter-rater agreement). If two judges independently reach the same verdicts, we have more confidence in the results. If they disagree, we flag it.

| Kappa | Interpretation |
|-------|----------------|
| ≥ 0.6 | Substantial agreement — judge is reliable |
| 0.4-0.6 | Moderate — proceed with caution |
| < 0.4 | Poor — verdicts are unreliable, flag in output |

**Classification:**

| Status | Condition |
|--------|-----------|
| GREEN | \|FDS\| < 0.02 |
| AMBER | 0.02 ≤ \|FDS\| < 0.10 |
| RED | \|FDS\| ≥ 0.10 |

**What it catches:** The model hallucinating, making up facts, quoting wrong thresholds, or gradually becoming less grounded in the source documents.

**What it does NOT catch:** Problems with the source documents themselves. If GOV.UK publishes incorrect information, the system would faithfully report it and FDS would show GREEN.

**How we respond:** RED on FDS is the most serious alert. It means the chatbot is giving people wrong information. The response is:
1. Identify which queries have the lowest faithfulness scores
2. Review the specific unsupported claims
3. Determine whether this is a model issue (retrain/rollback) or a retrieval issue (re-index GOV.UK pages)
4. Consider temporarily displaying a disclaimer or routing to human agents

---

### Metric 4: DDI — Differential Drift Index

**The idea in one sentence:** Even if drift is happening, is it affecting all user groups equally?

**Why this matters:**

Imagine the chatbot gets 5% worse overall. That is concerning, but manageable. Now imagine it gets 15% worse for disability benefits queries but stays the same for council tax queries. That is not just a performance issue — it is a fairness issue. People asking about disability benefits are disproportionately likely to be disabled. Under Section 149 of the Equality Act 2010, public bodies have a duty to consider the impact of their services on people with protected characteristics.

DDI is designed to catch this specific failure mode: **non-uniform drift**.

**How it works:**

We group all queries by topic: Universal Credit, Housing Benefit, Disability Benefits, Council Tax, Homelessness, Pensions. For each group, we compute a quality proxy — a score that captures how well the system is serving that group, based on:

- **Completeness** (40% weight) — how detailed are the responses?
- **Citation count** (30%) — are responses grounded in source documents?
- **Non-refusal rate** (20%) — is the system actually answering or just refusing?
- **Latency** (10%) — are some topics taking abnormally long?

For each topic segment, we compare the quality distribution between the reference and current windows using JSD. This gives us a drift score per segment.

The DDI is then the **standard deviation** of those per-segment drift scores.

This is the key mathematical insight: DDI does not measure whether drift is happening (CDS does that). DDI measures whether drift is happening **unevenly**. If every segment drifts by the same amount, the standard deviation is close to zero — DDI stays GREEN even though overall quality dropped. But if one segment drifts much more than others, the standard deviation is high — DDI goes RED.

**The topic-to-protected-characteristic proxy:**

We do not have demographic data about users (nor should we, for privacy reasons). But we can reason that:

- Questions about disability benefits are more likely to come from disabled people
- Questions about pensions are more likely to come from elderly people
- Questions about homelessness are more likely to come from vulnerable people

This is a proxy, not a certainty. A non-disabled person might ask about PIP. But at a population level, the correlation is strong enough to be a useful signal for fairness monitoring.

**Classification:**

| Status | Condition |
|--------|-----------|
| GREEN | DDI < 0.05 |
| AMBER | 0.05 ≤ DDI < 0.15 |
| RED | DDI ≥ 0.15 |

**What it catches:** A model that degrades its quality specifically for one user group while maintaining performance for others.

**What it does NOT catch:** Overall uniform degradation (CDS catches that). Also, the topic-based proxy may miss cases where within a single topic, certain sub-groups are affected differently.

**How we respond:** RED on DDI requires identifying which segment is the outlier. The report says "Disability benefits queries drifted 3× more than the average." The action is to investigate that specific topic — check the retrieval quality for disability-related pages, review recent responses, and determine whether the disparity is harmless (e.g., those pages were recently updated) or harmful (e.g., the model got worse at understanding disability terminology).

---

## Part 4: The Mathematical Backbone — JSD

Three of our four metrics rely on Jensen-Shannon Divergence. Here is why.

**The problem:** We need to compare two distributions. "Were the response lengths last week distributed the same way as this week?" This is fundamentally a distributional comparison problem.

**Why JSD specifically?**

We needed a measure that is:
- **Symmetric** — JSD(A, B) = JSD(B, A). The comparison should not depend on which window we call "reference."
- **Bounded** — JSD always falls between 0 and ln(2) ≈ 0.693. This makes thresholds meaningful and interpretable.
- **Zero when identical** — JSD(A, A) = 0. No false positives from comparing a distribution to itself.
- **Smooth** — small changes produce small JSD values. Unlike KL divergence, JSD never gives infinity.

KL Divergence was the natural first candidate but fails on the symmetry and smoothness requirements. A simple difference of means would miss distributional shape changes. JSD hits the right balance.

**The small-sample problem we had to solve:**

JSD on continuous data requires converting values to histograms. The number of histogram bins matters enormously. Too many bins relative to data size creates sparse histograms with artificial divergence. With 30 data points and 50 bins, most bins are empty, and JSD reports high divergence even between identical distributions.

The theoretical bias is approximately: bias ≈ (bins − 1) / (2N). For N=30 and bins=50, that is 49/60 ≈ 0.82 — meaning the bias alone is larger than our RED threshold.

We solved this with adaptive binning using the Freedman-Diaconis rule, which automatically selects the number of bins based on the data's spread and size. For small samples, it uses fewer bins (reducing bias). For large samples, it uses more (preserving detail). We tested this with Monte Carlo simulation: the bias dropped from ~0.82 to ~0.22 — a 3.7× reduction.

---

## Part 5: How Monitoring Works in Practice

### The Operational Loop

In production, the monitoring runs on a schedule:

**Daily:** TRCI probes all canary queries. This is the early warning system — if the model changed overnight, TRCI catches it by morning.

**Weekly:** CDS, FDS, and DDI run on the past week's production data compared to the previous 30-day reference window. These need enough data to produce statistically meaningful distributions.

Each metric produces a traffic-light result: GREEN, AMBER, or RED. The operator sees a dashboard with four lights.

### Interpreting the Signals Together

The power of the system is in how the four metrics combine:

| TRCI | CDS | FDS | DDI | Likely Situation |
|------|-----|-----|-----|------------------|
| RED | GREEN | GREEN | GREEN | Model changed but answers are still faithful. Possibly a minor version update. Monitor closely. |
| GREEN | RED | GREEN | GREEN | Response properties shifted (e.g., longer responses, more hedging) but content is still accurate. Low urgency. |
| GREEN | GREEN | RED | GREEN | Faithfulness is declining while everything else looks normal. Most dangerous — the system *looks* fine but is giving wrong information. High urgency. |
| GREEN | GREEN | GREEN | RED | Overall quality is stable but one user group is getting worse service. Fairness concern. Investigate which topic segment is affected. |
| RED | RED | RED | GREEN | Major model change degraded everything uniformly. Likely a model update that needs rollback. |
| RED | RED | RED | RED | Major model change degraded everything AND did so unevenly across groups. Highest urgency. |

### The Response Framework

When a metric goes RED, the response depends on which metric and what the details show:

1. **Check the explanation.** Every metric result includes a human-readable explanation that identifies the specific issue (e.g., "Faithfulness is decaying. Worst query: 'What are the PIP daily living components?' — 1/4 claims supported.")

2. **Diagnose the root cause.** Is it a model change (TRCI would also be RED)? A retrieval failure (citation count would drop in CDS)? A data freshness issue (GOV.UK pages may have been updated but not re-ingested)?

3. **Act proportionally.** AMBER means "investigate and prepare." RED means "act now." For FDS RED on a public-sector chatbot, the appropriate response might be adding a disclaimer, routing sensitive queries to human agents, or rolling back to the previous model version if possible.

4. **Document the incident.** The monitoring system logs every metric result with a timestamp, score, status, and explanation. This creates the audit trail that the compliance question implicitly asks for.

---

## Part 6: Limitations — What We Are Honest About

### Things We Know Are Not Perfect

**1. The corpus is small.** We have 33 GOV.UK pages. A real deployment would need hundreds. Our JSD estimates have higher variance than they would with more data, though adaptive binning mitigates the worst of it.

**2. The LLM-as-Judge is imperfect.** FDS relies on one AI model judging another. The judge (Llama 3.1 8B) is a smaller model that may make mistakes — especially on nuanced or partially supported claims. We mitigate this with cross-validation and a 55-entry calibration set, but we cannot eliminate the uncertainty entirely. A proper production deployment would include periodic human review of judge verdicts.

**3. Topic segments are a proxy for protected characteristics.** DDI assumes that questions about disability benefits come from disabled people. This is a reasonable population-level assumption but not guaranteed for any individual query. Demographic data would be better, but collecting it raises its own privacy and ethical concerns.

**4. Thresholds are not calibrated on real traffic.** Our GREEN/AMBER/RED boundaries (0.05, 0.15 for CDS and DDI; 0.02, 0.10 for FDS; 0.95, 0.90 for TRCI) are educated starting points based on the mathematical properties of the measures. In production, you would run the system for 2-4 weeks on real data, observe the natural baseline variation, and adjust thresholds so that GREEN covers normal variation and AMBER/RED catch genuine anomalies.

**5. No external benchmark exists for this domain.** We looked for public human-annotated datasets that could independently validate our faithfulness evaluation. None exist for UK benefits chatbot accuracy. TruthfulQA, FEVER, HaluEval, and other academic benchmarks are in different domains with different evaluation formats. Our calibration set is self-authored, not independently annotated. This is a limitation of the field, not just our system.

**6. Rate limits constrain monitoring speed.** The free-tier Groq API allows 12,000 tokens per minute. TRCI needs one API call per canary, FDS needs multiple calls per sampled query. A full monitoring run can take 10+ minutes just waiting for rate limits. A production deployment would need paid-tier API access.

**7. We cannot pin model versions.** On free-tier APIs, we have no control over when the provider updates their model. Ironically, this is exactly the kind of silent change our metrics are designed to detect — but it also means we cannot do controlled before/after testing.

### What Would Come Next

If this system were being prepared for production deployment, the natural next steps would be:

1. **Run on real traffic for 2-4 weeks** to establish baselines and calibrate thresholds
2. **Commission human annotation** of a sample of chatbot responses to validate FDS judge accuracy
3. **Expand the GOV.UK corpus** to cover more of the benefits landscape
4. **Integrate with an alerting system** (email, Slack, PagerDuty) so that RED results trigger real notifications
5. **Build a feedback loop** where monitoring results inform re-training or re-indexing decisions

---

## Part 7: Connecting Back to the Compliance Question

The question was: *"How does the organisation detect and respond to changes in data or model behaviour over time to ensure the system remains valid, unbiased, and fit for purpose?"*

Our answer, structured:

**Detection:**
- TRCI detects model-level changes by probing with canary queries daily
- CDS detects response distribution shifts across 9 measurable descriptors
- FDS detects declining faithfulness to source documents
- DDI detects uneven drift that could indicate bias

**Response:**
- Traffic-light classification (GREEN/AMBER/RED) provides clear, actionable signals
- Per-metric explanations identify the specific issue (which canaries drifted, which descriptors shifted, which claims were unsupported, which topic segments were affected)
- The persistence filter (CDS) prevents false alarms from transient noise
- Cross-validation (FDS) warns when the judge itself is unreliable
- Audit logging provides a documented trail of all monitoring results

**Validity:**
- 192 unit tests validate the mathematical properties of all metrics
- All metrics are bounded, interpretable, and produce scalar values suitable for dashboards
- Adaptive binning ensures statistical reliability even on small samples

**Unbiased:**
- DDI specifically monitors for differential impact across user groups linked to protected characteristics
- The Equality Act 2010 Section 149 PSED is directly addressed through topic-segment fairness monitoring

**Fit for purpose:**
- Each metric targets a distinct failure mode relevant to public-sector information chatbots
- The combined four-metric system covers model changes, response quality, content accuracy, and fairness
- The design is extensible — adding more topics, more descriptors, or more canaries does not require architectural changes

---

*End of document.*
