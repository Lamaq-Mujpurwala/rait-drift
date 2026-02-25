# RAIT Intern Assessment

## Introductory Walkthrough — Please Read Before You

## Begin

**What This Assessment Is About**
AI systems deployed in the real world don't just need to perform well — they need to be
measurable, accountable, and explainable to the people they affect and the organisations
responsible for them. This is the core of what RAIT works on: building the frameworks and
metrics that make AI governance operational rather than theoretical.
This assessment is designed to test one specific skill: can you construct a reasoned chain
from a compliance question — something a regulator, auditor, or affected individual might
ask — all the way to a concrete, computable metric that answers it?
That's not a lookup exercise. It's a reasoning exercise.
**The Scenario**
The real-world context:
A UK public sector organisation uses an AI system to handle queries from members of the
public. Users submit questions related to government services, policies, or support (e.g.,
housing, benefits, or local services), and the system generates responses to assist them.
This is the lens through which you should read every question in this assessment.

## What You’ve Been Given

You have been assigned a single pairing consisting of:
● **A compliance question** — the kind of question a regulator, an affected individual, or
an internal ethics board might ask about this system
● **An ethical dimension** — the governance lens through which the question should be
understood


Your task is to design **3-4 appropriate metrics** to answer the compliance question within the
given ethical dimension.
**RAG structure:**
Design a RAG system aligned with your dimension. You may define additional constraints
where necessary to reflect real-world considerations. Feel free to choose any data sources,
policies, or assumptions that support your design.
**Available Data & Access:**
You may assume access to the following:
● User queries
● LLM responses
● Model client (API access for inference; no direct access to model internals)
● Ground truth is available for some queries as human-provided responses; it is not
available for all queries.
● System Performance Metrics
**Important:** Do not assume access to any additional data beyond what is provided (e.g., no
RAG-specific logs).
**Ethical Dimension Compliance Question Candidate names
Bias & Fairness** Does the system systematically assign
different outcomes to individuals or groups
based on protected or sensitive
characteristics, thereby disadvantaging
them compared to others?
Anushka
**Model Performance** Does the system perform reliably and
accurately for its intended use, and are
there defined performance standards that
are consistently maintained over time?
Manav


**Explainability &
Transparency**
Can the organisation provide explanations
of model outputs that are consistent with
the input and internally coherent?
Krisha
**Data & Model Drift** How does the organisation detect and
respond to changes in data or model
behaviour over time to ensure the system
remains valid, unbiased, and fit for
purpose?
Lamaq
**Monitoring & Compliance** Does the organisation maintain continuous
monitoring, audit trails, and oversight
mechanisms to ensure the system
consistently meets regulatory and ethical
standards?
Yash
**Security & Adversarial
Robustness**
Does the AI system maintain consistent
and intended behaviour when exposed to
adversarial, manipulative, or malicious
inputs, without producing harmful or
incorrect outputs, based on observable
model behaviour?
Vedant
**Ethical Alignment / Value
Alignment**
Does the AI system generate outputs that
comply with defined organisational safety,
ethical, and usage policies, based on
observable model behaviour?
Krishil
**Human-AI Interaction** Does the AI system generate responses
that are clear, relevant, and appropriate for
the user’s request, and adhere to
expected communication?
Keyush


## What You Need To Do

For your assigned pairing, you must produce the following:

_1._ **Connect:**
    Explain how the ethical dimension frames the compliance question. What is the
    underlying concern, and how does it translate into something that can be measured?
_2._ **Contextualise:**
    Apply the ethical dimension and compliance question to the scenario. Be specific —
    what could go wrong when an AI system generates responses to public queries
    about government services? What would good look like in a well-functioning system?
_3._ **Design & Justify:**
    Propose 3–4 measurable metrics that could be used to answer the compliance
    question. For each:
       _○_ Define what it measures
       _○_ Describe how it would be computed
       _○_ Specify the data required
       _○_ Justify why it is an appropriate and defensible measure
       _○_ Highlight any limitations or risks
_4._ **Operationalise:**
    Explain how these metrics would be monitored in a production system. What
    thresholds or patterns would indicate an issue, and what actions should follow?
**An Important Note on Metrics**
Not everything labelled a "metric" in AI governance literature actually produces a single
number out of the box. Some produce distributions, vectors, plots, or ranked lists.
If your metric falls into this category, your task includes an additional step: figure out how you
would derive a scalar value from its output — a single number that could represent this
metric's contribution to an overall dimension-level score.
There is no single correct answer to this. What we are looking for is a justified, defensible
approach. Explain your reasoning.


### Format

Write your response in four sections, each covering Connect, Contextualise, Justify and
operationalise in whatever order or structure works best for your reasoning. There is no
prescribed template beyond this.
At the end of each metric, add a **confidence rating on a scale of 1–5:
Rating Meaning**
1 I'm guessing — I don't think this holds together
2 I see a connection, but I'm uncertain about my
reasoning
3 Reasonable attempt — I'm aware of some gaps
4 Solid reasoning — minor uncertainties remain
5 Fully confident — I could defend this in a client meeting
Be honest. Calibration matters as much as the answer itself.

## Ground Rules

**Time limit:**
Due 25 Feb 2026, 23:45(IST)
**Open book:**
You may use any publicly available resources — documentation, research papers,
frameworks, or technical references.
If you reference external materials (e.g., research papers), please include the source link.
**What not to do:**
Do not provide generic definitions of metrics or concepts. The focus is always on why a
particular metric is appropriate for the given compliance question in this specific context.
Answers that remain at a theoretical level without application will not score well.
You should derive metrics only from the data available (as specified above). Do not assume
access to any additional data.


**Metric Design Expectations:**
You are expected to define your own metrics. Simply naming standard metrics without
explanation will not be sufficient.
**Model Performance Dimension (if applicable):**
Please avoid using traditional classification metrics such as accuracy, precision, recall, or F
score.
Focus on evaluation approaches that are appropriate for the system and scenario.

#### What We're Looking For

We are not expecting perfect answers. We are looking for evidence that you can:
● Think in structured chains from a problem to a measurement
● Apply abstract governance concepts to a concrete, consequential scenario
● Recognise the limits of your own reasoning and be honest about them
● Communicate clearly and without unnecessary filler
If you find yourself uncertain, work through it openly in your response. Showing how you
reason through uncertainty is more valuable than a confident answer that doesn't hold up.
**Good luck. Take your time reading before you start.**


##### —------------------------------------------------------------


