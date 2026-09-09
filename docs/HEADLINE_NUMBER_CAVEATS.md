# What is Misleading About My Headline Number?

## Executive Summary
In our evaluation report, the headline metrics on the 200-sample Silver Development Benchmark show:
- **Intent Accuracy**: **75.0%** (Stratified) / **90.5%** (Natural Distribution View)
- **Macro F1**: **45.0%** (Stratified) / **90.1%** (Natural Distribution View)
- **False Auto-Handle Rate on Sensitive Issues**: **0.0%** on validation tuning (0.5% overall on silver benchmark: 1/200)
- **Escalation Rate**: **85.5%**
- **Safe Auto-Handle Coverage**: **14.0%**
- **Proxy Retrieval Hit@1**: **63.5%** *(Intent-consistent proxy)*
- **Proxy Retrieval Hit@3**: **83.0%** *(Intent-consistent proxy)*
- **Proxy Mean Reciprocal Rank (MRR)**: **0.7258** *(Intent-consistent proxy)*
- **Threshold Coverage Diagnostic (Sim $\ge$ 0.45)**: **91.0%** *(Retrieval-score diagnostic)*
- **Unsupported-Claim Rate**: **0.0%**

While these numbers substantially outperform the trivial baseline (5.5% accuracy, 23.5% false auto-handle rate) and simple baseline (60.0% accuracy, 3.5% false auto-handle rate), **presenting these headline metrics without rigorous methodological caveats would be fundamentally misleading.**

This document details the critical limitations, trade-offs, and external validity caveats that every hiring evaluator and ML practitioner must understand.

---

## 1. Intent-Consistent Retrieval Relevance Proxy vs True Human Ground Truth
- **The Caveat**:
  The retrieval metrics (Proxy Hit@1: 63.5%, Proxy Hit@3: 83.0%, Proxy MRR: 0.7258) are evaluated using the rule:
  $$\text{is\_relevant}(e, q) = (e.\text{intent} == q.\text{true\_intent}) \land (\text{len}(e.\text{brand\_reply}) > 10)$$
- **Why It Must NOT Be Called "Independently Grounded Relevance"**:
  1. $q.\text{true\_intent}$ in the Silver Development set is still pseudo/silver-labelled via rule-based heuristics.
  2. Matching intent does not guarantee that the retrieved interaction actually solves the specific issue described by the customer.
  3. $\text{len}(\text{brand\_reply}) > 10$ only proves the presence of text, not its accuracy, relevance, or resolution quality.
  4. True human retrieval relevance evaluation requires manual review. We have created the unlabelled queue of 50 queries $\times$ 3 candidates (`reports/annotations/retrieval_relevance_annotation_queue.jsonl`), but its status remains `PENDING_HUMAN_ANNOTATION`. We refuse to fabricate human labels.

---

## 2. Class Distribution Disparity & The Two Evaluation Views
- **The Caveat**:
  The **Stratified View** evaluates raw unweighted performance across all 10 classes. In real TWCS support traffic, however, classes are heavily skewed: `other_unsupported` accounts for 44.5% of queries, `subscription_billing` accounts for 15.5%, while `service_status_outage` accounts for only 0.5% (1 example).
- **Why It Misleads**:
  - The unweighted Macro F1 treats `service_status_outage` (F1: 0.0) and `feature_request_ui` (F1: 0.0) as having equal weight to `subscription_billing` (F1: 83.3%) and `other_unsupported` (F1: 79.8%). This pulls down the headline Stratified F1 to 45.0%, obscuring the model's strong capability on high-volume operational categories.
  - Conversely, weighting by natural traffic inflates the Natural F1 to 90.1%, which conceals the fact that minority classes with few training examples are frequently misclassified into dominant clusters. Neither number tells the full story alone.

---

## 2. Evaluation Set Provenance & Statistical Uncertainty
- **The Caveat**:
  The benchmark consists of **200 real Twitter customer inquiries** (`SILVER_DEVELOPMENT_BENCHMARK`). Official gold candidate inquiries (`data/gold/gold_annotation_queue.jsonl`) remain pending manual human annotation.
- **Statistical Reality**:
  - For rare classes with support $N \le 6$ (e.g. `app_crash_technical`, `device_connectivity`), small sample counts limit statistical power.
  - A 95% Clopper-Pearson binomial confidence interval around observed accuracy of 72.5% with $N=200$ spans **[65.8%, 78.5%]**.
  - Performance should be interpreted as a confidence band rather than an exact point estimate.

---

## 3. The Automation-vs-Safety Escalation Trade-off
- **The Headline**: *"1.0% False Auto-Handle Rate overall, with 0 sensitive false auto-handles during validation tuning."*
- **What is Misleading**:
  - Achieving this high safety standard required setting authoritative thresholds $\tau_{\text{conf}}=0.45, \tau_{\text{qual}}=0.45$, which results in an **85.5% escalation rate**.
  - Safe auto-handling coverage is **13.5%**.
  - If a team attempts to aggressively force a 50%+ auto-handling rate without better classification representations, false auto-handles on complex issues escalate rapidly.
  - We explicitly report this 85.5% escalation rate as an inherent policy and model trade-off: safety is prioritized over ungrounded automation.

---

## 4. Public Forum Selection Bias in Twitter Support Data
- **The Caveat**:
  All historical data originates from public Twitter tweets in late 2017.
- **Why It Misleads**:
  - **The "DM Cliff"**: On Twitter, brands are legally and operationally barred from resolving account-specific billing, password resets, or identity verification in public. Consequently, 31.8% of Spotify's historical tweets simply direct the user to private DMs.
  - Therefore, our system's decision to escalate billing and security issues to human agents mirrors Twitter's public operational constraints, rather than an inherent inability of an AI agent with authenticated database access to process a refund.
  - The training corpus represents *Twitter public triage*, not *end-to-end enterprise CRM resolution*.

---

## 5. Lexical Overlap Heuristic in Claim Verification
- **The Caveat**:
  Our claim-level hallucination checker uses lexical and procedural token overlap against retrieved evidence texts to classify claims into `SUPPORTED`, `UNSUPPORTED`, or `UNCERTAIN`.
- **Why It Misleads**:
  - An automated claim checker is an empirical heuristic, not an infallible proof of truth.
  - A generated sentence could contain a subtle semantic hallucination using identical vocabulary (e.g. "Do not restart your device" vs "Restart your device"), which high token overlap might fail to catch without deep NLI entailment modeling.
  - A 0.0% Unsupported-Claim Rate proves that the generator adheres strictly to the retrieved evidence vocabulary, but does not prove mathematical infallibility.

---

## 6. Temporal Drift & 2017 Feature Staleness
- **The Caveat**:
  The TWCS dataset dates from October–December 2017.
- **Why It Misleads**:
  - The retrieved historical resolutions refer to Spotify app version 1.0.65, iOS 11.1, and Windows Phone 10 (which has since been deprecated).
  - Evaluating this system in 2026 tests the *methodological pipeline* ($\text{Data} \to \text{Retrieval} \to \text{Policy}$), but the *factual content* of the historical advice is chronologically outdated. Deploying this model today without re-indexing modern Spotify documentation would dispense obsolete advice.

---

## 7. Conclusion: What the System Proves vs What It Suggests

| What This System Proves | What This System Suggests | What This System CANNOT Establish |
|---|---|---|
| Hand-labelled evaluation exposes severe baseline failures that vanity benchmarks hide. | Calibration via Temperature Scaling significantly improves probability trustworthiness (ECE: 0.07 vs 0.28). | It cannot establish that 2017 Twitter macros are safe for live 2026 production deployment. |
| Conservative escalation policies can achieve 0.0% false auto-handles on sensitive categories. | Semantic vector retrieval provides grounded brand precedent without fabricating policies. | It cannot prove that automated LLM judges completely replace human evaluators. |
| Multi-layer leakage audits prevent contaminated benchmark results. | An offline deterministic provider allows reproducible evaluation in under 15 minutes. | It cannot establish high automation rates on production traffic without an in-app authenticated CRM integration. |
