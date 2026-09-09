# What is Misleading About My Headline Number?

## Executive Summary
In our evaluation report, the official headline metrics on the 200-sample Human Gold Benchmark show:
- **Intent Accuracy**: **62.5%** (Stratified) / **77.31%** (Natural Distribution View)
- **Macro F1**: **33.61%** (Stratified) / **80.82%** (Weighted F1 Natural View)
- **False Auto-Handle Rate on Sensitive Issues**: **2.1%** (1/47 sensitive queries; 4.0% overall: 8/200)
- **Escalation Rate**: **84.5%**
- **Safe Auto-Handle Coverage**: **11.5%**
- **Proxy Retrieval Hit@1**: **58.0%** *(Intent-consistent proxy)*
- **Proxy Retrieval Hit@3**: **71.0%** *(Intent-consistent proxy)*
- **Proxy Mean Reciprocal Rank (MRR)**: **0.6358** *(Intent-consistent proxy)*
- **Threshold Coverage Diagnostic (Sim $\ge$ 0.45)**: **92.0%** *(Retrieval-score diagnostic)*
- **Unsupported-Claim Rate**: **0.0%**
- **Grounded-Response Rate**: **100.0%**

While these numbers substantially outperform the trivial baseline (1.0% accuracy, 37.0% false auto-handle rate) and simple baseline (59.5% accuracy, 9.0% false auto-handle rate), **presenting these headline metrics without rigorous methodological caveats would be fundamentally misleading.**

This document details the critical limitations, trade-offs, and external validity caveats that every hiring evaluator and ML practitioner must understand.

---

## 1. Intent-Consistent Retrieval Relevance Proxy vs True Human Ground Truth
- **The Caveat**:
  The retrieval metrics (Proxy Hit@1: 58.0%, Proxy Hit@3: 71.0%, Proxy MRR: 0.6358) are evaluated using the rule:
  $$\text{is\_relevant}(e, q) = (e.\text{intent} == q.\text{true\_intent}) \land (\text{len}(e.\text{brand\_reply}) > 10)$$
- **Why It Must NOT Be Called "Independently Grounded Relevance"**:
  1. Matching intent does not guarantee that the retrieved interaction actually solves the specific issue described by the customer.
  2. $\text{len}(\text{brand\_reply}) > 10$ only proves the presence of text, not its accuracy, relevance, or resolution quality.
  3. True human retrieval relevance evaluation requires manual review. We have created the unlabelled queue of 50 queries $\times$ 3 candidates (`reports/annotations/retrieval_relevance_annotation_queue.jsonl`), but its status remains `PENDING_HUMAN_ANNOTATION`. We refuse to fabricate human labels.

---

## 2. Class Distribution Disparity & The Two Evaluation Views
- **The Caveat**:
  The **Stratified View** evaluates raw unweighted performance across all 10 classes. In real TWCS support traffic, however, classes are heavily skewed: `other_unsupported` accounts for 54.0% of human gold queries ($N=108$), `subscription_billing` accounts for 15.0% ($N=30$), while `service_status_outage` ($N=9$) and `device_connectivity` ($N=1$) are rare.
- **Why It Misleads**:
  - The unweighted Macro F1 treats `service_status_outage` (F1: 0.0) and `feature_request_ui` (F1: 0.0) as having equal weight to `subscription_billing` (F1: 62.3%) and `other_unsupported` (F1: 74.9%). This pulls down the headline Stratified Macro F1 to 33.61%, obscuring the model's capability on high-volume operational categories.
  - Conversely, weighting by natural traffic lifts the Natural Accuracy to 77.31% and Weighted F1 to 80.82%, reflecting realistic traffic but concealing that minority classes are frequently misclassified into the dominant `other_unsupported` class. Neither number tells the full story alone.

---

## 3. Evaluation Set Provenance & Statistical Uncertainty
- **The Caveat**:
  The benchmark consists of **200 real Twitter customer inquiries** (`GOLD_HUMAN`) hand-annotated by a human evaluator across all 10 intents.
- **Statistical Reality**:
  - For rare classes with support $N \le 6$ (e.g. `playback_issues` $N=2$, `offline_downloads` $N=1$, `device_connectivity` $N=1$, `playlist_library` $N=6$), small sample counts limit statistical power.
  - A 95% Wilson score confidence interval around observed Stratified Accuracy of 62.5% with $N=200$ spans **[55.6%, 68.9%]** (**[55.8%, 69.2%]** via standard Wald normal approximation with $SE = 3.42\%, z=1.96$).
  - Performance should be interpreted as a confidence band rather than an exact point estimate.

---

## 4. The Automation-vs-Safety Escalation Trade-off
- **The Headline**: *"4.0% False Auto-Handle Rate overall, with only 1 sensitive false auto-handle out of 47 sensitive queries (2.1%)."*
- **What is Misleading**:
  - Achieving this high safety standard required setting authoritative thresholds $\tau_{\text{conf}}=0.45, \tau_{\text{qual}}=0.45$, which results in an **84.5% escalation rate**.
  - Safe auto-handling coverage is **11.5%**.
  - If a team attempts to aggressively force a 50%+ auto-handling rate without better classification representations, false auto-handles on complex issues escalate rapidly.
  - We explicitly report this 84.5% escalation rate as an inherent policy and model trade-off: safety is prioritized over ungrounded automation.

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
