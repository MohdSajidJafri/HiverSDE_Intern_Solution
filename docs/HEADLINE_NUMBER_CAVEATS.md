# What is Misleading About My Headline Number?

## Executive Summary
In our evaluation report, the headline metrics show:
- **Intent Accuracy**: **56.5%** (Stratified) / **62.5%** (Natural Distribution View)
- **Macro F1**: **54.6%** (Stratified) / **65.8%** (Natural Distribution View)
- **False Auto-Handle Rate on Sensitive Inquiries**: **0.0%**
- **Expected Calibration Error (ECE)**: **0.0724**
- **Unsupported-Claim Rate**: **0.0%**

While these numbers outperform the trivial baseline (10.0% accuracy, 26.5% false auto-handle rate) and the simple baseline (22.5% accuracy, 7.0% false auto-handle rate), **presenting these headline metrics without rigorous methodological caveats would be fundamentally misleading.**

This document details the critical limitations, trade-offs, and external validity caveats that every hiring evaluator and ML practitioner must understand.

---

## 1. Class Distribution Disparity & The Two Evaluation Views
- **The Caveat**:
  The **Stratified View** enforces an artificial 10% representation across all 10 classes (20 queries each). In contrast, real customer support traffic is heavily skewed: `playback_issues` and `app_crash_technical` account for ~35% of all traffic, while `feature_request_ui` and `service_status_outage` account for <5% each.
- **Why It Misleads**:
  - The unweighted Macro F1 treats `feature_request_ui` (F1: 0.0) as equally important to `playback_issues` (F1: 0.65). This pulls down the headline Stratified F1 to 54.6%, under-representing the system's true operational utility on high-volume issues.
  - Conversely, re-weighting by empirical traffic inflates the Natural F1 to 65.8%, which masks the fact that the system completely fails on certain minority categories (such as UI feature requests). Neither number tells the full story alone.

---

## 2. Gold Evaluation Set Size & Confidence Intervals
- **The Caveat**:
  The gold evaluation set consists of **200 hand-labelled queries** (20 per class).
- **Statistical Reality**:
  - For a class with $N=20$, a single misclassification changes the class recall by **5.0 percentage points**.
  - A 95% Clopper-Pearson binomial confidence interval around an observed accuracy of 56.5% with $N=200$ spans **[49.3%, 63.5%]**.
  - Claiming that the primary model is "56.5% accurate" implies a false level of precision. In reality, performance is bounded within a 14-point confidence interval.

---

## 3. The Automation-vs-Safety Escalation Trade-off
- **The Headline**: *"0.0% False Auto-Handle Rate on sensitive security and billing issues."*
- **What is Misleading**:
  - Achieving zero false auto-handles was accomplished by choosing a **strictly conservative operating posture**: the system escalated **98.0%** of the 200 gold test queries!
  - In an operational call center, escalating 98% of queries saves almost zero agent labor.
  - The gold test set was deliberately designed with a disproportionate density of edge cases, out-of-scope anomalies, and sensitive billing/security inquiries (~40% of the gold set).
  - While this proves that the guardrail mechanism successfully prevents catastrophic errors on dangerous queries, it must not be misinterpreted as demonstrating a highly automated production agent. Real automation coverage on clean, unambiguous technical traffic is ~30–40%, not 98%.

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
