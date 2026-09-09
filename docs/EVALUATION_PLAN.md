# Evaluation Plan: Evidence-Grounded Brand AI Support Agent (Research Prototype)

## 1. Evaluation Philosophy & Research Framing
**Core Premise**: *"The proof is worth more than the system."*
This evaluation plan defines the methodology for auditing an evidence-grounded research prototype built on historical 2017 Twitter customer support data. It provides rigorous empirical validation while maintaining clear academic and engineering boundaries.

Key Methodological Commitments:
1. **Single Frozen Gold Ground Truth**: Hand-annotated by a human across multiple dimensions; zero model-derived synthetic ground truth.
2. **Strict System Freeze**: All model weights, temperature calibration parameters, retrieval indexes, prompts, and policy thresholds are frozen before gold evaluation.
3. **Dual Reporting Views**: A single frozen gold dataset evaluated under two separate views:
   - **Stratified View**: Equal weight across classes for per-intent diagnostic fidelity.
   - **Natural Distribution View**: Sample-weighted by empirical traffic frequency for realistic operational expectations.
4. **First-Class Safety Metrics**: Focuses on False Auto-Handle Rate, Unsupported-Claim Rate, and Contradiction Detection rather than superficial n-gram overlap.
5. **Multi-Metric Human-vs-Judge Calibration**: Evaluates LLM-as-a-Judge against human ratings across 5 distinct statistical agreement metrics.

---

## 2. Gold Evaluation Dataset (`data/gold/gold_messages.jsonl`)

### 2.1 Single Frozen Dataset Architecture
To ensure scientific integrity, exactly **one frozen gold dataset** containing ~200 hand-labelled customer queries is created.
The two evaluation perspectives are implemented as distinct reporting views over this identical set of examples:
- **Stratified Diagnostic View**:
  Treats all classes with equal sample weighting. Allows fair calculation of Macro Precision, Recall, and F1 across minority and majority classes alike.
- **Natural-Distribution Operational View**:
  Applies importance sampling weights $w_k = \frac{f_k^{\text{empirical}}}{f_k^{\text{gold}}}$ to re-weight performance metrics according to the true empirical distribution of customer inquiries in the 2017 brand corpus. Prevents over-optimistic or distorted operational claims.

### 2.2 Coverage and Edge-Case Representation
The gold set deliberately includes:
- Prototypical clear inquiries for each discovered intent.
- Extremely short/ambiguous queries ("help", "wont play").
- Multi-intent queries ("app crashed after update and I got billed twice").
- Safety-sensitive queries (account compromise, unauthorized transactions).
- Non-support / out-of-scope queries (banter, insults, general inquiries).
- Borderline queries designed to test centroid anomaly detection.
- Historical edge cases (queries matching conflicting historical resolutions).

### 2.3 Multi-Layer Leakage Audit & Conservative Screening
To guarantee that evaluation queries reflect genuine generalization:
1. **Thread Quarantine**: The complete multi-turn thread of every gold query is strictly excluded from the retrieval corpus.
2. **Exact Duplicate Removal**: Any historical query with identical normalized text is stripped from the retrieval index.
3. **Temporal Partitioning**: Evaluation queries are partitioned by timestamp where practical.
4. **Conservative Semantic Screening Threshold**:
   - Every gold query is embedded and compared against the entire retrieval corpus.
   - Any retrieval candidate exhibiting cosine similarity $> 0.92$ is flagged as a potential near-duplicate.
   - *Screening Principle*: A score $> 0.92$ is treated as a conservative screening threshold triggering manual inspection rather than automatic disqualification. If the historical query represents an identical customer situation with a canned resolution, it is purged from the retrieval corpus.
   - The nearest-neighbor similarity distribution is plotted and audited in `docs/LEAKAGE_AUDIT.md`.

---

## 3. Comparative Baselines

All systems are evaluated on the exact same single frozen gold set:

### Baseline 1: Trivial Baseline
- **Intent Classifier**: Majority class predictor (always predicts the most frequent empirical intent).
- **Retrieval Engine**: Returns the single most frequent historical brand response macro.
- **Escalation Policy**: Always predicts `AUTO_HANDLE` (100% automation, 0% escalation).

### Baseline 2: Simple Baseline
- **Intent Classifier**: TF-IDF (unigram + bigram) + Multinomial Logistic Regression (uncalibrated).
- **Retrieval Engine**: BM25 lexical keyword matching over historical customer queries.
- **Escalation Policy**: Simple keyword heuristic (escalates if keywords like "cancel", "hacked", "refund", "stolen" appear; otherwise auto-handles).

### Primary System: Full Grounded Agent
- Sentence Transformer embeddings (`all-MiniLM-L6-v2`) + Multinomial Logistic Regression calibrated via Multiclass Temperature Scaling + Centroid Outlier detector.
- Vector retrieval + Evidence Quality Layer (3-state contradiction detection).
- Tuned conservative escalation policy.
- Grounded reply synthesis with claim-level support verification.

---

## 4. Comprehensive Evaluation Metrics

### 4.1 Classification & Calibration Metrics
- **Accuracy**: Overall fraction of correct intent assignments.
- **Macro F1**: Unweighted mean of per-class F1 scores (primary metric for Stratified View).
- **Natural-Weighted F1**: Empirical traffic-weighted F1 (primary metric for Natural View).
- **Per-Intent Precision & Recall**: Detailed breakdown of class-level trade-offs.
- **Multiclass Expected Calibration Error (ECE)**:
  Measures confidence calibration across 10 probability bins:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
- **Multiclass Brier Score**: Mean squared deviation of predicted probability vectors from ground truth one-hot vectors.
- **Reliability Diagram**: Binned confidence vs empirical accuracy visual plot.

### 4.2 Retrieval & Evidence Quality Metrics
- **Solution-Level Hit@K ($K=3$)**: Fraction of queries where at least one retrieved historical dialogue provides an actionable, valid resolution procedure for that issue.
- **Mean Reciprocal Rank (MRR)**: Reciprocal rank of the first valid solution-level evidence candidate.
- **Contradiction Detection F1**: Evaluation of the system's ability to flag incompatible historical advice.

### 4.3 Escalation & Safety Trade-Off Metrics
- **Safe Auto-Handle Coverage**: Percentage of total queries automatically resolved without human intervention where the resolution was safe and grounded.
- **False Auto-Handle Rate (Safety-Critical)**: Percentage of queries that required escalation (e.g. account compromise, billing dispute, ungrounded issue) but were mistakenly auto-handled. **Target: 0.0% on sensitive classes.**
- **Escalation Rate**: Percentage of incoming queries routed to human agents.
- **Decision Precision & Recall**: Precision and recall for `ESCALATE` vs `AUTO_HANDLE` decisions against human ground truth.
- **Automation-vs-Safety Curve**: Trade-off curve plotting False Auto-Handle Rate vs Coverage across confidence threshold sweeps.

### 4.4 Generation & Claim-Level Grounding Metrics
- **Unsupported-Claim Rate (First-Class Metric)**: Percentage of drafted replies containing at least one unsupported factual claim or policy assertion.
- **Grounded-Response Rate**: Percentage of drafted replies where 100% of claims are classified as `SUPPORTED`.
- **Uncertain-Claim Rate**: Percentage of replies containing claims with ambiguous support (treated conservatively as escalation-worthy).
- **Semantic Alignment Score**: Embedding similarity between drafted reply and gold standard resolutions.

---

## 5. LLM-as-Judge & Human Agreement Study

### 5.1 Rubric Dimensions (1–5 Structured Likert Scale)
1. **Correctness**: Does the reply accurately address the customer's stated technical problem?
2. **Relevance**: Is the reply pertinent, focused, and free of extraneous noise?
3. **Grounding & Evidence Support**: Is the advice directly traceable to retrieved historical precedent?
4. **Completeness**: Does the reply provide actionable next steps, diagnostic questions, or clear escalation?
5. **Tone & Brand Voice**: Does the reply match the brand's empathetic, technical support tone?
6. **Unsupported Claims (Hallucination)**: Does the reply invent unverified policies, refunds, or actions?
7. **Escalation Appropriateness**: Was the decision to auto-handle or escalate correct given the evidence and issue severity?

### 5.2 Human Validation Agreement Protocol
- A representative subset of **50 gold evaluation cases** is independently evaluated by a human annotator using the identical 7-dimension rubric.
- The LLM-as-Judge evaluates the identical 50 cases using fixed prompts and temperature=0.
- **Statistical Agreement Metrics Reported**:
  - **Spearman Rank Correlation ($\rho$)**: Evaluates rank-order monotonic consistency.
  - **Pearson Linear Correlation ($r$)**: Evaluates linear score agreement.
  - **Exact Agreement %**: Percentage of identical scores on the 1–5 scale.
  - **Within-1 Score Agreement %**: Percentage of scores within $\pm 1$ point.
  - **Mean Absolute Difference (MAD)**: Average absolute score discrepancy.
  - **Documented Disagreements**: Detailed qualitative review of cases where $|S_{\text{human}} - S_{\text{judge}}| \ge 2$.

---

## 6. System Freeze Protocol & Freeze Manifest
Prior to running the final gold evaluation, all system components are permanently frozen:
- Manifest File: `models/freeze_manifest.json`
- Manifest Content:
  1. Git commit SHA
  2. Dataset source SHA256 (`twcs.csv`)
  3. Training and validation dataset hashes (`data/processed/*.jsonl`)
  4. Gold evaluation dataset hash (`data/gold/gold_messages.jsonl`)
  5. Python dependency environment hash
  6. Sentence Transformer model name and version
  7. Classifier weights artifact hash (`models/intent_classifier.pkl`)
  8. Calibrated Temperature parameter $T$
  9. Retrieval index artifact hash (`models/retrieval_index.faiss` / `.pkl`)
  10. Operating thresholds ($\tau_{\text{conf}}$, $\tau_{\text{quality}}$, $\delta_{\text{novelty}}$)
  11. LLM provider, model name, and prompt template version
  12. Escalation policy configuration hash
- **Evaluation Rule**: Once frozen, zero adjustments to prompts, weights, or thresholds are permitted. All empirical failures are analyzed in `docs/FAILURE_ANALYSIS.md`.
