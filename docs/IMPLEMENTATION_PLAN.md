# Implementation Plan: Production-Grade Brand AI Support Agent

## Executive Summary
This document establishes the detailed implementation plan for building an evidence-grounded, audit-ready AI customer support agent based on the Customer Support on Twitter dataset (`twcs.csv`).

The architecture follows a strict, inspectable engineering pipeline:
$$\text{DATA} \longrightarrow \text{PROFILING} \longrightarrow \text{RECONSTRUCTION} \longrightarrow \text{INTENT DISCOVERY} \longrightarrow \text{CALIBRATED CLASSIFICATION} \longrightarrow \text{EVIDENCE RETRIEVAL & QUALITY} \longrightarrow \text{POLICY & ESCALATION} \longrightarrow \text{GROUNDED REPLY} \longrightarrow \text{EVALUATION}$$

---

## Methodological Guardrails (20 Mandatory Rules)

1. **Brand Selection as an Empirical Outcome**: Candidate brands (`AmazonHelp`, `AppleSupport`, `SpotifyCares`, `Uber_Support`, `British_Airways`, `Delta`, `Tesco`) are profiled quantitatively via `scripts/run_brand_profiling.py`. `SpotifyCares` is treated as a provisional candidate pending complete profiling output; selection is justified strictly by data quality metrics.
2. **Data-Driven Intent Discovery**: Intent taxonomy is discovered via semantic clustering of customer queries. Intermediate cluster outputs (`data/interim/intent_clusters.json`) and consolidation rationales are preserved.
3. **Threshold Calibration on Validation Split**: Confidence and retrieval thresholds are not arbitrarily set. A development split (`data/val/dev_tuning.jsonl`) is used to tune operating points balancing false auto-handle rate, safe auto-handle coverage, and escalation rate.
4. **Frozen Evaluation**: All taxonomy definitions, classifier weights, prompts, retrieval indexes, thresholds, and policies are explicitly frozen prior to running the final gold evaluation. Zero modifications are permitted based on gold set feedback.
5. **Dual-Distribution Gold Set**: A hand-labelled evaluation set of ~200 examples is partitioned into:
   - **Stratified View**: balanced representation across all intents for per-class diagnostic metrics.
   - **Natural Distribution View**: preserves empirical frequency distribution for operational expectation.
   Macro metrics and natural-weighted metrics are reported separately.
6. **Multi-Layer Leakage Audit**: Evaluated customer messages and their conversation threads are quarantined. Leakage audit checks exact string duplicates, thread ID overlaps, temporal splits, and semantic near-duplicates ($>0.92$ embedding similarity). Artifact: `docs/LEAKAGE_AUDIT.md`.
7. **First-Class Reply Metrics**: Unsupported-claim rate is evaluated as a primary safety metric. Grounded-response rate, escalation rate, and safe auto-handle coverage are reported alongside semantic alignment. Generic text overlap (e.g. BLEU/ROUGE) is not treated as evidence of quality.
8. **Solution-Level Retrieval Relevance**: Evidence retrieval is evaluated on whether the retrieved historical interaction demonstrates a valid resolution path for the inquiry class, rather than requiring an exact 1:1 tweet match.
9. **Comprehensive LLM-as-Judge Validation**: The LLM judge is validated against human annotations on a 50-sample subset, reporting Spearman $\rho$, Pearson $r$, Exact Agreement %, Within-1 Score Agreement %, and Mean Absolute Difference across all 7 rubric dimensions.
10. **Strict Separation of Dual Generation Pipelines**:
    - **Pipeline A (Offline Deterministic)**: Template and evidence-grounded synthesis designed for zero-cost, 15-minute reproduction without API keys.
    - **Pipeline B (Generative LLM)**: Dynamic synthesis via external LLM APIs (OpenAI/Gemini/Ollama) with structured JSON enforcement.
    The offline pipeline is never presented as proof that the generative LLM functions.
11. **Contradictory Evidence Handling**: Retrieved historical resolutions are treated as historical precedent, not ground truth. When retrieved evidence contains conflicting or contradictory resolutions, the escalation policy triggers escalation.
12. **Evidence Quality Assessment Layer**: An explicit layer sits between vector retrieval and policy decision, evaluating semantic similarity, intent compatibility, resolution completeness, and cross-evidence consistency.
13. **Confidence Calibration**: The intent classifier provides calibrated probabilities via Platt scaling / temperature scaling. Evaluated via Expected Calibration Error (ECE), Brier score, and reliability diagrams.
14. **Escalation Trade-Off Metrics**: Safe auto-handle coverage, false auto-handle rate (critical safety violations), escalation rate, and decision precision/recall are reported.
15. **Engineering Targets over Arbitrary Gates**: Quantitative targets (e.g. Macro F1 > 0.80) are treated as engineering benchmarks, not pass/fail theater. All empirical failures are documented transparently.
16. **Edge-Case Regression Suite**: Explicitly evaluates: conflicting historical replies, weak evidence, high similarity with incorrect intent, novel questions, multi-intent inquiries, short/noisy queries, and outdated guidance.
17. **Data-Grounded Terminology**: Out-of-scope, ambiguous, adversarial, abusive, or safety-sensitive categories are used in place of informal "malicious" descriptions.
18. **Architectural Clarity**: The classifier architecture is explicitly defined: Sentence Transformer embeddings (`all-MiniLM-L6-v2`) fed into a Calibrated Multinomial Logistic Regression model, with prototype centroid distance serving as an inspectable geometric out-of-scope detector.
19. **Artifact Provenance**: All cached datasets, models, and indexes record SHA256 hashes, random seeds, configurations, and generation commands.
20. **Methodological Honesty**: `docs/HEADLINE_NUMBER_CAVEATS.md` outlines what the headline metrics prove, what they suggest, and what the evaluation cannot establish.

---

## Milestone Roadmap

### Phase 0 — Environment & Repository Setup
- Document environment in `docs/INITIAL_REPOSITORY_ASSESSMENT.md`.
- Set up `pyproject.toml`, standard package layout `src/hiver_agent`, and `.venv`.

### Phase 1 — Solution Architecture & Design Documentation
- Author `docs/ARCHITECTURE.md`, `docs/DATA_CARD.md`, `docs/EVALUATION_PLAN.md`, `docs/DECISION_LOG.md`.

### Phase 2 — Brand Profiling & Quantitative Selection
- Implement `src/hiver_agent/data/profiling.py` and `scripts/run_brand_profiling.py`.
- Quantitatively profile candidate brands.
- Author `docs/BRAND_SELECTION.md` documenting metrics and final brand choice.

### Phase 3 — Conversation Reconstruction & Provenance
- Implement `src/hiver_agent/data/reconstruction.py`.
- Construct customer-brand pairs with conversation thread tracking and provenance metadata.
- Implement automated thread structure tests.

### Phase 4 — Data-Driven Intent Discovery
- Implement `src/hiver_agent/nlp/clustering.py`.
- Run semantic clustering on customer queries; save intermediate outputs to `data/interim/intent_clusters.json`.
- Consolidate clusters into business taxonomy; author `docs/TAXONOMY.md`.

### Phase 5 — Calibrated Intent Classifier & Baselines
- Implement `src/hiver_agent/nlp/classifier.py` (Sentence Transformers + Calibrated Logistic Regression + Centroid distance).
- Implement Baseline 1 (Majority Intent) and Baseline 2 (TF-IDF + Logistic Regression).
- Compute calibration diagnostics (ECE, Brier score, reliability diagrams).

### Phase 6 — Evidence Retrieval & Quality Assessment
- Implement `src/hiver_agent/retrieval/vector_store.py` (cosine similarity vector index).
- Implement `src/hiver_agent/retrieval/evidence_quality.py` (solution-level relevance, contradiction detection, completeness).

### Phase 7 — Validation Tuning & System Freeze
- Create validation split `data/val/dev_tuning.jsonl`.
- Run `scripts/tune_thresholds.py` to tune confidence and quality thresholds on automation-vs-safety trade-off curve.
- **Freeze System**: lock all artifacts and generate `models/freeze_manifest.json`.

### Phase 8 — Grounded Generation & Dual Provider Interface
- Implement `src/hiver_agent/generation/provider.py`:
  - `DeterministicGroundedProvider` (offline reproducible synthesis).
  - `GenerativeLLMProvider` (live LLM API synthesis).
- Implement `src/hiver_agent/generation/hallucination_checker.py` for unsupported-claim detection.

### Phase 9 — Conservative Escalation Policy Engine
- Implement `src/hiver_agent/policy/escalation.py`.
- Implement rule checks: low calibrated confidence, weak evidence, contradictory resolutions, safety-sensitive intents, out-of-scope anomalies, ambiguity.
- Output structured decision with reason codes.

### Phase 10 — Gold Evaluation Set & Leakage Audit
- Create `data/gold/gold_messages.jsonl` (200 hand-labelled examples, stratified and natural distribution views).
- Implement `scripts/run_leakage_audit.py` (exact duplicates, thread overlap, semantic near-duplicates).
- Author `docs/LEAKAGE_AUDIT.md`, `docs/ANNOTATION_GUIDELINES.md`, and `docs/GOLD_SET_METHODOLOGY.md`.

### Phase 11 — Comprehensive Evaluation Harness
- Implement `src/hiver_agent/evaluation/harness.py`.
- Compute classification, retrieval, escalation, and generation metrics against frozen system and baselines.

### Phase 12 — LLM-as-Judge & Human Agreement Study
- Implement 7-dimension LLM judge in `src/hiver_agent/evaluation/judge.py`.
- Annotate 50 human validation examples.
- Compute Spearman $\rho$, Pearson $r$, exact agreement, within-1 agreement, and mean absolute difference. Document disagreements.

### Phase 13 — Failure Analysis & Edge-Case Suite
- Extract top 5 failure modes from evaluation results.
- Run dedicated historical edge-case tests.
- Author `docs/FAILURE_ANALYSIS.md`.

### Phase 14 — Headline Number Caveats
- Author `docs/HEADLINE_NUMBER_CAVEATS.md` detailing methodological limitations and operational caveats.

### Phase 15 — Final Reports & Decision Log
- Finalize `docs/DECISION_LOG.md` (12–15 entries).
- Author 6-page comprehensive report in `docs/FINAL_REPORT.md`.
- Maintain `docs/WALKTHROUGH.md`.

### Phase 16 — Testing, CLI & Reproducibility
- Build comprehensive pytest suite (`tests/unit/`, `tests/integration/`, `tests/fixtures/`).
- Build CLI entrypoints (`run_pipeline.py`, `evaluate.py`, `demo.py`).
- Author `README.md` enabling 15-minute headline reproduction.

### Phase 17 — Final Engineering Audit
- Perform repository audit for code quality, docstrings, type hints, secrets, and dead code.
- Produce `docs/FINAL_AUDIT.md`.
