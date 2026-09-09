# Project Walkthrough: Evidence-Grounded Brand AI Support Agent (Research Prototype)

This walkthrough documents the step-by-step implementation, empirical milestones, verification evidence, and key design choices of the project.

---

## Milestone 0: Environment Setup & Repository Assessment
- **Status**: Completed
- **Changes**:
  - Inspected repository runtime: Python 3.10 and 3.13, git 2.47.1, 182+ GB disk space on drive D.
  - Documented initial state in `docs/INITIAL_REPOSITORY_ASSESSMENT.md`.
  - Created standardized project layout: `src/hiver_agent/`, `tests/`, `scripts/`, `data/`, `models/`, `reports/`, `docs/`.
  - Initialized `pyproject.toml` and virtual environment `.venv`.

---

## Milestone 1: Solution Design & Technical Specifications
- **Status**: Completed
- **Changes**:
  - Authored foundational documentation incorporating all 20 methodological principles and 10 technical refinements:
    - `docs/IMPLEMENTATION_PLAN.md`
    - `docs/ARCHITECTURE.md`
    - `docs/DATA_CARD.md`
    - `docs/EVALUATION_PLAN.md`
    - `docs/DECISION_LOG.md` (15 decisions)
  - Locked core specifications:
    - Multiclass Temperature Scaling for confidence calibration.
    - Training intent classifier on real labelled examples; using centroid distances for novelty detection.
    - 3-state contradiction analysis in Evidence Quality Layer.
    - Claim-level support verification (`SUPPORTED`, `UNSUPPORTED`, `UNCERTAIN`).
    - Single frozen gold dataset with dual reporting views (Stratified vs Natural Traffic Weighting).
    - Conservative leakage screening with cosine similarity $>0.92$.
    - Comprehensive freeze manifest schema.
    - Research prototype framing.

---

## Milestone 2: Reproducible Brand Profiling & Selection
- **Status**: Completed
- **Script**: `scripts/run_brand_profiling.py`
- **Artifact**: `reports/results/brand_profiling_results.json`, `docs/BRAND_SELECTION.md`
- **Empirical Findings**:
  - Evaluated 7 candidate brands across 5 quantitative criteria: Interaction Volume, English Purity, Query Diversity, Concrete Troubleshooting Rate, and Low Canned-DM Rate.
  - `SpotifyCares` achieved rank 1 (Composite Score: 96.4/100) due to 19.9% concrete troubleshooting density (vs 6.0% for AppleSupport and 1.3% for AmazonHelp) and 99.1% English purity.
  - AppleSupport was penalized for 47.4% canned DM deflection rate; AmazonHelp for 6.8% multi-language dispersion.

---

## Milestone 3: Conversation Reconstruction & Ingestion
- **Status**: Completed
- **Module**: `src/hiver_agent/data/reconstruction.py`
- **Results**:
  - Successfully linked customer initial inquiries to historical brand resolution tweets using integer tweet IDs (`Int64`).
  - Extracted full conversation graphs while preserving provenance (`in_response_to_tweet_id`, timestamps, author handles).

---

## Milestone 4: Data-Driven Intent Discovery & Taxonomy
- **Status**: Completed
- **Script**: `scripts/run_intent_discovery.py`
- **Artifacts**: `data/interim/intent_clusters.json`, `docs/TAXONOMY.md`
- **Results**:
  - Clustered 2,328 customer queries into 12 clusters using TF-IDF + KMeans.
  - Consolidated clusters into 10 operational intents with clear boundaries and policy sensitivity tags:
    1. `subscription_billing` (SENSITIVE)
    2. `account_access_security` (SENSITIVE)
    3. `audio_playback_streaming`
    4. `offline_sync_downloads`
    5. `app_crash_freeze`
    6. `local_files_import`
    7. `playlist_library_management`
    8. `device_connectivity_bluetooth`
    9. `search_catalog_metadata`
    10. `social_sharing_lyrics`

---

## Milestone 5: Gold Set Construction & Multi-Layer Leakage Audit
- **Status**: Completed
- **Script**: `scripts/build_gold_and_leakage_audit.py`
- **Artifacts**: `data/gold/gold_messages.jsonl`, `data/val/dev_tuning.jsonl`, `data/processed/retrieval_corpus.jsonl`, `docs/LEAKAGE_AUDIT.md`
- **Results**:
  - Single frozen gold benchmark created ($N=200$, 20 per intent).
  - Validation split created ($N=60$) for threshold tuning.
  - Cleaned retrieval corpus of 2,268 historical Q&A pairs.
  - Multi-layer leakage audit executed:
    - Zero tweet ID overlaps (0/200).
    - Zero author-day overlaps (0/200).
    - Zero verbatim query duplicates (0/200).
    - Conservative cosine screening ($>0.92$): 5 borderline pairs flagged and manually inspected; confirmed as distinct historical user sessions with independent tweet IDs.

---

## Milestone 6: Intent Classifier Training & Temperature Scaling
- **Status**: Completed
- **Script**: `scripts/train_classifier.py`
- **Modules**: `src/hiver_agent/nlp/classifier.py`, `src/hiver_agent/nlp/calibration.py`
- **Results**:
  - Trained Multinomial Logistic Regression on dense embeddings (`all-MiniLM-L6-v2`, 384 dimensions).
  - Fitted multiclass temperature scaling parameter via L-BFGS ($T=0.96$).
  - Expected Calibration Error (ECE) reduced to 0.0724 (vs 0.2779 for uncalibrated baseline).
  - Computed 10 intent centroids for auxiliary novelty / out-of-scope detection.

---

## Milestone 7: Evidence Quality & Escalation Policy
- **Status**: Completed
- **Modules**: `src/hiver_agent/retrieval/evidence_quality.py`, `src/hiver_agent/policy/escalation.py`
- **Results**:
  - Implemented 3-state contradiction assessment (`COMPATIBLE`, `INCOMPATIBLE`, `INSUFFICIENT_EVIDENCE`).
  - Implemented rule-based safety policy escalating sensitive intents (`subscription_billing`, `account_access_security`), low intent confidence ($<0.55$), weak evidence ($<0.55$), centroid novelty ($>0.45$), or detected contradictions.

---

## Milestone 8: Threshold Tuning & Freeze Manifest
- **Status**: Completed
- **Script**: `scripts/tune_thresholds.py`
- **Artifact**: `reports/results/threshold_tuning_results.json`, `models/freeze_manifest.json`
- **Results**:
  - Evaluated 16 candidate operating thresholds on $N=60$ validation set.
  - Selected $\tau_{\text{conf}}=0.55, \tau_{\text{qual}}=0.55$: achieved 0.0% False Auto-Handle Rate on sensitive issues while retaining safe auto-handle capability.
  - Locked all asset SHA256 hashes, git commit SHA, and configurations in `models/freeze_manifest.json`.

---

## Milestone 9: Evaluation Benchmark & LLM Judge Validation
- **Status**: Completed
- **Scripts**: `evaluate.py`, `scripts/evaluate_judge_agreement.py`
- **Artifacts**: `reports/results/evaluation_results.json`, `reports/results/baseline_comparison.json`, `reports/results/human_vs_judge_agreement.json`
- **Benchmark Summary ($N=200$ Frozen Gold Set)**:
  | Metric | Primary Agent | Baseline 2 (Simple) | Baseline 1 (Trivial) |
  | :--- | :--- | :--- | :--- |
  | **Stratified Accuracy** | **56.5%** | 22.5% | 10.0% |
  | **Stratified Macro F1** | **55.0%** | 19.5% | 1.8% |
  | **Natural Weighted F1** | **65.8%** | 20.0% | 5.5% |
  | **ECE (Calibration Error)** | **0.0724** | 0.2779 | 0.9000 |
  | **False Auto-Handle Rate** | **0.0%** | 7.0% | 26.5% |
  | **Unsupported-Claim Rate** | **0.0%** | N/A | N/A |
  | **Escalation Coverage** | 98.0% | 5.0% | 0.0% |
- **Human vs LLM Judge Agreement ($N=50$)**:
  - Exact match agreement: **82.0%**
  - Within-1 score agreement: **100.0%**
  - Mean Absolute Difference (MAD): **0.1800**
  - Hallucination / Escalation correlation: **1.0000**

---

## Milestone 10: Test Suite & Repository Integration
- **Status**: Completed
- **Test Suite**:
  - 21 Unit Tests in `tests/unit/`
  - 8 Integration Tests in `tests/integration/`
  - **All 29 tests passed** (0 failures, 0 errors) in 96.47s.
- **Git State**:
  - Initialized git repository on branch `master`.
  - Remote origin set to: `https://github.com/MohdSajidJafri/HiverSDE_Intern_Solution.git`.
  - Working tree clean with all 73 files committed.

---

## Verification & Execution Commands
To reproduce all results offline in under 15 minutes:

```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Run full unit and integration test suite
pytest -v

# 3. Run frozen evaluation benchmark
python evaluate.py

# 4. Run interactive demonstration CLI
python demo.py
```

