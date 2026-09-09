# Evidence-Grounded Brand AI Customer Support Agent (Research Prototype)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reliable, evidence-grounded, audit-ready AI customer support agent developed for **`@SpotifyCares`** using historical interactions from the Customer Support on Twitter dataset (`twcs.csv`).

> [!IMPORTANT]
> **Research Prototype Framing**:
> This system is an evidence-grounded research and evaluation prototype built on historical 2017 public Twitter support data. It demonstrates calibration, data-driven intent discovery, semantic evidence retrieval, conservative escalation, and claim verification. It is **not** intended for live commercial deployment without modern documentation re-indexing and authenticated CRM integration.

---

## ⏱️ Quickstart: 15-Minute Headline Reproduction Guide

Evaluators can reproduce all headline benchmark numbers on a standard developer laptop in **under 5 minutes** without requiring paid external API credits or downloading the complete 3M+ dataset.

### Step 1: Environment Setup
```powershell
# 1. Clone repository and navigate to workspace
git clone <repo-url>
cd HiverSolution

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate

# 3. Install dependencies in editable mode
pip install -e .
```

### Step 2: Run Full Test Suite (Unit + Integration + Edge Cases)
```powershell
pytest -v
```

### Step 3: Run Master Frozen Evaluation
Evaluates the primary system and two baselines against the **200-sample Silver Development Benchmark** (real `@SpotifyCares` inquiries from TWCS; gold human queue pending in `data/gold/gold_annotation_queue.jsonl`):
```powershell
python evaluate.py
```
*Output: Displays the headline comparison table and writes detailed JSON to `reports/results/evaluation_results.json`.*

### Step 4: Run Interactive Demo CLI
Test the complete pipeline on real customer inquiries:
```powershell
# Run showcase on representative customer queries:
python demo.py

# Or test a custom inquiry:
python demo.py --query "Desktop app crashes immediately on launch after update"
```

---

## 📊 Headline Benchmark Results

Evaluated on the frozen silver development evaluation benchmark ($N=200$ real customer inquiries from `@SpotifyCares`) against two operational baselines under identical frozen conditions:

| Metric | Baseline 1 (Trivial) | Baseline 2 (Simple) | Primary Agent (Frozen) |
|---|---|---|---|
| **Intent Accuracy (Stratified)** | 1.0% | 59.5% | **62.5%** |
| **Intent Macro F1 (Stratified)** | 0.2% | 17.1% | **33.6%** |
| **Intent Accuracy (Natural View)** | 0.0% | 85.4% | **77.3%** |
| **Intent Weighted F1 (Natural View)** | 0.0% | 84.8% | **80.8%** |
| **Expected Calibration Error (ECE)** | 0.9900 | 0.0989 | **0.1070** |
| **Brier Calibration Score** | 1.9800 | 0.5747 | **0.5212** |
| **Safe Auto-Handle Coverage** | 63.0% | 38.5% | **11.5%** |
| **False Auto-Handle Rate (CRITICAL)** | 37.0% | 9.0% | **4.0%** *(1 sensitive false auto)* |
| **Escalation Rate** | 0.0% | 52.5% | **84.5%** *(Conservative safety posture)* |
| **Proxy Retrieval Hit@1** | 0.0000 | 0.0000 | **0.5800** *(Intent-consistent proxy)* |
| **Proxy Retrieval Hit@3** | 0.0000 | 0.0000 | **0.7100** *(Intent-consistent proxy)* |
| **Proxy Mean Reciprocal Rank (MRR)** | 0.0000 | 0.0000 | **0.6358** *(Intent-consistent proxy)* |
| **Threshold Coverage Diagnostic (Sim $\ge$ 0.45)** | 0.0% | 27.0% | **92.0%** *(Retrieval-score diagnostic)* |
| **Unsupported-Claim Rate (Safety)** | 0.0% | 0.0% | **0.0%** *(Strict claim verification)* |
| **Grounded-Response Rate** | 100.0% | 100.0% | **100.0%** |

---

## 🏛️ System Architecture

```
[ Incoming Customer Tweet ]
             │
             ▼
[ Text Normalization & Cleaning ]
             │
             ▼
[ Sentence Embedding Generator ] (all-MiniLM-L6-v2)
             │
     ┌───────┴───────────────────────────────────────┐
     ▼                                               ▼
[ Calibrated Classifier ]                 [ Centroid Outlier Detector ]
(Multinomial Logistic Regression +         (Cosine distance to class
 Multiclass Temperature Scaling T=0.7911)   centroids; catches novelty anomalies)
     │                                               │
     └───────────────────────┬───────────────────────┘
                             │
                             ▼
            [ Semantic Evidence Retrieval ]
         (Vector store over 1,427 historical Spotify pairs)
                             │
                             ▼
            [ Evidence Quality & Contradiction Layer ]
         - Solution-level relevance check
         - 3-State Contradiction check:
           * Compatible historical advice
           * Incompatible historical advice
           * Insufficient evidence
                             │
                             ▼
            [ Conservative Escalation Policy ]
         Evaluates:
         - Confidence < 0.45
         - Evidence quality < 0.45
         - Incompatible contradictions
         - Sensitive account/billing domains
         - Novelty outliers
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   [ Decision: ESCALATE ]       [ Decision: AUTO_HANDLE ]
              │                             │
              │                             ▼
              │                 [ Grounded Reply Synthesis ]
              │                 (Dual Provider Interface)
              │                 - Provider A: Offline Deterministic
              │                 - Provider B: Live Generative API
              │                             │
              │                             ▼
              │                 [ Claim-Level Support Verifier ]
              │                 (Supported, Unsupported, Uncertain)
              │                             │
              └──────────────┬──────────────┘
                             │
                             ▼
                [ Structured JSON Payload ]
```

---

## 📂 Repository Structure

```
├── config.yaml                       # Externalized system configuration & thresholds
├── pyproject.toml                    # Package metadata & build definition
├── README.md                         # 15-minute reproduction guide & architecture
├── evaluate.py                       # Master frozen evaluation runner
├── demo.py                           # Interactive CLI demo
│
├── docs/                             # Engineering & architectural documentation
│   ├── INITIAL_REPOSITORY_ASSESSMENT.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── ARCHITECTURE.md
│   ├── DATA_CARD.md
│   ├── BRAND_SELECTION.md            # Empirical brand comparison & selection
│   ├── TAXONOMY.md                   # Data-driven intent taxonomy & cluster provenance
│   ├── ANNOTATION_GUIDELINES.md      # Human annotation guidelines
│   ├── GOLD_SET_METHODOLOGY.md       # Gold dataset methodology & dual views
│   ├── LEAKAGE_AUDIT.md              # Multi-layer quarantine report (>0.92 screening)
│   ├── EVALUATION_PLAN.md            # Comprehensive evaluation plan
│   ├── FAILURE_ANALYSIS.md           # Top 5 empirical failure modes with root causes
│   ├── HEADLINE_NUMBER_CAVEATS.md    # Mandatory "What is misleading about my headline?"
│   ├── DECISION_LOG.md               # 15 non-obvious engineering decisions
│   ├── FINAL_REPORT.md               # 6-page comprehensive hiring report
│   ├── WALKTHROUGH.md                # Step-by-step milestone walkthrough
│   └── FINAL_AUDIT.md                # Final engineering verification audit
│
├── src/hiver_agent/                  # Core package
│   ├── config.py                     # Pydantic configuration schemas
│   ├── data/
│   │   ├── ingestion.py              # Range streaming & dataset loading
│   │   ├── profiling.py              # Brand quality profiling & scoring
│   │   └── reconstruction.py         # Conversation thread reconstruction
│   ├── nlp/
│   │   ├── normalizer.py             # Twitter text cleaning & feature extraction
│   │   ├── clustering.py             # Unsupervised intent discovery (KMeans)
│   │   ├── classifier.py             # SentenceTransformer + Logistic Regression
│   │   └── calibration.py            # Multiclass Temperature Scaling & ECE
│   ├── retrieval/
│   │   ├── vector_store.py           # Semantic vector store (cosine similarity)
│   │   └── evidence_quality.py       # 3-state contradiction assessment
│   ├── policy/
│   │   └── escalation.py             # Conservative escalation policy
│   ├── generation/
│   │   ├── provider.py               # Dual generation provider interface
│   │   └── hallucination_checker.py  # Claim-level support checking
│   └── evaluation/
│       ├── harness.py                # Comprehensive metrics (ECE, Brier, F1)
│       ├── judge.py                  # LLM-as-a-Judge 7-dimension rubric
│       └── baselines.py              # Trivial & Simple baselines
│
├── scripts/                          # Reproducible CLI automation scripts
│   ├── run_brand_profiling.py        # Reproduces brand comparison
│   ├── run_intent_discovery.py       # Reproduces intent clustering & taxonomy
│   ├── build_gold_and_leakage_audit.py # Generates gold set & runs leakage audit
│   ├── train_classifier.py           # Trains & calibrates intent classifier
│   ├── tune_thresholds.py            # Tunes thresholds on validation split
│   └── evaluate_judge_agreement.py   # Runs human vs LLM judge agreement study
│
├── data/
│   ├── gold/gold_annotation_queue.jsonl # Quarantined human gold annotation queue (200 records)
│   ├── interim/silver_eval_set.jsonl    # Silver development benchmark (200 records, also gold_messages.jsonl)
│   ├── interim/unselected_multiturn_interactions.jsonl # Quarantined multi-turn turns (345 records: 190 Gold + 155 Silver)
│   ├── val/dev_tuning.jsonl             # Quarantined validation tuning split (156 records, 100 components)
│   ├── interim/intent_clusters.json     # Discovered cluster intermediate artifact
│   └── processed/retrieval_corpus.jsonl # 1,427 clean historical Spotify pairs (852 components)
│
├── models/
│   ├── freeze_manifest.json          # Cryptographic freeze manifest (hashes & config)
│   ├── intent_classifier.pkl         # Trained calibrated classifier
│   └── retrieval_index.pkl           # Persisted vector store index
│
└── reports/results/                  # Output JSON reports
    ├── evaluation_results.json
    ├── baseline_comparison.json
    ├── brand_profiling_results.json
    ├── threshold_tuning_results.json
    └── human_vs_judge_agreement.json
```

---

## 🔄 Cached Artifact Provenance & End-to-End Reproduction

To re-run the entire pipeline from scratch (including re-profiling, re-clustering, re-auditing, and re-training):

```powershell
# 1. Re-profile brands from raw TWCS slice and generate docs/BRAND_SELECTION.md
python scripts/run_brand_profiling.py

# 2. Re-run unsupervised clustering and generate docs/TAXONOMY.md
python scripts/run_intent_discovery.py

# 3. Re-build datasets and execute multi-layer leakage audit (>0.92 screening)
python scripts/build_gold_and_leakage_audit.py

# 4. Re-train and calibrate intent classifier
python scripts/train_classifier.py

# 5. Re-tune operating thresholds on validation split and update freeze manifest
python scripts/tune_thresholds.py

# 6. Re-evaluate Human vs LLM Judge agreement study
python scripts/evaluate_judge_agreement.py

# 7. Execute master evaluation
python evaluate.py
```

### Artifact Checksum Provenance (`models/freeze_manifest.json`)
Every cached artifact records cryptographic SHA256 hashes in `models/freeze_manifest.json`:
- `gold_messages.jsonl`: `e5ab60f0ff06c5c165701832531907607259b177...`
- `dev_tuning.jsonl`: `50d3d97a29c54cf23ce9695091524636a881bfffdc...`
- `retrieval_corpus.jsonl`: `743b68a1adfea183ccddab2f7af6b1a3140c10b6...`
- `retrieval_index.pkl`: `a0f9db01bf96da17b796862dc31bc3f61a4a304cbca...`

---

## 📚 References & Attribution

1. **Dataset**: *Customer Support on Twitter*, thoughtvector / Kaggle & SunidhiSriram / Hugging Face (`twcs.csv`).
2. **Embeddings**: *all-MiniLM-L6-v2*, Wang et al., 2020 (*MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*), Hugging Face / SentenceTransformers.
3. **Calibration**: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML 2017. (Temperature Scaling formulation).
4. **LLM-as-a-Judge**: Zheng et al., 2023. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023.
5. **Evaluation Methodology**: Ribeiro et al., 2020. *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. ACL 2020.
