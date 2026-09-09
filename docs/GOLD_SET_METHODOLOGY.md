# Gold Set Methodology & Dual-Distribution Reporting

## 1. Provenance and Integrity Policy
A primary commitment of this project is absolute methodological honesty:
- **Zero Fabricated Annotations**: We do not generate synthetic user queries or mock human agreement statistics.
- **Explicit Tier Distinctions**:
  - **Gold Tier (`GOLD_HUMAN`)**: Reserved strictly for records labelled by actual human annotators.
  - **Silver Tier (`SILVER_DEVELOPMENT_BENCHMARK`)**: Real TWCS customer queries paired with taxonomy-derived labels used for automated benchmark evaluation.
  - **Pending Status (`PENDING_HUMAN_ANNOTATION`)**: Annotation queues awaiting human completion.

---

## 2. Four-Way Quarantined Dataset Architecture
The dataset of 2,328 reconstructed `@SpotifyCares` interaction pairs was partitioned across 1,352 isolated connected components of the `(customer_author_id, conversation_id)` bipartite graph:

1. **Candidate Gold Queue (`data/gold/gold_annotation_queue.jsonl` & `.csv`)**:
   - **Sample Size**: 200 real customer inquiries from 200 disjoint graph components (components 1–200).
   - **Status**: Strictly quarantined with blank `gold_intent` and `ground_truth_decision`.
   - **Quarantine Policy**: Completely held out and permanently absent from model training, threshold tuning, temperature calibration, and development evaluation.

2. **Silver Development Benchmark (`data/interim/silver_eval_set.jsonl` & `data/gold/gold_messages.jsonl`)**:
   - **Sample Size**: 200 real customer inquiries from 200 separate disjoint graph components (components 201–400).
   - **Status**: Interim automated development benchmark (`SILVER_DEVELOPMENT`).
   - **Role**: Used by `evaluate.py` to evaluate agent performance on held-out queries. Completely disjoint from Gold.

3. **Quarantined Validation Split (`data/val/dev_tuning.jsonl`)**:
   - **Sample Size**: 156 interaction pairs from 100 disjoint graph components (components 401–500).
   - **Purpose**: Exclusively used for fitting multiclass temperature scaling ($T = 0.7911$) and deterministic multi-objective threshold sweeps ($\tau_{\text{conf}} = 0.45, \tau_{\text{qual}} = 0.45$).
   - **Calibration Provenance**: Explicitly documented as silver-label calibration on the validation split.

4. **Clean Retrieval & Training Corpus (`data/processed/retrieval_corpus.jsonl`)**:
   - **Sample Size**: 1,427 interaction pairs from 852 disjoint graph components (components 501–1,352).
   - **Purpose**: Semantic vector search knowledge base (`models/retrieval_index.pkl`) and supervised training examples (`silver_training_data.jsonl`).
   - **Metadata**: Every record carries an explicit `intent` tag for retrieval relevance proxy evaluation.

5. **Full 4-Way Overlap Matrix (`docs/LEAKAGE_AUDIT.md`)**:
   - **Tweet ID Overlap**: **0 across all 6 pairs**
   - **Thread ID Overlap**: **0 across all 6 pairs**
   - **Author ID Overlap**: **0 across all 6 pairs**
   - **Author-Day Overlap**: **0 across all 6 pairs**

### 2.1 Dataset Accounting & Arithmetic Reconciliation

$$\begin{aligned}
\text{Raw TWCS records scanned: } & 167,821 \\
\longrightarrow\quad \text{Reconstructed @SpotifyCares interaction pairs: } & \mathbf{2,328} \\
\longrightarrow\quad \text{Filtered/empty/invalid interactions: } & \mathbf{0} \\
\longrightarrow\quad \text{Eligible connected graph components: } & \mathbf{1,352} \\
\longrightarrow\quad \text{Quarantined Gold Candidate Queue: } & \mathbf{200} \\
\longrightarrow\quad \text{Silver Development Benchmark: } & \mathbf{200} \\
\longrightarrow\quad \text{Quarantined Validation Split: } & \mathbf{156} \\
\longrightarrow\quad \text{Clean Retrieval \& Training Corpus: } & \mathbf{1,427} \\
\longrightarrow\quad \text{Quarantined Multi-Turn Secondary Turns: } & \mathbf{345} \text{ (190 Gold + 155 Silver Dev)} \\
\hline
\text{Final Reconciled Total: } & \mathbf{2,328} \text{ (Exact Match)}
\end{aligned}$$

#### Audit Table of Partitions & Excluded Groups

| Partition / Group | Graph Components | Records / Pairs | Role in Pipeline | Cross-Split Overlap |
| :--- | :---: | :---: | :--- | :---: |
| **Gold Candidate Queue** | 200 | **200** | Quarantined primary inquiries for future human gold annotation | **0** |
| **Silver Dev Benchmark** | 200 | **200** | Quarantined primary inquiries for automated development evaluation | **0** |
| **Quarantined Validation Split** | 100 | **156** | Quarantined pairs for temperature calibration & threshold tuning | **0** |
| **Clean Retrieval Corpus** | 852 | **1,427** | Dense vector store & classifier training set | **0** |
| **Subtotal (Active Partitions)** | **1,352** | **1,983** | **Retained across the four operational partitions** | **0** |
| *Excluded Group 1: Gold Secondary Turns* | *(in Gold)* | **190** | Quarantined follow-ups in Gold threads (`data/interim/unselected_multiturn_interactions.jsonl`) | **0** (0 with Silver, Val, Ret) |
| *Excluded Group 2: Silver Secondary Turns* | *(in Silver)* | **155** | Quarantined follow-ups in Silver threads (`data/interim/unselected_multiturn_interactions.jsonl`) | **0** (0 with Gold, Val, Ret) |
| **Subtotal (Quarantined Exclusions)** | — | **345** | **Quarantined Secondary Multi-Turn Turns** | **0** |
| **Final Reconciled Total** | **1,352** | **2,328** | **Matches Reconstructed Interaction Count Exactly** | **0** |

$$\text{Reconciliation Formula: } 200 + 200 + 156 + 1,427 + 190 + 155 = \mathbf{2,328}$$

#### Accounting Explanation:
- **Evaluation Benchmark Design**: For single-turn evaluation queues, each benchmark item must represent an independent initial customer inquiry (`comp[0]`). Secondary customer turns within multi-turn dialogues (e.g., "thanks, that worked", "still having issues") cannot serve as primary inquiries.
- **Component Quarantine**: Because these 345 secondary turns belong to the threads and authors of the quarantined Gold and Silver components, placing them in retrieval or validation would cause severe conversational and author leakage. They are therefore quarantined within their respective component partitions and saved to `data/interim/unselected_multiturn_interactions.jsonl`.
- **Zero Leakage**: Cross-split overlap is verified to be exactly 0 across all splits.
- **Enforced via Regression Test**: `tests/unit/test_four_way_disjointness.py::test_exact_dataset_accounting_reconciliation`.

---

## 3. Retrieval Evaluation Hierarchy (Three Separate Measurements)

Rather than conflating retrieval similarity with relevance, the evaluation framework implements three separate measurements:

1. **Intent-Consistent Retrieval Relevance Proxy**:
   - Definition: $\text{is\_relevant}(e, q) = (e.\text{intent} == q.\text{true\_intent}) \land (\text{len}(e.\text{brand\_reply}) > 10)$
   - Reported Metrics: **Proxy Hit@1** (63.5%), **Proxy Hit@3** (83.0%), **Proxy MRR** (0.7258).
   - Honest Reporting Policy: Clearly reported as a proxy for retrieval consistency, NOT as human-grounded or human-validated relevance.

2. **Threshold Coverage Diagnostic**:
   - Definition: Top retrieved candidates meeting the configured similarity threshold ($\text{similarity} \ge 0.45$).
   - Reported Metric: **Top-1 Coverage** (91.0%), **Top-3 Coverage** (91.0%).
   - Honest Reporting Policy: Labeled strictly as a retrieval-score diagnostic, not as ground-truth semantic relevance.

3. **Human Retrieval Relevance (Future Benchmark)**:
   - Queue: `reports/annotations/retrieval_relevance_annotation_queue.jsonl` (50 Silver Dev queries $\times$ 3 retrieved candidates = 150 pairs).
   - Rubric: Human annotators evaluate candidates on a 3-level scale: `relevant`, `partially_relevant`, `irrelevant`.
   - Status: Kept strictly as `PENDING_HUMAN_ANNOTATION` with blank labels. No synthetic labels are fabricated.

---

## 3. Dual Reporting Views
To prevent distorted claims while preserving diagnostic fidelity, evaluation is reported under two complementary perspectives:

### 3.1 Stratified Diagnostic View (Unweighted)
- Evaluates raw macro precision, recall, and F1 across all 10 intents.
- Reveals model capability on challenging, low-frequency categories (e.g. `feature_request_ui`, `service_status_outage`) without allowing majority classes to dominate the metric.

### 3.2 Natural Distribution View (Empirical Frequency Weighted)
- Derives sample weights directly from the observed empirical frequencies in the dataset:
  - `playback_issues`: 8.5%
  - `app_crash_technical`: 3.0%
  - `offline_downloads`: 3.0%
  - `device_connectivity`: 3.0%
  - `playlist_library`: 8.5%
  - `subscription_billing`: 15.5%
  - `account_access_security`: 9.5%
  - `feature_request_ui`: 4.0%
  - `service_status_outage`: 0.5%
  - `other_unsupported`: 44.5%
- Reflects realistic production operational impact on real customer traffic.

---

## 4. Human Annotation Protocol
To transition the candidate queue to official Gold Ground Truth, annotators follow this protocol:

### Step 1: Open Annotation Queue
Open `data/gold/gold_annotation_queue.jsonl` or `data/gold/gold_annotation_queue.csv`.

### Step 2: Intent Label Assignment
Assign exactly one of the 10 operational intents:
1. `playback_issues`: Audio stops, stuttering, shuffle/repeat errors.
2. `app_crash_technical`: App crashes, won't launch, black screen, frozen UI.
3. `offline_downloads`: Downloaded songs greyed out, offline sync failing.
4. `device_connectivity`: Bluetooth, CarPlay, Chromecast, Amazon Echo streaming.
5. `playlist_library`: Lost playlists, missing songs, library sorting issues.
6. `subscription_billing`: Charges, refunds, payment method updates, student discounts.
7. `account_access_security`: Hacked accounts, unauthorized password/email changes.
8. `feature_request_ui`: UI layout feedback, requests for new features.
9. `service_status_outage`: Inquiries regarding widespread service downtime.
10. `other_unsupported`: Banter, social greetings, out-of-scope non-support inquiries.

### Step 3: Decision & Sensitivity Assignment
- `is_sensitive`: Set to `true` if inquiry involves financial billing or account security compromises.
- `ground_truth_decision`:
  - `ESCALATE`: Required for sensitive domains, ambiguous/contradictory issues, or novel out-of-scope queries.
  - `AUTO_HANDLE`: Allowed only for routine technical troubleshooting with verified brand precedent.

### Step 4: Finalize and Freeze
Save the file as `data/gold/gold_messages.jsonl` with `metadata.status = "GOLD_HUMAN"` and re-run `python evaluate.py`.
