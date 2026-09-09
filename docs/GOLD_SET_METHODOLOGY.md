# Gold Set Methodology & Dual-Distribution Reporting

## 1. Provenance and Integrity Policy
A primary commitment of this project is absolute methodological honesty:
- **Zero Fabricated Annotations**: We do not generate synthetic user queries or mock human agreement statistics.
- **Explicit Tier Distinctions**:
  - **Gold Tier (`GOLD_HUMAN`)**: Reserved strictly for records labelled by actual human annotators.
  - **Silver Tier (`SILVER_DEVELOPMENT_BENCHMARK`)**: Real TWCS customer queries paired with taxonomy-derived labels used for automated benchmark evaluation.
  - **Pending Status (`PENDING_HUMAN_ANNOTATION`)**: Annotation queues awaiting human completion.

---

## 2. Dataset Partition Architecture
The dataset of 2,328 reconstructed `@SpotifyCares` interaction pairs was partitioned by computing connected components on the `(customer_author_id, conversation_id)` bipartite graph:

1. **Gold Candidate Queue (`data/gold/gold_annotation_queue.jsonl` & `.csv`)**:
   - **Sample Size**: 200 real customer inquiries from 100 disjoint graph components.
   - **Status**: Quarantined with blank `gold_intent` and `ground_truth_decision`.
   - **Integrity**: Completely isolated from training and validation data.

2. **Quarantined Validation Split (`data/val/dev_tuning.jsonl`)**:
   - **Sample Size**: 184 interaction pairs from 100 disjoint graph components.
   - **Purpose**: Exclusively used for fitting multiclass temperature scaling ($T = 0.7820$) and deterministic multi-objective threshold sweeps ($\tau_{\text{conf}} = 0.45, \tau_{\text{qual}} = 0.45$).

3. **Retrieval & Training Corpus (`data/processed/retrieval_corpus.jsonl`)**:
   - **Sample Size**: 1,754 interaction pairs from 1,052 disjoint graph components.
   - **Purpose**: Semantic vector search knowledge base and supervised training examples (`silver_training_data.jsonl`).

4. **Multi-Layer Graph Leakage Audit (`docs/LEAKAGE_AUDIT.md`)**:
   - **Tweet ID Overlap**: **0**
   - **Thread ID Overlap**: **0**
   - **Author-Day Overlap**: **0**

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
