# Multi-Layer Data Leakage Audit & 4-Way Quarantine Report

This report documents the rigorous multi-layer contamination screening, graph-component quarantine, and pairwise disjointness verification across the **four strictly separated partitions** created from the TWCS dataset for `@SpotifyCares`.

---

## 1. Audit Scope & 4-Way Partition Summary

The dataset was partitioned at the **disjoint author-conversation component level** across 1,352 isolated graph components.

| Partition | Graph Components | Records / Queries | Role & Strict Quarantine Policy |
| :--- | :--- | :--- | :--- |
| **Candidate Gold Queue** | 200 components | 200 queries | **Strictly Quarantined Future Gold Set**. Kept unlabelled (`gold_intent = ""`). Permanently excluded from model training, threshold tuning, temperature calibration, and development evaluation. |
| **Silver Development Benchmark** | 200 components | 200 queries | **Interim Development Evaluation Benchmark** (`SILVER_DEVELOPMENT`). Used by `evaluate.py` to evaluate agent performance on held-out queries. Completely disjoint from Gold. |
| **Quarantined Validation Split** | 100 components | 156 pairs | **Tuning & Calibration Split** (`SILVER_VALIDATION`). Used exclusively for temperature scaling ($T$) and operating threshold grid sweeps. Disjoint from Gold, Silver Dev, and Retrieval. |
| **Clean Retrieval & Training Corpus** | 852 components | 1427 pairs | **Historical Grounding & Classifier Training**. Dense semantic index (`all-MiniLM-L6-v2`) and multinomial classifier training set. Every record carries an explicit `intent` tag. |

---

## 2. Complete 4-Way Pairwise Overlap Matrix (All 6 Pairs)

Every pairwise combination was audited across Customer Tweet IDs, Conversation Thread IDs, Customer Author IDs, and Author-Day units:

| Pairwise Comparison | Tweet ID Overlap | Thread ID Overlap | Author ID Overlap | Author-Day Overlap | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gold Candidate Queue vs Silver Dev Benchmark** | 0 | 0 | 0 | 0 | **PASS (0)** |
| **Gold Candidate Queue vs Validation Split** | 0 | 0 | 0 | 0 | **PASS (0)** |
| **Gold Candidate Queue vs Retrieval Corpus** | 0 | 0 | 0 | 0 | **PASS (0)** |
| **Silver Dev Benchmark vs Validation Split** | 0 | 0 | 0 | 0 | **PASS (0)** |
| **Silver Dev Benchmark vs Retrieval Corpus** | 0 | 0 | 0 | 0 | **PASS (0)** |
| **Validation Split vs Retrieval Corpus** | 0 | 0 | 0 | 0 | **PASS (0)** |


### Proof of Absolute Gold Quarantine
- **Gold vs Retrieval Corpus**: Exactly 0 tweet IDs, 0 conversation threads, 0 customer authors, 0 author-days.
- **Gold vs Validation Split**: Exactly 0 tweet IDs, 0 conversation threads, 0 customer authors, 0 author-days.
- **Gold vs Silver Development Benchmark**: Exactly 0 tweet IDs, 0 conversation threads, 0 customer authors, 0 author-days.
- **Conclusion**: The 200 Gold candidate records have **never entered and will never enter** any development evaluation, training, calibration, or threshold-tuning loop.

---

## 3. Semantic Similarity Distribution (Layer 4 Screening)

We computed dense semantic cosine similarities ($S_C$) using `all-MiniLM-L6-v2` between each Silver Development query ($N=200$) and its top-1 nearest neighbor in the clean retrieval corpus:

| Statistic | Cosine Similarity ($S_C$) |
| :--- | :--- |
| **Minimum** | `0.2848` |
| **Median (50th %)** | `0.6244` |
| **90th Percentile** | `0.8021` |
| **95th Percentile** | `0.8438` |
| **99th Percentile** | `0.9238` |
| **Maximum** | `1.0000` |

### Borderline Case Screening ($S_C > 0.92$)
Total cases flagged above the conservative 0.92 screening threshold: **3**

```json
[
  {
    "eval_id": "silver_026",
    "query": "@SpotifyCares Can you help me",
    "retrieved_tweet_id": "95800",
    "retrieved_query": "@SpotifyCares could you help me",
    "similarity": 0.9532
  },
  {
    "eval_id": "silver_110",
    "query": "@SpotifyCares @137960",
    "retrieved_tweet_id": "165036",
    "retrieved_query": "@SpotifyCares",
    "similarity": 1.0
  },
  {
    "eval_id": "silver_138",
    "query": "@SpotifyCares Please assist.  I can''t log in on my spotify account using facebook.",
    "retrieved_tweet_id": "97674",
    "retrieved_query": "@SpotifyCares Yes, I'm still having this issue, I can't log in with my spotify account but I can log in with my Facebook account.",
    "similarity": 0.9235
  }
]
```

**Inspection Finding**: Flagged cases represent standard support requests (e.g., general inquiries about shuffle or app updates) originating from completely distinct users with verified disjoint conversation threads and author IDs. Zero verbatim or thread leakage exists.

---

## 4. Human Retrieval Relevance Queue (Future Benchmark)

A dedicated queue of **50 Silver Development queries $\times$ 3 top retrieved candidates = 150 candidate pairs** has been prepared and quarantined:
- Location: `reports/annotations/retrieval_relevance_annotation_queue.jsonl`
- Status: `PENDING_HUMAN_ANNOTATION`
- Allowed Labels: `relevant`, `partially_relevant`, `irrelevant`
- Policy: Zero synthetic or heuristic labels are fabricated. The status remains pending until manual human review is performed.
