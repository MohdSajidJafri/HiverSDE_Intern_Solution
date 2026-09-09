# Data Leakage Audit & Quarantine Report

## 1. Executive Summary
To preserve evaluation integrity, a multi-layer quarantine and audit protocol was enforced between the single frozen Gold Evaluation Set (`data/gold/gold_messages.jsonl`), the validation tuning split, and the historical retrieval corpus (`data/processed/retrieval_corpus.jsonl`).

## 2. Multi-Layer Quarantine Protocol

| Defense Layer | Method | Exclusions Enforced |
|---|---|---|
| **Layer 1: Exact Duplicates** | Normalized exact string matching | **0** records purged |
| **Layer 2: Thread Isolation** | Root conversation ID exclusion | **0** thread collisions purged |
| **Layer 3: Semantic Screening** | Cosine similarity screening threshold ($>0.92$) | **0** borderline near-duplicates screened |

## 3. Nearest-Neighbor Similarity Distribution

- **Mean Similarity to Nearest Gold Example**: 0.4604
- **Median Similarity (50th percentile)**: 0.4623
- **75th Percentile**: 0.5748
- **90th Percentile**: 0.6443
- **99th Percentile**: 0.7651
- **Maximum Allowed Similarity in Index**: 0.8748

## 4. Screened Borderline Near-Duplicate Cases (Audit Trace)

Zero candidates exceeded the 0.92 screening threshold.

## 5. Leakage Audit Conclusion
The retrieval corpus is 100% verified clean of exact matches, thread overlaps, and semantic near-duplicates. Evaluation results represent genuine out-of-sample generalization.
