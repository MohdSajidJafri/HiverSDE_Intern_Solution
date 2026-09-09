# Gold Set Methodology & Dual-Distribution Reporting

## 1. Single Frozen Gold Dataset
To prevent dataset drift, exactly one frozen gold dataset (`data/gold/gold_messages.jsonl`) containing **200 hand-labelled examples** is used for all evaluations.

## 2. Dual Reporting Views
- **Stratified Diagnostic View**: Equal sample weighting across all 10 intents (20 per intent). Provides unskewed, statistically meaningful per-class metrics.
- **Natural Distribution View**: Importance-weighted metrics using empirical cluster frequencies from the 2017 Twitter dataset. Prevents misleading operational claims.
