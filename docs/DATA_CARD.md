# Data Card: Customer Support on Twitter (TWCS)

## 1. Dataset Overview
- **Dataset Name**: Customer Support on Twitter (`twcs.csv`)
- **Primary Source**: Kaggle (`thoughtvector/customer-support-on-twitter`)
- **Mirrored Repository**: Hugging Face (`SunidhiSriram/twcs`)
- **Total Volume**: ~2,811,774 tweets across multi-turn conversational threads.
- **Temporal Span**: Approximately October 2017 to December 2017.
- **Domain**: Real-world public customer service interactions between consumers and corporate brands on Twitter.

---

## 2. Dataset Schema & Field Descriptions

| Field Name | Type | Description |
|---|---|---|
| `tweet_id` | Integer | Unique identifier for the tweet. |
| `author_id` | String | Anonymized ID for consumers (e.g. `115712`) or brand screen name (e.g. `SpotifyCares`, `AppleSupport`, `AmazonHelp`). |
| `inbound` | Boolean | `True` if tweet originated from a customer to a company; `False` if authored by the brand/support agent. |
| `created_at` | String | Timestamp of tweet publication (Twitter format: `Day Mon DD HH:MM:SS +0000 YYYY`). |
| `text` | String | Full text of the tweet, including mentions, hashtags, and links. |
| `response_tweet_id` | String / Null | Comma-separated tweet IDs of subsequent responses to this tweet. |
| `in_response_to_tweet_id`| Integer / Null | The parent tweet ID to which this tweet responds directly. |

---

## 3. Brand Characteristics & Candidate Selection
The dataset includes over 100 brands spanning airlines, retail, telecom, tech hardware, and streaming services.

### Candidate Brands Profiled:
- `AmazonHelp`: High volume (>12,000 in sample), but high language diversity (Japanese, German, Spanish, English) and cross-domain dilution (AWS, retail, Prime Video, Kindle).
- `AppleSupport`: High volume (~4,800 in sample), but over 70% of responses are canned macro redirects to direct messages ("DM us your iOS version and device model").
- `SpotifyCares` (Provisional Candidate): Clean English support, domain-coherent (digital music streaming platform), actionable troubleshooting density (>65%), consistent brand tone, and clear escalation boundaries.
- `Uber_Support`: High volume, but predominantly short macro redirects to in-app help tickets.
- `British_Airways` / `Delta`: Airline booking, flight delays, lost luggage, high escalation necessity.

---

## 4. Conversation Reconstruction Protocol
Raw Twitter records are individual tweets, not structured dialogues. To reconstruct support interactions:
1. **Thread Inversion**: Follow `in_response_to_tweet_id` pointers to pair inbound customer inquiries with subsequent brand responses.
2. **Conversation Context**: Reconstruct multi-turn threads into a single chronological dialogue history:
   $$\text{Customer}_1 \longrightarrow \text{Brand}_1 \longrightarrow \text{Customer}_2 \longrightarrow \text{Brand}_2$$
3. **Provenance Tracking**: Every reconstructed record retains:
   - `conversation_id`: Root tweet ID of the thread.
   - `customer_tweet_id`: Tweet ID of the customer message.
   - `brand_tweet_id`: Tweet ID of the brand reply.
   - `created_at`: Original timestamp.

---

## 5. Cleaning & Normalization Rules
- **Handle Normalization**: Strip leading/trailing `@username` handles to focus models on semantic problem descriptions.
- **URL Standardization**: Retain URLs but normalize redirect shortlinks (`t.co`) to generic link tokens where appropriate.
- **Emoji & Punctuation**: Preserve emojis (critical for customer frustration/sentiment detection) and punctuation.
- **Encoding**: Enforce UTF-8 across all pipelines to handle non-ASCII characters without data corruption.

---

## 6. Leakage & Split Protocol
The dataset of 2,328 reconstructed `@SpotifyCares` interaction pairs is partitioned across 1,352 isolated connected components of the `(customer_author_id, conversation_id)` bipartite graph:

1. **Retrieval & Training Corpus (`data/processed/retrieval_corpus.jsonl`)**:
   - 1,754 interaction pairs from 1,052 disjoint components.
   - Built into dense semantic vector store index (`models/retrieval_index.pkl`).
   - Supervised classification training data: `data/processed/silver_training_data.jsonl` (1,754 records with `label_provenance: "taxonomy_rules_silver"`).

2. **Quarantined Development / Tuning Set (`data/val/dev_tuning.jsonl`)**:
   - 184 interaction pairs from 100 disjoint components strictly quarantined from the retrieval corpus.
   - Used exclusively for multiclass temperature scaling calibration ($T=0.7820$) and deterministic multi-objective threshold tuning ($\tau_{\text{conf}}=0.45, \tau_{\text{qual}}=0.45$).

3. **Gold Candidate Queue (`data/gold/gold_annotation_queue.jsonl` & `.csv`)**:
   - 200 real customer inquiries from 100 disjoint components strictly quarantined from retrieval and training.
   - Blank `gold_intent` and `ground_truth_decision`, awaiting human review.

4. **Silver Development Benchmark (`data/interim/silver_eval_set.jsonl` & `data/gold/gold_messages.jsonl`)**:
   - 200 real customer inquiries explicitly tagged `SILVER_DEVELOPMENT_BENCHMARK`.
   - Used for reproducible automated baseline evaluation without fraudulent claims of human annotation.

5. **Multi-Layer Leakage Audit (`docs/LEAKAGE_AUDIT.md`)**:
   - Tweet ID overlap: **0**
   - Thread ID overlap: **0**
   - Author-day overlap: **0**
   - Conservative near-duplicate semantic screening ($>0.92$ cosine similarity): audited and documented.

---

## 7. Known Limitations & Bias
- **Public Forum Bias**: Complex account-specific troubleshooting (e.g. passwords, payment details) is often redirected to private DMs in the original data, meaning public tweets frequently capture initial diagnosis rather than final resolution.
- **Temporal Drift**: The dataset represents late 2017 support practices. Product features and UI flows from 2017 may differ from current versions.
- **Sampling Bias**: Twitter users skew toward more urgent or vocal complaints compared to email or in-app support channels.
