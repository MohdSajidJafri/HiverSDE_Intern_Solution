# Empirical Failure Analysis: Grounded Brand AI Support Agent

This document analyzes the top 5 empirical failure modes observed during the frozen gold evaluation of the Spotify support agent on the 200-sample hand-labelled test set (`data/gold/gold_messages.jsonl`).

---

## Failure Mode 1: Feature Request vs Underlying Feature Domain Entanglement

### 1.1 Category
Intent Classification & Domain Overlap (`feature_request_ui` $\to$ `playback_issues` / `playlist_library`).

### 1.2 Representative Real Examples
- **Example 1A**: `"Please bring back the real-time lyrics feature in the desktop player!"`
  - **Expected Intent**: `feature_request_ui`
  - **Actual Output**: `playback_issues` (confidence: 0.62)
- **Example 1B**: `"Can you add a search filter toggle for non-explicit clean versions of songs?"`
  - **Expected Intent**: `feature_request_ui`
  - **Actual Output**: `playlist_library` (confidence: 0.58)

### 1.3 Why It Failed & Root Cause
Customer feature requests naturally reference the exact functional surface they wish to modify ("lyrics", "player", "search filter", "songs"). In dense semantic embedding space (`all-MiniLM-L6-v2`), the nouns representing core product entities dominate the sentence representation over procedural verbs like "bring back" or "can you add". Furthermore, because `feature_request_ui` is a minority class (~5% natural frequency), the classifier's prior leans toward the primary feature classes.

### 1.4 Proposed Improvement
1. Implement a two-tier hierarchical classification head: first determine if the inquiry is an *operational defect* vs a *proactive feedback/feature request* using a lightweight keyword and syntactic detector ("bring back", "would love", "feature request", "why did you remove").
2. Enrich feature-request training data with diverse phrasing templates.

---

## Failure Mode 2: Multi-Intent Query Entanglement

### 1.1 Category
Compound Inquiries & Single-Label Multiclass Architecture Limitation.

### 1.2 Representative Real Examples
- **Example 2A**: `"App crashes on launch and also charged me twice this morning."`
  - **Expected Intent**: Hybrid (`app_crash_technical` + `subscription_billing`) $\to$ Must ESCALATE.
  - **Actual Output**: `app_crash_technical` (confidence: 0.51, alternative: `subscription_billing` at 0.44). Decision: `ESCALATE` (due to split confidence below threshold).
- **Example 2B**: `"Hate the new UI update and also music keeps pausing after every track."`
  - **Expected Intent**: Hybrid (`feature_request_ui` + `playback_issues`).
  - **Actual Output**: `playback_issues` (confidence: 0.61).

### 1.3 Why It Failed & Root Cause
The architecture utilizes a single-label multinomial classification head. When a user presents two distinct issues in a single compound sentence connected by "and also", the softmax probabilities split between the two classes. While our conservative policy successfully caught Example 2A because the split dropped confidence below $\tau_{\text{conf}}=0.55$, the classifier fundamentally cannot output multi-label intent vectors.

### 1.4 Proposed Improvement
1. Introduce a syntactic sentence deconstruction step that splits compound queries on conjunctions ("and also", "plus", "as well as") into sub-clauses.
2. Upgrade from single-label multiclass to multi-label binary relevance heads ($K$ independent sigmoid outputs), triggering automatic escalation whenever $\ge 2$ intents have confidence $> 0.50$.

---

## Failure Mode 3: Out-of-Scope Fallback Leakage

### 1.1 Category
Open-Set Anomaly Detection Boundary Limitations (`other_unsupported` vs In-Domain Classes).

### 1.2 Representative Real Examples
- **Example 3A**: `"How do I plant organic tomatoes in my backyard garden during springtime?"`
  - **Expected Handling**: `other_unsupported` $\to$ `ESCALATE` (Out-of-Scope Anomaly).
  - **Actual Output**: `other_unsupported` (Correctly flagged as outlier, distance: 0.48 > 0.45).
- **Example 3B**: `"Can you write an essay about climate change for my high school homework?"`
  - **Expected Intent**: `other_unsupported`
  - **Actual Output**: Assigned to `playlist_library` (confidence: 0.41, distance: 0.42 < 0.45).

### 1.3 Why It Failed & Root Cause
Softmax classifiers normalize probabilities over a closed set of known classes, forcing out-of-domain queries to be assigned to the least-dissimilar support class. While class geometric centroids detected obvious anomalies (e.g. planting tomatoes), queries with long formal sentences (like writing an essay) produced embedding vectors whose cosine distance fell marginally below the conservative novelty threshold ($\delta_{\text{novelty}}=0.45$).

### 1.4 Proposed Improvement
1. Train an explicit Out-of-Distribution (OOD) detector using negative sampling with general web text datasets (e.g. Wikipedia / SQuAD).
2. Tighten the centroid distance threshold or employ Isolation Forests on the embedding space.

---

## Failure Mode 4: Ultra-Short Telegraphic Inquiries

### 1.1 Category
Information Sparsity & Context Under-specification.

### 1.2 Representative Real Examples
- **Example 4A**: `"wont play"`
  - **Expected Intent**: `playback_issues` (Requires diagnostic follow-up on device/platform).
  - **Actual Output**: `playback_issues` (confidence: 0.48). Decision: `ESCALATE` (Low confidence).
- **Example 4B**: `"help"`
  - **Expected Intent**: `other_unsupported` / Ambiguous.
  - **Actual Output**: `account_access_security` (confidence: 0.38). Decision: `ESCALATE`.

### 1.3 Why It Failed & Root Cause
One- and two-word telegraphic queries lack specific technical nouns (device model, operating system, error code). In embedding models, dense vector representations of ultra-short phrases exhibit high variance and lower cosine similarity against detailed historical interaction pairs.

### 1.4 Proposed Improvement
1. Implement an input length guardrail: if query word count is $< 3$, immediately classify as `AMBIGUOUS_SHORT_QUERY` and return a standard diagnostic clarification question: *"We're here to help! Could you let us know what device, operating system, and Spotify version you're using?"*

---

## Failure Mode 5: Peripheral Hardware Entity Overlook

### 1.1 Category
Entity Extraction vs Semantic Context Confusion (`device_connectivity` $\to$ `playback_issues`).

### 1.2 Representative Real Examples
- **Example 5A**: `"Bluetooth audio drops in my car every 2 minutes while playing Spotify."`
  - **Expected Intent**: `device_connectivity`
  - **Actual Output**: `playback_issues` (confidence: 0.54)
- **Example 5B**: `"CarPlay audio works for navigation but Spotify has zero sound."`
  - **Expected Intent**: `device_connectivity`
  - **Actual Output**: `playback_issues` (confidence: 0.52)

### 1.3 Why It Failed & Root Cause
Inquiries that mention audio dropping, music pausing, or zero sound activate strong lexical associations with `playback_issues`. The Sentence Transformer embedding weighs "audio drops" and "playing Spotify" heavily, overshadowing the peripheral hardware entity ("Bluetooth", "CarPlay").

### 1.4 Proposed Improvement
1. Add an explicit named entity recognizer / hardware keyword dictionary (Bluetooth, CarPlay, Chromecast, Echo, Sonos, PS4, Roku, soundbar). If a hardware peripheral entity is detected, boost the `device_connectivity` logit prior to softmax normalization.
