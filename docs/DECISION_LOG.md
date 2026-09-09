# Decision Log: Evidence-Grounded Brand AI Support Agent (Research Prototype)

This log records 15 non-obvious engineering and architectural decisions made during system design, implementation, and evaluation. Each entry details the context, alternatives considered, chosen approach, rationale, and operational trade-offs.

---

### Decision 1: Empirical Quantitative Profiling for Brand Selection over Arbitrary Choice
- **Context**: The Twitter Customer Support dataset contains dozens of brands with varying conversational volume, quality, and domain focus.
- **Alternatives Considered**:
  1. Arbitrarily declare a brand (e.g. AmazonHelp or AppleSupport) without empirical profiling.
  2. Treat candidate brands (including provisional candidates like SpotifyCares) neutrally, executing an empirical profiling script (`run_brand_profiling.py`) across top candidates before declaring the selection.
- **Chosen Option**: Alternative 2.
- **Rationale**: Prior testing indicated that `AmazonHelp` contains heavy multilingual noise (Japanese, German, Spanish) and sprawling subdomains, while `AppleSupport` relies predominantly on canned DM redirects (>70%). Profiling allows a defensible, reproducible choice based on measurable support evidence density.
- **Trade-off**: Requires building and executing an analytical profiling pipeline first, rather than hardcoding assumptions into early code.

---

### Decision 2: Data-Driven Semantic Clustering for Intent Discovery over Off-the-Shelf Taxonomies
- **Context**: Categorizing customer inquiries requires an intent taxonomy.
- **Alternatives Considered**:
  1. Adopt a generic customer support taxonomy (or BANKING77) directly.
  2. Cluster real customer tweets from the selected brand to discover empirical themes, saving intermediate cluster outputs (`data/interim/intent_clusters.json`) and consolidating them into a business-oriented taxonomy with documented provenance.
- **Chosen Option**: Alternative 2.
- **Rationale**: Generic taxonomies fail to capture brand-specific operational realities (e.g., audio stutter, offline sync, playlist restoration, student discount verification).
- **Trade-off**: Requires cluster inspection, intermediate artifact preservation, and explicit consolidation documentation in `docs/TAXONOMY.md`.

---

### Decision 3: Sentence Transformer Embeddings + Multinomial Logistic Regression Calibrated via Temperature Scaling
- **Context**: The intent classifier must produce calibrated probabilities to drive threshold-based escalation safely.
- **Alternatives Considered**:
  1. Prompt an LLM zero-shot/few-shot to emit intent labels.
  2. Train Multinomial Logistic Regression on one centroid embedding per intent.
  3. Train Multinomial Logistic Regression on diverse labelled development examples represented by Sentence Transformer embeddings (`all-MiniLM-L6-v2`), calibrated via Multiclass Temperature Scaling ($T > 0$).
- **Chosen Option**: Alternative 3.
- **Rationale**: Training on diverse labelled examples captures class variance and intra-class feature distribution that single centroids erase. Temperature scaling optimizes a single scalar parameter $T$ via negative log-likelihood on validation logits, preserving the argmax classification while aligning confidence with empirical accuracy.
- **Trade-off**: Requires feature extraction and supervised fitting on a labelled development split rather than zero-shot prompt engineering.

---

### Decision 4: Geometric Centroid Distance as an Auxiliary Out-of-Scope / Novelty Detector
- **Context**: Softmax classifiers are prone to overconfidence when presented with out-of-distribution or novel queries.
- **Alternatives Considered**:
  1. Rely exclusively on calibrated softmax confidence.
  2. Separately compute the cosine distance to the nearest intent geometric centroid in embedding space to detect out-of-scope or novel anomalies.
- **Chosen Option**: Alternative 2.
- **Rationale**: Softmax probabilities must sum to 1.0, forcing unrelated inputs (e.g. asking about gardening) to be assigned to the least-dissimilar support intent with potentially moderate probability. Geometric distance detects inputs far from all known support clusters.
- **Trade-off**: Requires storing and computing distances to class centroids at inference time.

---

### Decision 5: Solution-Level Evidence Retrieval over Exact 1:1 Tweet Matching
- **Context**: Retrieval quality must be evaluated to ensure historical conversations provide actionable grounding.
- **Alternatives Considered**:
  1. Define retrieval success strictly as retrieving the exact historical tweet that followed the customer message in the original dataset.
  2. Define retrieval success as retrieving any historical brand interaction that demonstrates an actionable, valid resolution for that problem category (solution-level relevance).
- **Chosen Option**: Alternative 2.
- **Rationale**: In real customer support, multiple historical threads address identical common issues (e.g., clearing app cache or resetting network settings). Rejecting a valid resolution because its tweet ID differs from the gold source is methodologically flawed.
- **Trade-off**: Requires evaluating semantic and procedural relevance rather than binary ID matching.

---

### Decision 6: Dedicated Evidence Quality Layer with 3-State Contradiction Analysis
- **Context**: High embedding similarity can retrieve historical tweets that offer conflicting or bad guidance.
- **Alternatives Considered**:
  1. Pass the top vector search results directly to the reply generator and escalation engine.
  2. Implement an Evidence Quality layer that distinguishes: (a) compatible historical advice, (b) genuinely incompatible advice, and (c) insufficient evidence, triggering escalation when incompatible resolutions are detected.
- **Chosen Option**: Alternative 2.
- **Rationale**: Historical support data is noisy. Brands occasionally provide conflicting advice across different agents or time periods (e.g., "reinstall app" vs "known server outage, do not reinstall"). Detecting contradiction is an essential safety mechanism for automated agents.
- **Trade-off**: Automated contradiction detection is inherently an empirical approximation; conservative thresholds must be applied.

---

### Decision 7: Threshold Tuning on Validation Split over Hardcoded Heuristics
- **Context**: Operating thresholds for intent confidence ($\tau_{\text{conf}}$) and retrieval quality ($\tau_{\text{quality}}$) dictate the escalation operating point.
- **Alternatives Considered**:
  1. Hardcode arbitrary thresholds (e.g., 0.65 confidence, 0.70 similarity).
  2. Tune thresholds on a dedicated validation split (`data/val/dev_tuning.jsonl`) to optimize an explicit objective balancing false auto-handle rate, safe auto-handle coverage, and escalation rate.
- **Chosen Option**: Alternative 2.
- **Rationale**: Arbitrary thresholds lack empirical justification. Validation tuning allows finding an optimal operating point on the automation-vs-safety trade-off curve and reporting the exact tradeoff.
- **Trade-off**: Requires maintaining a separate validation split distinct from the final gold set.

---

### Decision 8: Comprehensive Freeze Manifest Prior to Gold Evaluation
- **Context**: Iterative tweaking of prompts or thresholds on test data leads to p-hacking and inflated benchmark claims.
- **Alternatives Considered**:
  1. Iteratively tune the system against the gold evaluation set.
  2. Implement an explicit freeze stage that locks all parameters and logs a comprehensive audit manifest (`models/freeze_manifest.json`) prior to running gold evaluation.
- **Chosen Option**: Alternative 2.
- **Rationale**: Scientific rigor requires that evaluation data is evaluated only once on a frozen system. The freeze manifest records git commit SHA, dataset hashes, model weights hash, temperature $T$, retrieval index hash, prompt version, and threshold configuration.
- **Trade-off**: Errors discovered during gold evaluation cannot be quietly patched; they must be reported honestly in `docs/FAILURE_ANALYSIS.md`.

---

### Decision 9: Single Frozen Gold Dataset with Dual Reporting Views
- **Context**: Evaluation metrics can be skewed by class balance or real-world traffic distributions.
- **Alternatives Considered**:
  1. Create two separate gold datasets (one balanced, one natural).
  2. Create exactly one frozen gold dataset (~200 hand-labelled examples) and evaluate it through two distinct reporting views: Stratified View (unweighted, for per-class diagnostic depth) and Natural Distribution View (sample-weighted by empirical traffic frequency).
- **Chosen Option**: Alternative 2.
- **Rationale**: Maintaining a single frozen dataset eliminates dataset divergence while providing both balanced diagnostic metrics and realistic operational projections.
- **Trade-off**: Requires computing both unweighted and importance-weighted metrics during evaluation runs.

---

### Decision 10: Multi-Layer Leakage Audit with Conservative Screening Threshold
- **Context**: Retrieval-augmented systems can leak test examples into the retrieval corpus.
- **Alternatives Considered**:
  1. Perform simple random splitting.
  2. Enforce multi-layer leakage defense: exact string matching, full conversation thread isolation, temporal boundaries, and a conservative semantic screening threshold ($>0.92$ cosine similarity) with manual audit of borderline cases.
- **Chosen Option**: Alternative 2.
- **Rationale**: In Twitter data, users frequently retweet or repeat identical complaints, or continue multi-turn threads. Treating $>0.92$ similarity as a screening filter allows auditing borderline near-duplicates to ensure the retrieval corpus does not contain leaked answers.
- **Trade-off**: Requires running pairwise similarity audits across the entire retrieval corpus during dataset preparation.

---

### Decision 11: Dual Generation Architecture (Offline Deterministic vs Generative LLM)
- **Context**: Evaluators need to reproduce headline results in under 15 minutes without mandatory API keys, but the system must also support state-of-the-art generative LLMs.
- **Alternatives Considered**:
  1. Require an external LLM API key for all evaluation.
  2. Build only a deterministic template generator and claim it represents an LLM agent.
  3. Implement a dual-provider interface: an offline deterministic provider for zero-cost reproduction and CI testing, and a generative LLM provider for live API execution, with clear architectural separation.
- **Chosen Option**: Alternative 3.
- **Rationale**: Solves the 15-minute evaluation constraint while maintaining full generative LLM capabilities. Transparently documents that deterministic execution is for reproduction and testing, not proof of LLM generation.
- **Trade-off**: Requires maintaining and testing two provider implementations under the same interface.

---

### Decision 12: Claim-Level Support Verification (`SUPPORTED`, `UNSUPPORTED`, `UNCERTAIN`)
- **Context**: In customer support, an ungrounded promise or hallucinated policy can cause severe operational damage.
- **Alternatives Considered**:
  1. Rely solely on token overlap or BLEU/ROUGE metrics.
  2. Implement claim/sentence-level support checking, categorizing statements into `SUPPORTED`, `UNSUPPORTED`, or `UNCERTAIN`, treating `UNCERTAIN` conservatively.
- **Chosen Option**: Alternative 2.
- **Rationale**: Token overlap metrics cannot distinguish between a harmless filler phrase and an invented refund amount. Sentence-level claim verification directly flags ungrounded assertions. Treating uncertain claims conservatively prevents dangerous auto-handling.
- **Trade-off**: Acknowledged as an empirical heuristic, not an infallible hallucination detector.

---

### Decision 13: Multi-Metric Human-vs-Judge Agreement Analysis
- **Context**: LLM-as-a-Judge must be proven reliable before its evaluation scores can be trusted.
- **Alternatives Considered**:
  1. Report only a single Pearson correlation between human and LLM judge.
  2. Report Spearman $\rho$, Pearson $r$, exact agreement %, within-1 score agreement %, and mean absolute difference across all 7 rubric dimensions, along with qualitative analysis of disagreements.
- **Chosen Option**: Alternative 2.
- **Rationale**: Multi-metric evaluation provides an honest, granular view of judge calibration and detects systematic score bias.
- **Trade-off**: Requires detailed human scoring of a 50-sample slice and multi-variable statistical reporting.

---

### Decision 14: Engineering Targets over Arbitrary Pass/Fail Gates
- **Context**: Evaluating complex NLP pipelines against arbitrary benchmarks.
- **Alternatives Considered**:
  1. Define strict pass/fail gates (e.g. "Macro F1 must exceed 0.80 or the build fails").
  2. Treat quantitative metrics as directional engineering targets, reporting empirical numbers and analyzing failure modes transparently.
- **Chosen Option**: Alternative 2.
- **Rationale**: Arbitrary pass/fail gates encourage benchmark gaming and selective test filtering. Real engineering honesty requires reporting actual results, even when challenging edge cases reduce headline scores.
- **Trade-off**: System acceptance is judged on methodological rigor and honest failure characterization rather than artificial green checkmarks.

---

### Decision 15: Research-Prototype Framing & Methodological Honesty
- **Context**: Communicating the production readiness and operational boundaries of the system.
- **Alternatives Considered**:
  1. Market the system as production-ready for live commercial deployment.
  2. Explicitly frame the system as an evidence-grounded research prototype operating on historical 2017 public support data, documenting operational boundaries and caveats.
- **Chosen Option**: Alternative 2.
- **Rationale**: Professional engineering integrity demands distinguishing between a high-quality prototype demonstrating core concepts and a live production system handling real customer accounts.
- **Trade-off**: Prevents hyperbolic claims while highlighting methodological rigor.

---

### Decision 16: Bipartite Graph Connected-Component Partitioning for Leakage Prevention
- **Context**: In Twitter customer support, single customer authors engage across multiple threads, and multiple customer authors can reply within the same shared conversation. Naive partitioning by author alone left 11 thread overlaps; naive partitioning by thread alone left 94 author-day overlaps.
- **Alternatives Considered**:
  1. Split randomly by tweet ID (causes massive thread leakage).
  2. Split by author only (leaks shared conversation threads).
  3. Form a bipartite graph $G = (U_{\text{authors}}, V_{\text{threads}}, E)$ and partition by connected components.
- **Chosen Option**: Alternative 3.
- **Rationale**: Connected-component partitioning guarantees that any customer author and all conversation threads they touched exist exclusively in exactly one partition.
- **Result**: Zero tweet ID overlap, zero thread ID overlap, and zero author-day overlap across retrieval, validation, and gold sets.

---

### Decision 17: Multi-Objective Threshold Optimization under Hard Safety Constraint
- **Context**: Setting escalation thresholds ($\tau_{\text{conf}}, \tau_{\text{qual}}$) dictates the balance between automation and safety.
- **Alternatives Considered**:
  1. Hardcode thresholds to minimize escalation without validation tuning.
  2. Select thresholds solely to produce 0.0% False Auto-Handle Rate regardless of escalation impact.
  3. Run a deterministic grid sweep over the quarantined validation set, treat 0 false auto-handles on sensitive issues as a hard safety constraint, and among satisfying candidates select the least-conservative (highest safe auto-handle coverage) operating point.
- **Chosen Option**: Alternative 3.
- **Rationale**: Optimizing for maximum coverage under a hard safety constraint produces an empirically defensible operating point ($\tau_{\text{conf}}=0.45, \tau_{\text{qual}}=0.45$), achieving 0 sensitive false auto-handles with 11.4% safe auto-handling coverage on validation.

---

### Decision 18: Silver Development Benchmark vs Quarantined Gold Queue Distinction
- **Context**: Automated benchmark evaluation requires labelled test queries, but fabricating human annotations compromises scientific truthfulness.
- **Alternatives Considered**:
  1. Synthetically generate test queries with LLMs and claim they are "hand-labelled gold".
  2. Block all evaluation until an annotator manually reviews all 200 items.
  3. Establish an explicit tier distinction: a Silver Development Benchmark (`data/interim/silver_eval_set.jsonl`) for automated evaluation, while strictly isolating the Gold Candidate Queue (`data/gold/gold_annotation_queue.jsonl`) with blank labels awaiting manual human review.
- **Chosen Option**: Alternative 3.
- **Rationale**: Enables reproducible pipeline evaluation without deceptive claims, while providing a clear human-in-the-loop completion path.

---

### Decision 19: Four-Way Component Disjointness & Permanent Gold Candidate Quarantine
- **Context**: Using the same 200 inquiries for both the future Gold annotation queue and the development evaluation benchmark risks implicit metric-driven tuning leakage.
- **Alternatives Considered**:
  1. Share the same 200 records between the Gold queue and Silver Dev benchmark.
  2. Partition the graph into four strictly isolated, disjoint sets:
     - Quarantined Gold Candidate Queue: 200 components (200 inquiries, held out permanently)
     - Silver Development Benchmark: 200 components (200 inquiries, used for evaluation)
     - Quarantined Validation Split: 100 components (156 pairs, used for calibration & tuning)
     - Clean Retrieval & Training Corpus: 852 components (1,427 pairs, used for grounding & training)
- **Chosen Option**: Alternative 2.
- **Rationale**: Guarantees true quarantine of the future Gold set. A complete 4-way overlap audit verified 0 tweet ID, 0 thread ID, 0 author ID, and 0 author-day overlap across all 6 pairwise combinations.
- **Result**: The Gold candidate set has never entered and will never enter model training, calibration, threshold tuning, or development evaluation.

---

### Decision 20: 3-Tier Retrieval Evaluation Hierarchy & Honest Proxy Naming
- **Context**: When vector search ranks candidates by cosine similarity descending ($s_1 \ge s_2 \ge s_3$), defining relevance as $s_i \ge \tau$ makes rank 1 always the first relevant result, producing identical Hit@1, Hit@3, and MRR. Conversely, defining relevance by intent matching is a proxy, not true human ground truth.
- **Alternatives Considered**:
  1. Keep the similarity threshold test and claim high retrieval numbers.
  2. Call intent matching "independently grounded human semantic relevance".
  3. Establish three distinct, transparent measurements:
     - **Intent-Consistent Retrieval Relevance Proxy**: $(\text{candidate.intent} == \text{query.true\_intent}) \land (\text{len}(\text{brand\_reply}) > 10)$, reporting Proxy Hit@1, Proxy Hit@3, Proxy MRR, and first-relevant rank distribution. Explicitly documented as a proxy, NOT human-grounded relevance.
     - **Threshold Coverage Diagnostic**: Proportion of queries meeting similarity $\ge 0.45$, labeled as a retrieval-score diagnostic.
     - **Human Retrieval Relevance (Future)**: 50 queries $\times$ top-3 candidates (150 pairs) queued in `reports/annotations/retrieval_relevance_annotation_queue.jsonl` with status `PENDING_HUMAN_ANNOTATION`.
- **Chosen Option**: Alternative 3.
- **Rationale**: Enforces methodological honesty: separates retrieval score coverage from intent consistency proxy from human ground truth.
- **Result**: Decoupled first-relevant rank distribution (rank 1: 127, rank 2: 31, rank 3: 8, not in top 3: 34), yielding mathematically sound Proxy Hit@1: 63.5%, Proxy Hit@3: 83.0%, Proxy MRR: 0.7258, and Threshold Coverage: 91.0%.
