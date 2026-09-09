# System Architecture: Evidence-Grounded Brand AI Support Agent (Research Prototype)

## 1. Architectural Philosophy & Research Framing
**System Classification**: This system is an evidence-grounded research and evaluation prototype designed to demonstrate auditability, calibration, and conservative escalation on historical customer support interactions (Customer Support on Twitter dataset, late 2017). It maintains production-quality engineering standards while explicitly acknowledging that it is not deployed into live production for any commercial brand.

The central design principle:
**Historical brand interactions are empirical precedent, but must be audited, calibrated, and guarded before driving automated customer responses.**

The system enforces a strict, inspectable pipeline:
```
Raw Message 
  -> Calibrated Classification (Sentence Transformer + Multinomial Logistic Regression + Temperature Scaling)
  -> Auxiliary Centroid Distance (Novelty / Out-of-Scope Detection)
  -> Semantic Evidence Retrieval (Historical Brand Resolutions)
  -> Evidence Quality & Contradiction Audit (Compatible vs Incompatible vs Insufficient Evidence)
  -> Conservative Policy Decision (Auto-Handle vs Escalate)
  -> Grounded Reply Synthesis (Dual Provider: Offline Deterministic vs Generative LLM)
  -> Sentence/Claim Support Verification (Supported, Unsupported, Uncertain)
  -> Output Validation & Provenance Logging
```

---

## 2. Pipeline Flow Diagram

```
                                 [ Customer Message ]
                                          │
                                          ▼
                         [ Text Cleaning & Normalization ]
                                          │
                                          ▼
                         [ Sentence Embedding Generator ]
                            (all-MiniLM-L6-v2 / 384-d)
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
    [ Calibrated Classifier ]                          [ Auxiliary Centroid Detector ]
(Multinomial Logistic Regression trained               (Cosine distance to class
 on training embeddings; calibrated via                 geometric centroids; flags
 Multiclass Temperature Scaling T > 0)                  novel / out-of-scope queries)
                 │                                                 │
   (Intent Logits & Calibrated P)                                  │
                 │                                                 │
                 └────────────────────────┬────────────────────────┘
                                          │
                                          ▼
                         [ Semantic Evidence Retrieval ]
                      (Vector Index over Historical Pairs)
                                          │
                            (Top-K Evidence Candidates)
                                          │
                                          ▼
                         [ Evidence Quality & Audit Layer ]
                      - Solution-Level Relevance Assessment
                      - 3-State Contradiction Analysis:
                        * Compatible historical advice
                        * Incompatible historical advice
                        * Insufficient evidence
                      - Intent Alignment Check
                      - Resolution Completeness Assessment
                                          │
                          (Audited Evidence Assessment)
                                          │
                                          ▼
                         [ Escalation Policy Engine ]
                  (Frozen Thresholds: Conf_th, Quality_th)
                  Evaluates:
                  - Calibrated intent probability < Conf_th
                  - Top evidence quality score < Quality_th
                  - Incompatible historical resolutions flagged
                  - Safety-sensitive / account-critical intent
                  - Centroid novelty / out-of-scope anomaly
                  - Query ambiguity / multi-intent syntax
                                          │
                           ┌──────────────┴──────────────┐
                           ▼                             ▼
                [ Decision: ESCALATE ]       [ Decision: AUTO_HANDLE ]
                           │                             │
                           │                             ▼
                           │             [ Grounded Generation Pipeline ]
                           │             Dual Provider Abstraction:
                           │             - Provider A: Offline Deterministic
                           │               (Local synthesis for <15 min eval)
                           │             - Provider B: Generative LLM
                           │               (Live API with JSON schema)
                           │                             │
                           │                             ▼
                           │             [ Claim-Level Support Verifier ]
                           │             Classifies claims:
                           │             * Supported
                           │             * Unsupported
                           │             * Uncertain (treated conservatively)
                           │                             │
                           └──────────────┬──────────────┘
                                          │
                                          ▼
                             [ Structured Output Payload ]
                             (Decision, Reason, Grounding, Trace)
```

---

## 3. Component Specifications

### 3.1 Text Normalization Layer (`src/hiver_agent/nlp/normalizer.py`)
- Strips Twitter handle mentions (`@username`), excessive whitespace, and malformed encoding artifacts.
- Preserves URLs, emojis, and punctuation necessary for intent and frustration detection.
- Retains raw customer text alongside normalized text for auditing.

### 3.2 Calibrated Intent Classifier (`src/hiver_agent/nlp/classifier.py`)
- **Embedding Backbone**: Sentence Transformers (`all-MiniLM-L6-v2`, 384-dimensional dense vectors).
- **Classification Model**: Multinomial Logistic Regression trained on diverse labelled training/development examples represented by their sentence embeddings.
- **Multiclass Temperature Scaling Calibration**:
  Logits $\mathbf{z} \in \mathbb{R}^K$ produced by the linear model are scaled by a learned scalar parameter $T > 0$:
  $$p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$
  The optimal temperature $T$ is fitted on the validation set logits by minimizing multiclass negative log-likelihood (cross-entropy loss) via L-BFGS. Temperature scaling preserves the argmax prediction while directly aligning confidence with empirical accuracy.
- **Calibration Diagnostics**: Expected Calibration Error (ECE) across 10 confidence bins and multiclass Brier Score:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|, \quad \text{Brier} = \frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K (p_{nk} - y_{nk})^2$$
- **Auxiliary Centroid Novelty Detector**: Separately maintains the class geometric centroid $\mathbf{c}_k = \frac{1}{|S_k|} \sum_{\mathbf{x} \in S_k} \mathbf{x}$ for each intent. Incoming queries with $\min_k (1 - \cos(\mathbf{x}, \mathbf{c}_k)) > \delta_{\text{novelty}}$ are flagged as novel/out-of-scope anomalies regardless of softmax confidence.

### 3.3 Semantic Evidence Retrieval (`src/hiver_agent/retrieval/vector_store.py`)
- Built exclusively from historical conversation pairs (`customer_message` $\to$ `brand_reply`) from the selected brand corpus.
- Employs cosine similarity over normalized embeddings.
- Fast Top-$K$ retrieval ($K=3$) with complete provenance metadata (`evidence_id`, `historical_tweet_id`, `conversation_id`, `timestamp`).

### 3.4 Evidence Quality & Contradiction Layer (`src/hiver_agent/retrieval/evidence_quality.py`)
Embedding proximity alone does not guarantee reliable evidence. This layer evaluates:
1. **Solution-Level Relevance**: Verifies that the historical brand reply contains actionable guidance (diagnostic questions, settings instructions, troubleshooting steps, official policy) rather than non-informative pleasantries.
2. **Intent Consistency**: Verifies that the historical interaction aligns with the predicted intent category.
3. **3-State Contradiction Analysis**:
   - **Compatible Advice**: Different historical suggestions that can coexist peacefully (e.g., candidate 1 suggests restarting device, candidate 2 suggests clearing app cache).
   - **Incompatible Advice**: Genuinely conflicting resolutions (e.g., candidate 1 advises uninstalling and reinstalling app; candidate 2 explicitly advises against reinstallation due to an ongoing server outage).
   - **Insufficient Evidence**: Retained historical interactions are too sparse or vague to confirm compatibility.
   *Empirical Limitation*: Automated contradiction detection is inherently an approximation based on action extraction and semantic divergence; any ambiguous or borderline conflict is treated conservatively to trigger escalation.

### 3.5 Conservative Escalation Policy Engine (`src/hiver_agent/policy/escalation.py`)
A transparent rule-based policy enforces safe routing based on parameters tuned on the validation set and frozen:
- **Decision Outputs**: `AUTO_HANDLE` or `ESCALATE`
- **Inspectable Reason Codes**:
  - `LOW_INTENT_CONFIDENCE`: Calibrated confidence $< \tau_{\text{conf}}$
  - `LOW_EVIDENCE_QUALITY`: Top evidence quality $< \tau_{\text{quality}}$
  - `CONTRADICTORY_HISTORICAL_EVIDENCE`: Incompatible advice detected across retrieved precedent
  - `SENSITIVE_ACCOUNT_DOMAIN`: Intents requiring human authority (e.g. billing disputes, account compromise, refund requests)
  - `OUT_OF_SCOPE_ANOMALY`: Distance to nearest intent centroid exceeds $\delta_{\text{novelty}}$
  - `AMBIGUOUS_OR_MULTI_INTENT`: High probability split across incompatible intents

### 3.6 Grounded Reply Generation & Dual Provider Interface (`src/hiver_agent/generation/provider.py`)
```python
class LLMProvider(ABC):
    @abstractmethod
    def generate_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        evidence: List[EvidenceRecord],
        system_constraints: Dict[str, Any]
    ) -> GenerationResult:
        pass
```
- **Provider A (`DeterministicGroundedProvider`)**:
  Synthesizes replies purely from retrieved historical resolutions using deterministic slot-filling and verified brand macros. Enables instant, zero-cost, offline reproduction of the headline evaluation in under 15 minutes.
- **Provider B (`GenerativeLLMProvider`)**:
  Connects to external LLM APIs (OpenAI, Gemini, Anthropic, or Ollama) using structured JSON prompts and strict schema validation.
  *Note*: The deterministic provider is provided for reproducible testing and CI, and is never conflated with or claimed as evidence of generative LLM capability.

### 3.7 Claim-Level Support Verifier (`src/hiver_agent/generation/hallucination_checker.py`)
- Deconstructs drafted replies into discrete sentences/factual claims.
- Assesses evidence support against retrieved historical text, classifying each claim into:
  - `SUPPORTED`: Traceable directly to verified historical evidence.
  - `UNSUPPORTED`: Factual assertions, policies, or timeline guarantees absent from retrieved evidence.
  - `UNCERTAIN`: Claims where alignment is partial or ambiguous.
- **Conservative Treatment**: Any `UNCERTAIN` claim is treated as ungrounded for risk scoring, triggering escalation if confidence is marginal.
- *Empirical Limitation*: The claim verifier is an automated heuristic, not a guarantee against all possible hallucinations.

---

## 4. Structured Output Contract
```json
{
  "message_id": "cust_msg_001",
  "customer_text": "My desktop app crashes immediately on launch after update.",
  "intent": {
    "predicted": "app_crash_technical",
    "calibrated_confidence": 0.892,
    "raw_logit": 3.45,
    "temperature": 1.18,
    "is_novelty_outlier": false,
    "centroid_distance": 0.28,
    "alternatives": [
      {"intent": "playback_issues", "confidence": 0.061}
    ]
  },
  "retrieval": {
    "evidence_count": 3,
    "top_similarity": 0.845,
    "evidence_quality_score": 0.88,
    "contradiction_status": "COMPATIBLE",
    "evidence": [
      {
        "evidence_id": "ev_4821",
        "historical_tweet_id": 116128,
        "similarity": 0.845,
        "historical_customer": "Desktop app still not working after update.",
        "historical_brand_reply": "Thanks! It's not ideal, but a clean reinstall of the app should help out: https://t.co/EqisD",
        "solution_type": "troubleshooting_step"
      }
    ]
  },
  "decision": {
    "action": "AUTO_HANDLE",
    "reason_code": "HIGH_CONFIDENCE_GROUNDED",
    "reason": "Calibrated confidence and evidence quality exceed operating thresholds with compatible historical advice.",
    "risk_score": 0.12
  },
  "reply": {
    "draft": "We understand how frustrating that is! A clean reinstall usually clears up post-update crashes on desktop. Could you give that a try and let us know your exact OS version if it persists?",
    "grounded_in_evidence_ids": ["ev_4821"],
    "claim_verification": {
      "supported_claims": 2,
      "unsupported_claims": 0,
      "uncertain_claims": 0,
      "verdict": "VERIFIED_GROUNDED"
    }
  },
  "provenance": {
    "brand": "SpotifyCares",
    "model_version": "v1.0-frozen",
    "freeze_manifest_sha": "a1b2c3d4e5f6...",
    "timestamp": "2026-09-09T14:35:00Z"
  }
}
```

---

## 5. Security, Reliability & Freeze Manifest
- **Environment-Based Secrets**: Zero API keys committed to source control.
- **Fail-Safe Fallback**: If LLM generation or verification fails, the system safely falls back to `ESCALATE` with reason `LLM_SERVICE_UNAVAILABLE`.
- **System Freeze Manifest (`models/freeze_manifest.json`)**:
  Prior to gold evaluation, an audit manifest records:
  - Git commit SHA
  - Source dataset SHA256
  - Training, validation, and gold dataset SHA256 hashes
  - Dependency environment snapshot / package hashes
  - Embedding model name and version
  - Classifier weights artifact hash
  - Retrieval index artifact hash
  - Temperature scaling parameter $T$
  - Threshold parameters ($\tau_{\text{conf}}$, $\tau_{\text{quality}}$, $\delta_{\text{novelty}}$)
  - Policy configuration hash
  - Prompt template version hash
