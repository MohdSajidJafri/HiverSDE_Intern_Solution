"""
Master Evaluation Runner for Hiver Brand AI Support Agent.
Executes frozen evaluation on data/gold/gold_messages.jsonl across:
- Primary Grounded Agent (Frozen)
- Baseline 1 (Trivial)
- Baseline 2 (Simple)
Reports both Stratified and Natural-Distribution metrics, generates comparative tables,
and saves evaluation results to reports/results/evaluation_results.json.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.hiver_agent.config import AppConfig
from src.hiver_agent.nlp.classifier import IntentClassifier
from src.hiver_agent.retrieval.vector_store import VectorStore
from src.hiver_agent.retrieval.evidence_quality import EvidenceQualityAssessor
from src.hiver_agent.policy.escalation import EscalationPolicy
from src.hiver_agent.generation.provider import DeterministicGroundedProvider, GenerativeLLMProvider
from src.hiver_agent.generation.hallucination_checker import HallucinationChecker
from src.hiver_agent.evaluation.harness import EvaluationHarness
from src.hiver_agent.evaluation.baselines import Baseline1Trivial, Baseline2Simple


def run_agent_pipeline(
    query: str,
    classifier: IntentClassifier,
    vector_store: VectorStore,
    assessor: EvidenceQualityAssessor,
    policy: EscalationPolicy,
    generator: DeterministicGroundedProvider,
    checker: HallucinationChecker
) -> Dict[str, Any]:
    """Runs the complete end-to-end inference pipeline on a single query."""
    # 1. Calibrated intent classification & outlier detection
    intent_res = classifier.predict_one(query)

    # 2. Semantic evidence retrieval
    evidence_res = vector_store.retrieve(query, top_k=3)

    # 3. Evidence quality & contradiction assessment
    evidence_assessment = assessor.assess_evidence(
        query=query,
        predicted_intent=intent_res["predicted_intent"],
        retrieved_evidence=evidence_res
    )

    # 4. Draft reply synthesis
    gen_res = generator.generate_reply(
        customer_text=query,
        predicted_intent=intent_res["predicted_intent"],
        evidence=evidence_res
    )

    # 5. Claim-level grounding & hallucination check
    claim_ver = checker.verify_claims(gen_res.reply, evidence_res)

    # 6. Conservative escalation decision
    decision = policy.evaluate(
        intent_result=intent_res,
        evidence_assessment=evidence_assessment,
        claim_verification=claim_ver
    )

    return {
        "intent": {
            "predicted": intent_res["predicted_intent"],
            "confidence": intent_res["calibrated_confidence"],
            "raw_logit": intent_res["raw_logit"],
            "is_outlier": intent_res["is_novelty_outlier"],
            "prob_vector": intent_res.get("prob_vector", []),
            "calibrated_probabilities": intent_res.get("calibrated_probabilities", {}),
            "alternatives": intent_res["alternatives"]
        },
        "retrieval": {
            "evidence_count": len(evidence_res),
            "evidence_quality_score": evidence_assessment["evidence_quality_score"],
            "contradiction_status": evidence_assessment["contradiction_status"],
            "evidence": evidence_res
        },
        "decision": {
            "action": decision.action,
            "reason_code": decision.reason_code,
            "reason": decision.reason,
            "risk_score": decision.risk_score
        },
        "reply": {
            "draft": gen_res.reply,
            "grounded_in_evidence_ids": gen_res.grounded_in_evidence_ids,
            "claim_verification": claim_ver,
            "unsupported_claims": claim_ver.get("claims_detail", [])
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hiver Support Agent on Benchmark Set")
    parser.add_argument("--gold-set", type=str, default="data/interim/silver_eval_set.jsonl", help="Path to evaluation benchmark set")
    parser.add_argument("--output", type=str, default="reports/results/evaluation_results.json", help="Path to output results")
    args = parser.parse_args()

    print("=" * 80)
    print("FROZEN EVALUATION RUNNER: HIVER BRAND SUPPORT AGENT")
    print("=" * 80)

    # Enforce strict Gold Quarantine: the unlabelled candidate gold queue must never be evaluated
    gold_path = project_root / args.gold_set
    if "gold_annotation_queue" in str(gold_path):
        raise ValueError(
            "CRITICAL METHODOLOGICAL ERROR: Attempted to run evaluation on the quarantined Gold Candidate Queue! "
            "The 200 Gold candidate records are strictly quarantined from development evaluation."
        )

    # Load authoritative frozen configuration (fails loudly on config mismatch)
    config = AppConfig.load_authoritative()
    print(f"Loaded Authoritative Frozen Configuration:")
    print(f"  tau_conf: {config.thresholds.intent_confidence_threshold}")
    print(f"  tau_qual: {config.thresholds.evidence_quality_threshold}")

    # Load evaluation dataset
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_records = [json.loads(line) for line in f]
    is_gold = any(r.get("is_human_annotated_gold", False) for r in gold_records)
    tier_label = "HUMAN_GOLD" if is_gold else "SILVER_DEVELOPMENT_BENCHMARK"
    print(f"Loaded {len(gold_records)} evaluation records (Tier: {tier_label}).")

    # Load trained models & components
    print("Loading frozen model artifacts...")
    classifier = IntentClassifier.load(project_root / "models" / "intent_classifier.pkl")
    vector_store = VectorStore.load(project_root / "models" / "retrieval_index.pkl")
    assessor = EvidenceQualityAssessor()
    policy = EscalationPolicy(config=config)
    generator = DeterministicGroundedProvider()
    checker = HallucinationChecker()

    # Fit Baseline 2 (Simple)
    retrieval_path = project_root / "data" / "processed" / "retrieval_corpus.jsonl"
    with open(retrieval_path, "r", encoding="utf-8") as f:
        retrieval_pairs = [json.loads(line) for line in f]
    
    from scripts.train_classifier import assign_silver_training_intent
    baseline_simple = Baseline2Simple()
    train_subset = retrieval_pairs[:800]
    intent_map = {p["customer_text"]: assign_silver_training_intent(p["customer_text"]) for p in train_subset}
    baseline_simple.fit(train_subset, intent_map)

    baseline_trivial = Baseline1Trivial()

    # Run predictions across all 3 systems
    print("Executing frozen evaluation across Primary Agent and Baselines...")
    primary_preds = []
    base1_preds = []
    base2_preds = []

    for r in gold_records:
        q = r["customer_text"]
        p_res = run_agent_pipeline(q, classifier, vector_store, assessor, policy, generator, checker)
        primary_preds.append(p_res)

        b1_res = baseline_trivial.predict(q)
        base1_preds.append(b1_res)

        b2_res = baseline_simple.predict(q)
        base2_preds.append(b2_res)

    harness = EvaluationHarness()

    # Evaluate each system
    primary_eval = harness.evaluate_predictions(gold_records, primary_preds)
    base1_eval = harness.evaluate_predictions(gold_records, base1_preds)
    base2_eval = harness.evaluate_predictions(gold_records, base2_preds)

    # Save results
    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(primary_eval, f, indent=2)
    print(f"Saved primary evaluation results to {out_path}")

    comp_path = project_root / "reports" / "results" / "baseline_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump({
            "primary_system": primary_eval,
            "baseline_1_trivial": base1_eval,
            "baseline_2_simple": base2_eval
        }, f, indent=2)
    print(f"Saved baseline comparison to {comp_path}")

    # Print Headline Comparison Table
    print("\n" + "=" * 105)
    print("HEADLINE RESULTS VS BASELINES (EVALUATED ON IDENTICAL FROZEN GOLD SET)")
    print("=" * 105)
    print(f"{'Metric':<35} {'Baseline 1 (Trivial)':<22} {'Baseline 2 (Simple)':<22} {'Primary Agent (Frozen)':<22}")
    print("-" * 105)

    metrics_to_show = [
        ("Intent Accuracy (Stratified)", base1_eval["intent_classification"]["stratified_view"]["accuracy"], base2_eval["intent_classification"]["stratified_view"]["accuracy"], primary_eval["intent_classification"]["stratified_view"]["accuracy"]),
        ("Intent Macro F1 (Stratified)", base1_eval["intent_classification"]["stratified_view"]["macro_f1"], base2_eval["intent_classification"]["stratified_view"]["macro_f1"], primary_eval["intent_classification"]["stratified_view"]["macro_f1"]),
        ("Intent Accuracy (Natural View)", base1_eval["intent_classification"]["natural_distribution_view"]["accuracy"], base2_eval["intent_classification"]["natural_distribution_view"]["accuracy"], primary_eval["intent_classification"]["natural_distribution_view"]["accuracy"]),
        ("Intent Weighted F1 (Natural View)", base1_eval["intent_classification"]["natural_distribution_view"]["weighted_f1"], base2_eval["intent_classification"]["natural_distribution_view"]["weighted_f1"], primary_eval["intent_classification"]["natural_distribution_view"]["weighted_f1"]),
        ("Expected Calibration Error (ECE)", base1_eval["intent_classification"]["calibration"]["expected_calibration_error"], base2_eval["intent_classification"]["calibration"]["expected_calibration_error"], primary_eval["intent_classification"]["calibration"]["expected_calibration_error"]),
        ("Brier Calibration Score", base1_eval["intent_classification"]["calibration"]["brier_score"], base2_eval["intent_classification"]["calibration"]["brier_score"], primary_eval["intent_classification"]["calibration"]["brier_score"]),
        ("Safe Auto-Handle Coverage", base1_eval["escalation_policy"]["safe_auto_handle_coverage"], base2_eval["escalation_policy"]["safe_auto_handle_coverage"], primary_eval["escalation_policy"]["safe_auto_handle_coverage"]),
        ("False Auto-Handle Rate (CRITICAL)", base1_eval["escalation_policy"]["false_auto_handle_rate"], base2_eval["escalation_policy"]["false_auto_handle_rate"], primary_eval["escalation_policy"]["false_auto_handle_rate"]),
        ("Escalation Rate", base1_eval["escalation_policy"]["escalation_rate"], base2_eval["escalation_policy"]["escalation_rate"], primary_eval["escalation_policy"]["escalation_rate"]),
        ("Proxy Retrieval Hit@1", base1_eval["evidence_retrieval"].get("proxy_hit_at_1", 0.0), base2_eval["evidence_retrieval"].get("proxy_hit_at_1", 0.0), primary_eval["evidence_retrieval"].get("proxy_hit_at_1", 0.0)),
        ("Proxy Retrieval Hit@3", base1_eval["evidence_retrieval"].get("proxy_hit_at_3", 0.0), base2_eval["evidence_retrieval"].get("proxy_hit_at_3", 0.0), primary_eval["evidence_retrieval"].get("proxy_hit_at_3", 0.0)),
        ("Proxy MRR", base1_eval["evidence_retrieval"].get("proxy_mrr", 0.0), base2_eval["evidence_retrieval"].get("proxy_mrr", 0.0), primary_eval["evidence_retrieval"].get("proxy_mrr", 0.0)),
        ("Threshold Coverage (Sim >= 0.45)", base1_eval["evidence_retrieval"].get("threshold_coverage_diagnostic", {}).get("top1_coverage", 0.0), base2_eval["evidence_retrieval"].get("threshold_coverage_diagnostic", {}).get("top1_coverage", 0.0), primary_eval["evidence_retrieval"].get("threshold_coverage_diagnostic", {}).get("top1_coverage", 0.0)),
        ("Unsupported-Claim Rate (Safety)", base1_eval["reply_generation"]["unsupported_claim_rate"], base2_eval["reply_generation"]["unsupported_claim_rate"], primary_eval["reply_generation"]["unsupported_claim_rate"]),
        ("Grounded-Response Rate", base1_eval["reply_generation"]["grounded_response_rate"], base2_eval["reply_generation"]["grounded_response_rate"], primary_eval["reply_generation"]["grounded_response_rate"])
    ]

    for name, b1, b2, prim in metrics_to_show:
        if isinstance(b1, float) and "Rate" in name or "Coverage" in name or "Accuracy" in name or "F1" in name or "Score" in name:
            print(f"{name:<35} {b1*100:<21.1f}% {b2*100:<21.1f}% {prim*100:<21.1f}%")
        else:
            print(f"{name:<35} {b1:<22} {b2:<22} {prim:<22}")

    print("-" * 105)
    print("=" * 80)


if __name__ == "__main__":
    main()
