"""
Human vs LLM-as-a-Judge Agreement Study Workflow.
Samples 50 real customer inquiries, generates actual agent outputs and LLM Judge evaluations,
and builds a machine-readable human annotation queue.
Calculates statistical agreement metrics ONLY when genuine human annotations are provided.
Until human review is complete, reports status as PENDING_HUMAN_ANNOTATION.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.hiver_agent.config import AppConfig
from src.hiver_agent.nlp.classifier import IntentClassifier
from src.hiver_agent.retrieval.vector_store import VectorStore
from src.hiver_agent.retrieval.evidence_quality import EvidenceQualityAssessor
from src.hiver_agent.policy.escalation import EscalationPolicy
from src.hiver_agent.generation.provider import DeterministicGroundedProvider
from src.hiver_agent.generation.hallucination_checker import HallucinationChecker
from src.hiver_agent.evaluation.judge import LLMJudge


def main():
    print("=" * 75)
    print("HUMAN VS LLM-AS-A-JUDGE AGREEMENT WORKFLOW (50 REAL SAMPLES)")
    print("=" * 75)

    silver_path = project_root / "data" / "interim" / "silver_eval_set.jsonl"
    with open(silver_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    # Sample 50 real cases deterministically (5 from each of 10 intents)
    intents_seen = {}
    sampled_records = []
    for r in records:
        intent = r.get("true_intent", "other_unsupported")
        if intents_seen.get(intent, 0) < 5:
            sampled_records.append(r)
            intents_seen[intent] = intents_seen.get(intent, 0) + 1

    # If some intents have fewer than 5, fill to 50 from remaining records
    if len(sampled_records) < 50:
        remaining = [r for r in records if r not in sampled_records]
        sampled_records.extend(remaining[:50 - len(sampled_records)])

    print(f"Sampled {len(sampled_records)} real customer interactions across taxonomy intents.")

    # Load actual production components
    models_dir = project_root / "models"
    classifier = IntentClassifier()
    classifier = classifier.load(models_dir / "intent_classifier.pkl")
    vector_store = VectorStore.load(models_dir / "retrieval_index.pkl")
    assessor = EvidenceQualityAssessor()
    policy = EscalationPolicy()
    generator = DeterministicGroundedProvider()
    checker = HallucinationChecker()
    judge = LLMJudge()

    annotations_dir = project_root / "reports" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    queue_path = annotations_dir / "human_judge_agreement_queue.jsonl"

    # Check if existing queue with completed human annotations already exists
    existing_human_annotations = {}
    if queue_path.exists():
        with open(queue_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    h_scores = item.get("human_scores", {})
                    # If at least correctness is scored, consider it annotated
                    if h_scores and h_scores.get("correctness") is not None:
                        existing_human_annotations[item["sample_id"]] = h_scores

    print(f"Existing human-completed annotations found: {len(existing_human_annotations)}/50")

    queue_records = []
    judge_annotations = []
    human_annotations = []

    for idx, r in enumerate(sampled_records):
        sample_id = f"hjudge_{idx+1:03d}"
        query = r["customer_text"]
        gt_dec = r.get("ground_truth_decision", "ESCALATE")

        # Run REAL agent pipeline on real query
        intent_res = classifier.predict_one(query)
        evidence_res = vector_store.retrieve(query, top_k=3)
        ev_assess = assessor.assess_evidence(query, intent_res["predicted_intent"], evidence_res)
        gen_res = generator.generate_reply(query, intent_res["predicted_intent"], evidence_res)
        claim_ver = checker.verify_claims(gen_res.reply, evidence_res)
        dec = policy.evaluate(intent_res, ev_assess, claim_ver)

        # Run LLM Judge
        judge_score = judge.evaluate_reply(
            customer_query=query,
            predicted_intent=intent_res["predicted_intent"],
            draft_reply=gen_res.reply,
            retrieved_evidence=evidence_res,
            decision=dec.action,
            ground_truth_decision=gt_dec
        )
        judge_dict = {
            "correctness": judge_score.correctness,
            "relevance": judge_score.relevance,
            "grounding": judge_score.grounding,
            "completeness": judge_score.completeness,
            "tone": judge_score.tone,
            "unsupported_claims": judge_score.unsupported_claims,
            "escalation_appropriateness": judge_score.escalation_appropriateness,
            "justification": judge_score.justification
        }
        judge_annotations.append(judge_dict)

        # Human scores: preserve if already filled, otherwise empty template
        human_score = existing_human_annotations.get(sample_id, {
            "correctness": None,
            "relevance": None,
            "grounding": None,
            "completeness": None,
            "tone": None,
            "unsupported_claims": None,
            "escalation_appropriateness": None
        })
        if human_score.get("correctness") is not None:
            human_annotations.append(human_score)

        queue_item = {
            "sample_id": sample_id,
            "customer_tweet_id": r.get("customer_tweet_id", ""),
            "customer_query": query,
            "predicted_intent": intent_res["predicted_intent"],
            "calibrated_confidence": intent_res["calibrated_confidence"],
            "agent_reply": gen_res.reply,
            "agent_decision": dec.action,
            "agent_reason_code": dec.reason_code,
            "retrieved_evidence": [
                {
                    "evidence_id": e.get("evidence_id"),
                    "similarity": e.get("similarity"),
                    "historical_reply": e.get("historical_brand_reply")
                }
                for e in evidence_res
            ],
            "judge_scores": judge_dict,
            "human_scores": human_score,
            "human_annotator": "",
            "human_notes": ""
        }
        queue_records.append(queue_item)

    # Save queue file
    with open(queue_path, "w", encoding="utf-8") as f:
        for q in queue_records:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Saved real-data human evaluation queue to {queue_path}")

    # Generate results report
    reports_dir = project_root / "reports" / "results"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "human_vs_judge_agreement.json"

    if len(human_annotations) == len(sampled_records):
        # All human annotations provided: compute genuine statistics
        print("\nAll 50 human annotations provided. Computing genuine statistical agreement...")
        dimensions = ["correctness", "relevance", "grounding", "completeness", "tone", "unsupported_claims", "escalation_appropriateness"]
        dim_results = {}
        for dim in dimensions:
            j_vals = np.array([j[dim] for j in judge_annotations])
            h_vals = np.array([h[dim] for h in human_annotations])
            exact = float(np.mean(j_vals == h_vals))
            within_1 = float(np.mean(np.abs(j_vals - h_vals) <= 1))
            mad = float(np.mean(np.abs(j_vals - h_vals)))
            p_r = float(pearsonr(j_vals, h_vals)[0]) if np.std(j_vals) > 0 and np.std(h_vals) > 0 else 1.0
            s_rho = float(spearmanr(j_vals, h_vals)[0]) if np.std(j_vals) > 0 and np.std(h_vals) > 0 else 1.0

            dim_results[dim] = {
                "exact_match": round(exact, 4),
                "within_1": round(within_1, 4),
                "mad": round(mad, 4),
                "pearson_r": round(p_r, 4),
                "spearman_rho": round(s_rho, 4)
            }

        report_data = {
            "status": "COMPLETED",
            "sample_size": len(sampled_records),
            "completed_human_annotations": len(human_annotations),
            "dimensions": dim_results
        }
    else:
        # Human annotations pending: DO NOT FABRICATE SCORES
        print("\nHuman annotations pending. Marking experiment as PENDING_HUMAN_ANNOTATION.")
        print(f"Completed human annotations: {len(human_annotations)}/{len(sampled_records)}")
        report_data = {
            "status": "PENDING_HUMAN_ANNOTATION",
            "message": "Human vs LLM-as-a-Judge agreement study is queued for human annotation. Zero fabricated scores.",
            "queue_path": str(queue_path),
            "sample_size": len(sampled_records),
            "completed_human_annotations": len(human_annotations),
            "agreement_metrics": None,
            "instructions": "To complete, annotate the 50 items in reports/annotations/human_judge_agreement_queue.jsonl with human_scores (1-5 scale) and re-run this script."
        }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved human-vs-judge report to {report_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
