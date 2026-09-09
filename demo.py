"""
Interactive Demonstration CLI for Hiver Brand AI Support Agent.
Accepts customer inquiry and displays:
- Predicted intent & calibrated confidence
- Novelty outlier status
- Retrieved historical evidence with provenance
- Evidence quality & contradiction assessment
- Auto-Handle vs Escalate decision with explicit reason
- Drafted reply with claim-level grounding verification
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.hiver_agent.nlp.classifier import IntentClassifier
from src.hiver_agent.retrieval.vector_store import VectorStore
from src.hiver_agent.retrieval.evidence_quality import EvidenceQualityAssessor
from src.hiver_agent.policy.escalation import EscalationPolicy
from src.hiver_agent.generation.provider import DeterministicGroundedProvider
from src.hiver_agent.generation.hallucination_checker import HallucinationChecker


def process_query(query: str, clf, vstore, assessor, policy, generator, checker):
    print("\n" + "=" * 80)
    print(f"CUSTOMER QUERY: \"{query}\"")
    print("=" * 80)

    # 1. Intent Classification
    intent_res = clf.predict_one(query)
    print("\n[1] INTENT CLASSIFICATION")
    print(f"  Predicted Intent:      {intent_res['predicted_intent']}")
    print(f"  Calibrated Confidence: {intent_res['calibrated_confidence'] * 100:.1f}%")
    print(f"  Raw Logit:             {intent_res['raw_logit']:.2f}")
    print(f"  Temperature T:         {intent_res['temperature']:.2f}")
    print(f"  Min Centroid Distance: {intent_res['min_centroid_distance']:.3f} (Novelty Outlier: {intent_res['is_novelty_outlier']})")
    print("  Runner-up Intents:     " + ", ".join([f"{a['intent']} ({a['confidence']*100:.1f}%)" for a in intent_res['alternatives'][:2]]))

    # 2. Evidence Retrieval
    evidence_res = vstore.retrieve(query, top_k=3)
    print("\n[2] HISTORICAL EVIDENCE RETRIEVAL (SPOTIFYCARES CORPUS)")
    print(f"  Retrieved Evidence Candidates: {len(evidence_res)}")
    for i, ev in enumerate(evidence_res, 1):
        print(f"  Candidate {i} [{ev['evidence_id']}] (Cosine Similarity: {ev['similarity']:.3f}):")
        print(f"    Historical Cust:  {ev['historical_customer'][:80]}...")
        print(f"    Historical Reply: {ev['historical_brand_reply'][:90]}...")

    # 3. Evidence Quality & Contradiction Analysis
    assessment = assessor.assess_evidence(query, intent_res["predicted_intent"], evidence_res)
    print("\n[3] EVIDENCE QUALITY & CONTRADICTION ASSESSMENT")
    print(f"  Evidence Quality Score: {assessment['evidence_quality_score']:.3f}")
    print(f"  Contradiction Status:   {assessment['contradiction_status']}")
    print(f"  Contradiction Note:     {assessment['contradiction_explanation']}")

    # 4. Draft Reply Synthesis
    gen_res = generator.generate_reply(query, intent_res["predicted_intent"], evidence_res)
    claim_ver = checker.verify_claims(gen_res.reply, evidence_res)

    # 5. Escalation Decision Policy
    decision = policy.evaluate(intent_res, assessment, claim_ver)
    print("\n[4] ESCALATION POLICY DECISION")
    if decision.action == "AUTO_HANDLE":
        print(f"  ACTION:      >>> {decision.action} <<<")
    else:
        print(f"  ACTION:      *** {decision.action} ***")
    print(f"  Reason Code: {decision.reason_code}")
    print(f"  Explanation: {decision.reason}")
    print(f"  Risk Score:  {decision.risk_score:.2f}")

    # 6. Draft Reply
    print("\n[5] DRAFTED GROUNDED REPLY")
    print(f"  \"{gen_res.reply}\"")
    print(f"  Grounded in Evidence IDs: {gen_res.grounded_in_evidence_ids}")
    print(f"  Claim Support Verdict:    {claim_ver['total_claims']} claims verified ({claim_ver['supported_claims']} supported, {claim_ver['unsupported_claims']} unsupported)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Hiver Brand AI Support Agent Demo CLI")
    parser.add_argument("--query", type=str, help="Customer inquiry string")
    args = parser.parse_args()

    print("Loading model artifacts...")
    clf = IntentClassifier.load(project_root / "models" / "intent_classifier.pkl")
    vstore = VectorStore.load(project_root / "models" / "retrieval_index.pkl")
    assessor = EvidenceQualityAssessor()
    policy = EscalationPolicy()
    generator = DeterministicGroundedProvider()
    checker = HallucinationChecker()

    if args.query:
        process_query(args.query, clf, vstore, assessor, policy, generator, checker)
    else:
        # Interactive mode or sample showcase
        sample_queries = [
            "My desktop app crashes instantly on Windows 10 after today's update",
            "I was charged $9.99 twice for Premium this month. I want a refund.",
            "Someone hacked my account and changed the email address to Russia!",
            "How do I plant organic tomatoes in my garden?",
            "Music keeps pausing every 10 seconds while playing on my iPhone"
        ]
        print("\nRunning interactive showcase across 5 representative customer inquiries:\n")
        for q in sample_queries:
            process_query(q, clf, vstore, assessor, policy, generator, checker)


if __name__ == "__main__":
    main()
