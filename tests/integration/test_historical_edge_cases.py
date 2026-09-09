"""
Integration tests for historical customer support edge cases:
- Conflicting historical replies
- Weak evidence
- Highly similar but wrong evidence
- Novel questions / out-of-scope anomalies
- Multi-intent messages
- Short / noisy messages
- Stale / outdated historical guidance
"""

import pytest
from evaluate import run_agent_pipeline
from src.hiver_agent.nlp.classifier import IntentClassifier
from src.hiver_agent.retrieval.vector_store import VectorStore
from src.hiver_agent.retrieval.evidence_quality import EvidenceQualityAssessor
from src.hiver_agent.policy.escalation import EscalationPolicy
from src.hiver_agent.generation.provider import DeterministicGroundedProvider
from src.hiver_agent.generation.hallucination_checker import HallucinationChecker


@pytest.fixture(scope="module")
def pipeline_components():
    clf = IntentClassifier.load("models/intent_classifier.pkl")
    vstore = VectorStore.load("models/retrieval_index.pkl")
    assessor = EvidenceQualityAssessor()
    policy = EscalationPolicy()
    generator = DeterministicGroundedProvider()
    checker = HallucinationChecker()
    return clf, vstore, assessor, policy, generator, checker


def test_edge_case_novel_question(pipeline_components):
    """Novel out-of-scope question must trigger escalation."""
    clf, vstore, assessor, policy, generator, checker = pipeline_components
    query = "How do I plant organic red tomatoes in my backyard garden?"
    res = run_agent_pipeline(query, clf, vstore, assessor, policy, generator, checker)

    assert res["decision"]["action"] == "ESCALATE"
    # Either caught by novelty outlier or low confidence or out of scope
    assert res["decision"]["reason_code"] in ["OUT_OF_SCOPE_ANOMALY", "LOW_INTENT_CONFIDENCE", "LOW_EVIDENCE_QUALITY"]


def test_edge_case_multi_intent_compound_message(pipeline_components):
    """Multi-intent message must trigger escalation due to split confidence or sensitive component."""
    clf, vstore, assessor, policy, generator, checker = pipeline_components
    query = "App crashes repeatedly and also I was charged twice for Premium subscription"
    res = run_agent_pipeline(query, clf, vstore, assessor, policy, generator, checker)

    assert res["decision"]["action"] == "ESCALATE"


def test_edge_case_ultra_short_noisy_query(pipeline_components):
    """Ultra-short query lacks context and should trigger escalation."""
    clf, vstore, assessor, policy, generator, checker = pipeline_components
    query = "help"
    res = run_agent_pipeline(query, clf, vstore, assessor, policy, generator, checker)

    assert res["decision"]["action"] == "ESCALATE"


def test_edge_case_conflicting_historical_guidance(pipeline_components):
    """Simulated conflicting evidence must trigger escalation with CONTRADICTORY_HISTORICAL_EVIDENCE."""
    clf, vstore, assessor, policy, generator, checker = pipeline_components

    # Inject mock contradictory evidence
    evidence = [
        {"evidence_id": "ev_1", "historical_brand_reply": "Try a clean reinstall of the app /CH", "similarity": 0.85},
        {"evidence_id": "ev_2", "historical_brand_reply": "Do not reinstall! We have an ongoing server outage right now /GS", "similarity": 0.82}
    ]
    assessment = assessor.assess_evidence("App not opening", "app_crash_technical", evidence)
    assert assessment["has_contradiction"] is True

    intent_res = {"predicted_intent": "app_crash_technical", "calibrated_confidence": 0.88, "is_novelty_outlier": False}
    decision = policy.evaluate(intent_res, assessment)

    assert decision.action == "ESCALATE"
    assert decision.reason_code == "CONTRADICTORY_HISTORICAL_EVIDENCE"


def test_edge_case_weak_evidence(pipeline_components):
    """Weak / low-similarity evidence must trigger escalation."""
    clf, vstore, assessor, policy, generator, checker = pipeline_components

    evidence = [
        {"evidence_id": "ev_weak", "historical_brand_reply": "Thanks! /CH", "similarity": 0.35}
    ]
    assessment = assessor.assess_evidence("obscure bug", "playback_issues", evidence)
    intent_res = {"predicted_intent": "playback_issues", "calibrated_confidence": 0.80, "is_novelty_outlier": False}
    decision = policy.evaluate(intent_res, assessment)

    assert decision.action == "ESCALATE"
    assert decision.reason_code == "LOW_EVIDENCE_QUALITY"
