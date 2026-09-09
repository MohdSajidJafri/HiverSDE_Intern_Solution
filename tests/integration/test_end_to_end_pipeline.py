"""
Integration tests for the end-to-end support agent pipeline.
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


def test_end_to_end_technical_issue(pipeline_components):
    clf, vstore, assessor, policy, generator, checker = pipeline_components
    query = "Desktop app crashing repeatedly on Windows after update"
    res = run_agent_pipeline(query, clf, vstore, assessor, policy, generator, checker)

    assert "intent" in res
    assert "retrieval" in res
    assert "decision" in res
    assert "reply" in res
    assert res["intent"]["predicted"] in ["app_crash_technical", "playback_issues"]
    assert len(res["retrieval"]["evidence"]) > 0
    assert len(res["reply"]["draft"]) > 10


def test_end_to_end_sensitive_billing_escalation(pipeline_components):
    clf, vstore, assessor, policy, generator, checker = pipeline_components
    query = "I was charged $9.99 twice for Premium this month on my credit card"
    res = run_agent_pipeline(query, clf, vstore, assessor, policy, generator, checker)

    assert res["intent"]["predicted"] == "subscription_billing"
    # Must escalate strictly
    assert res["decision"]["action"] == "ESCALATE"
    assert res["decision"]["reason_code"] == "SENSITIVE_ACCOUNT_DOMAIN"


def test_end_to_end_account_security_escalation(pipeline_components):
    clf, vstore, assessor, policy, generator, checker = pipeline_components
    query = "Someone in Russia hacked my account and changed the email address"
    res = run_agent_pipeline(query, clf, vstore, assessor, policy, generator, checker)

    assert res["intent"]["predicted"] == "account_access_security"
    assert res["decision"]["action"] == "ESCALATE"
    assert res["decision"]["reason_code"] == "SENSITIVE_ACCOUNT_DOMAIN"
