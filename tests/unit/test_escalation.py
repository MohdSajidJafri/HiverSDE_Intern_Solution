"""
Unit tests for EscalationPolicy.
"""

import pytest
from src.hiver_agent.policy.escalation import EscalationPolicy


def test_escalation_sensitive_intent():
    policy = EscalationPolicy()
    intent_res = {
        "predicted_intent": "subscription_billing",
        "calibrated_confidence": 0.95,
        "is_novelty_outlier": False
    }
    evidence_res = {
        "evidence_quality_score": 0.90,
        "has_contradiction": False
    }

    decision = policy.evaluate(intent_res, evidence_res)
    assert decision.action == "ESCALATE"
    assert decision.reason_code == "SENSITIVE_ACCOUNT_DOMAIN"


def test_escalation_contradictory_evidence():
    policy = EscalationPolicy()
    intent_res = {
        "predicted_intent": "playback_issues",
        "calibrated_confidence": 0.90,
        "is_novelty_outlier": False
    }
    evidence_res = {
        "evidence_quality_score": 0.80,
        "has_contradiction": True,
        "contradiction_explanation": "Candidate 1 contradicts Candidate 2"
    }

    decision = policy.evaluate(intent_res, evidence_res)
    assert decision.action == "ESCALATE"
    assert decision.reason_code == "CONTRADICTORY_HISTORICAL_EVIDENCE"


def test_escalation_low_confidence():
    policy = EscalationPolicy()
    intent_res = {
        "predicted_intent": "playback_issues",
        "calibrated_confidence": 0.35,  # below authoritative 0.45 threshold
        "is_novelty_outlier": False
    }
    evidence_res = {
        "evidence_quality_score": 0.85,
        "has_contradiction": False
    }

    decision = policy.evaluate(intent_res, evidence_res)
    assert decision.action == "ESCALATE"
    assert decision.reason_code == "LOW_INTENT_CONFIDENCE"


def test_escalation_novelty_outlier():
    policy = EscalationPolicy()
    intent_res = {
        "predicted_intent": "playback_issues",
        "calibrated_confidence": 0.80,
        "is_novelty_outlier": True
    }
    evidence_res = {
        "evidence_quality_score": 0.80,
        "has_contradiction": False
    }

    decision = policy.evaluate(intent_res, evidence_res)
    assert decision.action == "ESCALATE"
    assert decision.reason_code == "OUT_OF_SCOPE_ANOMALY"


def test_auto_handle_clean_case():
    policy = EscalationPolicy()
    intent_res = {
        "predicted_intent": "playback_issues",
        "calibrated_confidence": 0.88,
        "is_novelty_outlier": False
    }
    evidence_res = {
        "evidence_quality_score": 0.85,
        "has_contradiction": False
    }
    claim_res = {
        "unsupported_claims": 0
    }

    decision = policy.evaluate(intent_res, evidence_res, claim_res)
    assert decision.action == "AUTO_HANDLE"
    assert decision.reason_code == "HIGH_CONFIDENCE_GROUNDED"
    assert decision.risk_score < 0.30
