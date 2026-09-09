"""
Unit tests for reply generation and claim verification.
"""

import pytest
from src.hiver_agent.generation.provider import DeterministicGroundedProvider, GenerativeLLMProvider
from src.hiver_agent.generation.hallucination_checker import HallucinationChecker, ClaimSupportStatus


def test_deterministic_grounded_provider():
    provider = DeterministicGroundedProvider()
    evidence = [
        {
            "evidence_id": "ev_102",
            "historical_brand_reply": "@115712 Try restarting your device by holding sleep/wake button /CH",
            "similarity": 0.88
        }
    ]

    res = provider.generate_reply("Music stops playing", "playback_issues", evidence)
    assert res.requires_escalation is False
    assert res.grounded_in_evidence_ids == ["ev_102"]
    assert "restarting your device" in res.reply
    assert "@115712" not in res.reply  # handle stripped


def test_hallucination_checker_supported_vs_unsupported():
    checker = HallucinationChecker()
    evidence = [
        {
            "evidence_id": "ev_102",
            "historical_brand_reply": "Try a clean reinstall of the app from our website.",
            "historical_customer": "Desktop app won't open."
        }
    ]

    # Grounded reply
    grounded_reply = "We suggest trying a clean reinstall of the app from our website. Let us know how it goes!"
    ver_grounded = checker.verify_claims(grounded_reply, evidence)
    assert ver_grounded["unsupported_claims"] == 0
    assert ver_grounded["is_fully_grounded"] is True

    # Hallucinated monetary claim
    hallucinated_reply = "We apologize for the trouble. We have issued a $50 refund to your account and credited your card."
    ver_hallucinated = checker.verify_claims(hallucinated_reply, evidence)
    assert ver_hallucinated["unsupported_claims"] > 0
    assert ver_hallucinated["is_fully_grounded"] is False
