"""
Unit tests for EvidenceQualityAssessor.
"""

import pytest
from src.hiver_agent.retrieval.evidence_quality import EvidenceQualityAssessor, ContradictionStatus


def test_contradiction_detection_incompatible():
    assessor = EvidenceQualityAssessor()

    evidence = [
        {
            "similarity": 0.85,
            "historical_brand_reply": "Try a clean reinstall of the app from spotify.com/download /CH"
        },
        {
            "similarity": 0.82,
            "historical_brand_reply": "Do not reinstall the app! We have an ongoing server outage right now /GS"
        }
    ]

    assessment = assessor.assess_evidence("App crashes on startup", "app_crash_technical", evidence)
    assert assessment["contradiction_status"] == ContradictionStatus.INCOMPATIBLE.value
    assert assessment["has_contradiction"] is True
    assert assessment["evidence_quality_score"] < 0.50  # heavily penalized


def test_contradiction_detection_compatible():
    assessor = EvidenceQualityAssessor()

    evidence = [
        {
            "similarity": 0.88,
            "historical_brand_reply": "Can you try restarting your device by holding sleep/wake button? /CH"
        },
        {
            "similarity": 0.84,
            "historical_brand_reply": "Could you check your audio quality settings and toggle offline mode? /GS"
        }
    ]

    assessment = assessor.assess_evidence("Playback issues", "playback_issues", evidence)
    assert assessment["contradiction_status"] == ContradictionStatus.COMPATIBLE.value
    assert assessment["has_contradiction"] is False
    assert assessment["evidence_quality_score"] > 0.60


def test_contradiction_insufficient_evidence():
    assessor = EvidenceQualityAssessor()

    # Single generic evidence
    evidence = [
        {
            "similarity": 0.40,
            "historical_brand_reply": "Thanks!"
        }
    ]

    assessment = assessor.assess_evidence("Hello", "other_unsupported", evidence)
    assert assessment["contradiction_status"] == ContradictionStatus.INSUFFICIENT_EVIDENCE.value
