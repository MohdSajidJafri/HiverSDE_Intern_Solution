"""
Unit tests for evaluation mathematics:
- Hit@1, Hit@3, and MRR (Mean Reciprocal Rank) across top-ranked evidence
- Multiclass Brier score hand-calculated correctness
- Multiclass Expected Calibration Error (ECE)
- Weight derivation and distribution reporting
"""

import pytest
import numpy as np
from src.hiver_agent.evaluation.harness import EvaluationHarness
from src.hiver_agent.nlp.calibration import MulticlassTemperatureScaler


def test_mrr_and_hit_at_k_exact_hand_calculation():
    """
    Verify Hit@1, Hit@3, and MRR calculations under the Intent-Consistent Retrieval Relevance Proxy.
    
    Setup: 4 samples with query true_intent = 'playback_issues'
    Sample 1: Rank 1 matches intent & non-empty reply -> First relevant at rank 1 (RR = 1.0, Hit@1=1, Hit@3=1)
    Sample 2: Rank 1 diff intent, Rank 2 matches intent & non-empty reply -> First relevant at rank 2 (RR = 0.5, Hit@1=0, Hit@3=1)
    Sample 3: Ranks 1 & 2 diff intent, Rank 3 matches intent & non-empty reply -> First relevant at rank 3 (RR = 1/3, Hit@1=0, Hit@3=1)
    Sample 4: No candidate in top 3 matches intent -> Not in top 3 (RR = 0.0, Hit@1=0, Hit@3=0)

    Expected:
    Proxy Hit@1 = 1 / 4 = 0.25
    Proxy Hit@3 = 3 / 4 = 0.75
    Proxy MRR = (1.0 + 0.5 + (1/3) + 0.0) / 4 = (11/6) / 4 = 11/24 ~= 0.458333...
    Rank distribution: {1: 1, 2: 1, 3: 1, 'not_in_top_3': 1}
    """
    intents = ["playback_issues", "app_crash_technical", "subscription_billing"]
    harness = EvaluationHarness(intents=intents)

    gold = [
        {"true_intent": "playback_issues", "ground_truth_decision": "AUTO_HANDLE"},
        {"true_intent": "playback_issues", "ground_truth_decision": "AUTO_HANDLE"},
        {"true_intent": "playback_issues", "ground_truth_decision": "AUTO_HANDLE"},
        {"true_intent": "playback_issues", "ground_truth_decision": "AUTO_HANDLE"},
    ]

    preds = [
        {
            "intent": {"predicted": "playback_issues", "confidence": 0.9},
            "decision": {"action": "AUTO_HANDLE"},
            "retrieval": {
                "evidence": [
                    {"evidence_id": "e1", "similarity": 0.80, "intent": "playback_issues", "historical_brand_reply": "Restart your Spotify player to restore audio."},
                    {"evidence_id": "e2", "similarity": 0.50, "intent": "playback_issues", "historical_brand_reply": "Clear your local cache."},
                ]
            }
        },
        {
            "intent": {"predicted": "playback_issues", "confidence": 0.9},
            "decision": {"action": "AUTO_HANDLE"},
            "retrieval": {
                "evidence": [
                    {"evidence_id": "e1", "similarity": 0.85, "intent": "app_crash_technical", "historical_brand_reply": "Reinstall the application from the app store."},
                    {"evidence_id": "e2", "similarity": 0.70, "intent": "playback_issues", "historical_brand_reply": "Restart your bluetooth device and retry."},
                ]
            }
        },
        {
            "intent": {"predicted": "playback_issues", "confidence": 0.9},
            "decision": {"action": "AUTO_HANDLE"},
            "retrieval": {
                "evidence": [
                    {"evidence_id": "e1", "similarity": 0.75, "intent": "subscription_billing", "historical_brand_reply": "Check your receipt in account settings."},
                    {"evidence_id": "e2", "similarity": 0.65, "intent": "app_crash_technical", "historical_brand_reply": "Update to the latest OS version."},
                    {"evidence_id": "e3", "similarity": 0.60, "intent": "playback_issues", "historical_brand_reply": "Disable hardware acceleration in settings."},
                ]
            }
        },
        {
            "intent": {"predicted": "playback_issues", "confidence": 0.9},
            "decision": {"action": "AUTO_HANDLE"},
            "retrieval": {
                "evidence": [
                    {"evidence_id": "e1", "similarity": 0.70, "intent": "subscription_billing", "historical_brand_reply": "Your subscription has been renewed."},
                    {"evidence_id": "e2", "similarity": 0.60, "intent": "subscription_billing", "historical_brand_reply": "Please check your bank statement."},
                    {"evidence_id": "e3", "similarity": 0.40, "intent": "app_crash_technical", "historical_brand_reply": "Force stop the app."},
                ]
            }
        }
    ]

    metrics = harness.evaluate_predictions(gold, preds)
    retrieval = metrics["evidence_retrieval"]

    # Test Proxy metrics
    proxy_res = retrieval["intent_consistent_proxy"]
    assert proxy_res["proxy_hit_at_1"] == pytest.approx(0.25, abs=1e-4)
    assert proxy_res["proxy_hit_at_3"] == pytest.approx(0.75, abs=1e-4)
    assert proxy_res["proxy_mean_reciprocal_rank"] == pytest.approx(11.0 / 24.0, abs=1e-4)

    # Test rank distribution
    dist = proxy_res["first_relevant_rank_distribution"]
    assert dist["rank_1"] == 1
    assert dist["rank_2"] == 1
    assert dist["rank_3"] == 1
    assert dist["not_in_top_3"] == 1

    # Test backward-compatible aliases
    assert retrieval["hit_at_1"] == pytest.approx(0.25, abs=1e-4)
    assert retrieval["hit_at_3"] == pytest.approx(0.75, abs=1e-4)
    assert retrieval["mean_reciprocal_rank"] == pytest.approx(11.0 / 24.0, abs=1e-4)

    # Test threshold coverage diagnostic
    diag = retrieval["threshold_coverage_diagnostic"]
    assert diag["threshold_applied"] == 0.45
    # All 4 samples have top1 >= 0.45 (0.80, 0.85, 0.75, 0.70)
    assert diag["top1_coverage"] == pytest.approx(1.0, abs=1e-4)


def test_brier_score_hand_calculation():
    """
    Verify multiclass Brier score matches standard definition:
    (1/N) * sum_i sum_k (p_{ik} - y_{ik})^2
    """
    # 2 samples, 3 classes
    # Sample 0: true class 0 -> y = [1, 0, 0], p = [0.8, 0.1, 0.1]
    # sum_sq_0 = (0.8 - 1)^2 + 0.1^2 + 0.1^2 = 0.04 + 0.01 + 0.01 = 0.06
    # Sample 1: true class 1 -> y = [0, 1, 0], p = [0.2, 0.7, 0.1]
    # sum_sq_1 = 0.2^2 + (0.7 - 1)^2 + 0.1^2 = 0.04 + 0.09 + 0.01 = 0.14
    # Mean = (0.06 + 0.14) / 2 = 0.1000

    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1]
    ])
    y_true = np.array([0, 1])

    brier = MulticlassTemperatureScaler.compute_brier_score(probs, y_true, n_classes=3)
    assert brier == pytest.approx(0.10, abs=1e-5)


def test_ece_perfect_calibration():
    """
    When confidence exactly equals empirical accuracy in every bin, ECE should be 0.0.
    """
    # 10 samples all predicted class 0 with confidence 1.0, and all true class 0
    probs = np.zeros((10, 3))
    probs[:, 0] = 1.0
    y_true = np.zeros(10, dtype=int)

    ece, bins = MulticlassTemperatureScaler.compute_ece(probs, y_true, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-5)
