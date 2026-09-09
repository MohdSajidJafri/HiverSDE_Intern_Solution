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
    Verify Hit@1, Hit@3, and MRR calculations against exact hand-calculated expected values.
    
    Setup: 4 samples
    Sample 1: First relevant at rank 1 (sim 0.80 >= 0.45) -> RR = 1.0, Hit@1=1, Hit@3=1
    Sample 2: First relevant at rank 2 (sim 0.30 < 0.45, sim 0.70 >= 0.45) -> RR = 0.5, Hit@1=0, Hit@3=1
    Sample 3: First relevant at rank 3 (sim 0.20, sim 0.25, sim 0.60 >= 0.45) -> RR = 1/3, Hit@1=0, Hit@3=1
    Sample 4: No relevant in top 3 (all sims < 0.45) -> RR = 0.0, Hit@1=0, Hit@3=0

    Expected:
    Hit@1 = 1 / 4 = 0.25
    Hit@3 = 3 / 4 = 0.75
    MRR = (1.0 + 0.5 + (1/3) + 0.0) / 4 = (11/6) / 4 = 11/24 ~= 0.458333...
    """
    intents = ["playback_issues", "app_crash_technical"]
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
                    {"evidence_id": "e1", "similarity": 0.80},
                    {"evidence_id": "e2", "similarity": 0.50},
                ]
            }
        },
        {
            "intent": {"predicted": "playback_issues", "confidence": 0.9},
            "decision": {"action": "AUTO_HANDLE"},
            "retrieval": {
                "evidence": [
                    {"evidence_id": "e1", "similarity": 0.30},
                    {"evidence_id": "e2", "similarity": 0.70},
                ]
            }
        },
        {
            "intent": {"predicted": "playback_issues", "confidence": 0.9},
            "decision": {"action": "AUTO_HANDLE"},
            "retrieval": {
                "evidence": [
                    {"evidence_id": "e1", "similarity": 0.20},
                    {"evidence_id": "e2", "similarity": 0.25},
                    {"evidence_id": "e3", "similarity": 0.60},
                ]
            }
        },
        {
            "intent": {"predicted": "playback_issues", "confidence": 0.9},
            "decision": {"action": "AUTO_HANDLE"},
            "retrieval": {
                "evidence": [
                    {"evidence_id": "e1", "similarity": 0.20},
                    {"evidence_id": "e2", "similarity": 0.35},
                    {"evidence_id": "e3", "similarity": 0.40},
                ]
            }
        }
    ]

    metrics = harness.evaluate_predictions(gold, preds)
    retrieval = metrics["evidence_retrieval"]

    assert retrieval["hit_at_1"] == pytest.approx(0.25, abs=1e-4)
    assert retrieval["hit_at_3"] == pytest.approx(0.75, abs=1e-4)
    assert retrieval["mean_reciprocal_rank"] == pytest.approx(11.0 / 24.0, abs=1e-4)


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
