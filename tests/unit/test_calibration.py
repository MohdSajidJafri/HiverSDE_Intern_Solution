"""
Unit tests for MulticlassTemperatureScaler.
"""

import pytest
import numpy as np
from src.hiver_agent.nlp.calibration import MulticlassTemperatureScaler


def test_temperature_scaling_optimization():
    # Synthetic overconfident logits
    np.random.seed(42)
    n_samples = 100
    n_classes = 5

    # True labels
    labels = np.random.randint(0, n_classes, size=n_samples)

    # Overconfident logits: scaled by 5.0
    logits = np.random.randn(n_samples, n_classes) * 5.0
    # Make correct class logit slightly higher
    for i in range(n_samples):
        logits[i, labels[i]] += 2.0

    scaler = MulticlassTemperatureScaler()
    scaler.fit(logits, labels)

    assert scaler.is_fitted is True
    # For overconfident logits, optimal T should be > 1.0 to soften probabilities
    assert scaler.temperature > 1.0

    calibrated_probs = scaler.predict_proba(logits)
    assert calibrated_probs.shape == (n_samples, n_classes)
    np.testing.assert_allclose(np.sum(calibrated_probs, axis=1), 1.0, atol=1e-5)


def test_ece_and_brier_computation():
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.1, 0.9]
    ])
    labels = np.array([0, 0, 1, 1])  # 100% accuracy

    ece, bins_data = MulticlassTemperatureScaler.compute_ece(probs, labels, n_bins=5)
    brier = MulticlassTemperatureScaler.compute_brier_score(probs, labels, n_classes=2)

    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 2.0
    assert len(bins_data) == 5
