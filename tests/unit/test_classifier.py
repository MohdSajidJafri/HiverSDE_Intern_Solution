"""
Unit tests for IntentClassifier.
"""

import pytest
from src.hiver_agent.nlp.classifier import IntentClassifier


def test_classifier_training_and_prediction():
    train_texts = [
        "App crashes when launching",
        "Freezes on startup immediately",
        "Music stops playing after 5 seconds",
        "Songs pause automatically and skip",
        "Charged twice for premium subscription",
        "Billed 9.99 twice this month"
    ]
    train_labels = [
        "app_crash_technical",
        "app_crash_technical",
        "playback_issues",
        "playback_issues",
        "subscription_billing",
        "subscription_billing"
    ]

    val_texts = [
        "Desktop app crash on launch",
        "Playback stopped randomly",
        "Unexpected renewal fee charged"
    ]
    val_labels = [
        "app_crash_technical",
        "playback_issues",
        "subscription_billing"
    ]

    clf = IntentClassifier()
    clf.fit(train_texts, train_labels, val_texts, val_labels)

    assert clf.is_trained is True
    assert len(clf.classes_) == 3

    # Test in-distribution prediction
    res = clf.predict_one("My phone app keeps crashing every time I start it")
    assert res["predicted_intent"] == "app_crash_technical"
    assert 0.0 <= res["calibrated_confidence"] <= 1.0
    assert res["is_novelty_outlier"] is False

    # Test novelty outlier detection (completely unrelated topic)
    outlier_res = clf.predict_one("How do I plant tomatoes in my garden during spring?")
    # Novelty distance to streaming support centroids should be high
    assert outlier_res["min_centroid_distance"] > 0.35
