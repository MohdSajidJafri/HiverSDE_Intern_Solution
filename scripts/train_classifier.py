"""
Trains and calibrates the IntentClassifier on the Spotify corpus.
Saves the trained model to models/intent_classifier.pkl.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.hiver_agent.nlp.classifier import IntentClassifier
from src.hiver_agent.nlp.normalizer import TextNormalizer


def assign_training_intent(text: str) -> str:
    """Heuristic labelling for training corpus based on discovered taxonomy keywords."""
    t = text.lower()
    if any(k in t for k in ["crash", "freeze", "black screen", "close", "helper quit", "install", "corrupted"]):
        return "app_crash_technical"
    if any(k in t for k in ["charged", "billing", "bill", "refund", "subscription", "99", "payment", "card", "paypal"]):
        return "subscription_billing"
    if any(k in t for k in ["hacked", "password", "email", "login", "stolen", "account access", "locked"]):
        return "account_access_security"
    if any(k in t for k in ["offline", "download", "downloaded", "airplane", "greyed", "sync"]):
        return "offline_downloads"
    if any(k in t for k in ["bluetooth", "echo", "connect", "carplay", "chromecast", "speaker", "soundbar", "ps4"]):
        return "device_connectivity"
    if any(k in t for k in ["playlist", "library", "saved", "liked", "disappeared", "album", "unliked"]):
        return "playlist_library"
    if any(k in t for k in ["lyrics", "filter", "ui", "clean", "explicit", "feature request", "bring back"]):
        return "feature_request_ui"
    if any(k in t for k in ["down", "outage", "500", "502", "status", "server error"]):
        return "service_status_outage"
    if any(k in t for k in ["pause", "skip", "shuffle", "stutter", "play", "stop", "volume", "audio"]):
        return "playback_issues"
    return "other_unsupported"


def main():
    print("=" * 75)
    print("TRAINING & CALIBRATING INTENT CLASSIFIER")
    print("=" * 75)

    retrieval_path = project_root / "data" / "processed" / "retrieval_corpus.jsonl"
    with open(retrieval_path, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    print(f"Loaded {len(pairs)} historical interaction pairs for training.")

    train_texts = [p["customer_text"] for p in pairs]
    train_labels = [assign_training_intent(t) for t in train_texts]

    # Load validation split for temperature scaling calibration
    val_path = project_root / "data" / "val" / "dev_tuning.jsonl"
    with open(val_path, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f]

    val_texts = [r["customer_text"] for r in val_records]
    val_labels = [assign_training_intent(t) for t in val_texts]

    clf = IntentClassifier()
    print("Generating dense embeddings and fitting Multinomial Logistic Regression...")
    clf.fit(train_texts, train_labels, val_texts, val_labels)

    out_path = project_root / "models" / "intent_classifier.pkl"
    clf.save(out_path)
    print(f"Saved calibrated classifier (T={clf.scaler.temperature:.2f}) to {out_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
