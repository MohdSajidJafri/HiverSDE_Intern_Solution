"""
Trains and calibrates the IntentClassifier on the clean historical Spotify corpus.
Supports:
- Human-labelled training data when available (data/train/gold_training_data.jsonl)
- Transparent silver/pseudo-labelled training data when human labels are unavailable
- Quarantined validation split (data/val/dev_tuning.jsonl) for temperature scaling
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


def assign_silver_training_intent(text: str) -> str:
    """
    Transparent rule-based heuristic assignment for training corpus based on discovered taxonomy keywords.
    Explicitly documented as silver pseudo-labeling, NOT supervised human gold.
    """
    t = text.lower()
    if any(k in t for k in ["crash", "freeze", "black screen", "close", "quit", "install", "corrupted", "bug", "w10m"]):
        return "app_crash_technical"
    if any(k in t for k in ["charged", "billing", "bill", "refund", "subscription", "payment", "card", "paypal", "receipt", "99"]):
        return "subscription_billing"
    if any(k in t for k in ["hacked", "password", "email", "login", "stolen", "account access", "locked", "reset"]):
        return "account_access_security"
    if any(k in t for k in ["offline", "download", "downloaded", "airplane", "greyed", "sync"]):
        return "offline_downloads"
    if any(k in t for k in ["bluetooth", "echo", "connect", "carplay", "chromecast", "speaker", "soundbar", "ps4", "alexa"]):
        return "device_connectivity"
    if any(k in t for k in ["playlist", "library", "saved", "liked", "disappeared", "album", "unliked", "songs disappeared"]):
        return "playlist_library"
    if any(k in t for k in ["lyrics", "filter", "ui", "clean", "explicit", "feature request", "bring back", "look"]):
        return "feature_request_ui"
    if any(k in t for k in ["down", "outage", "500", "502", "status", "server error", "maintenance"]):
        return "service_status_outage"
    if any(k in t for k in ["pause", "skip", "shuffle", "stutter", "play", "stop", "volume", "audio", "sound"]):
        return "playback_issues"
    return "other_unsupported"


def main():
    print("=" * 75)
    print("TRAINING & CALIBRATING INTENT CLASSIFIER (PROVENANCE AUDITED)")
    print("=" * 75)

    gold_train_path = project_root / "data" / "train" / "gold_training_data.jsonl"
    silver_train_path = project_root / "data" / "processed" / "silver_training_data.jsonl"
    retrieval_path = project_root / "data" / "processed" / "retrieval_corpus.jsonl"

    if gold_train_path.exists():
        print(f"Found human-labeled training set: {gold_train_path}")
        with open(gold_train_path, "r", encoding="utf-8") as f:
            train_records = [json.loads(line) for line in f]
        train_texts = [r["customer_text"] for r in train_records]
        train_labels = [r["gold_intent"] for r in train_records]
        training_tier = "HUMAN_GOLD"
    else:
        print(f"Human training data not present. Constructing explicit silver pseudo-labeled training split...")
        with open(retrieval_path, "r", encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]

        train_records = []
        for p in pairs:
            c_text = p["customer_text"]
            s_label = assign_silver_training_intent(c_text)
            train_records.append({
                "customer_tweet_id": p.get("customer_tweet_id", ""),
                "customer_text": c_text,
                "silver_pseudo_label": s_label,
                "label_provenance": "taxonomy_rules_silver",
                "is_human_annotated": False
            })

        # Persist silver training split explicitly
        silver_train_path.parent.mkdir(parents=True, exist_ok=True)
        with open(silver_train_path, "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(train_records):,} explicit silver training records to {silver_train_path}")

        train_texts = [r["customer_text"] for r in train_records]
        train_labels = [r["silver_pseudo_label"] for r in train_records]
        training_tier = "SILVER_PSEUDO_LABELED"

    print(f"Training Corpus Size: {len(train_texts):,} records | Tier: {training_tier}")

    # Load quarantined validation split for multiclass temperature scaling calibration
    val_path = project_root / "data" / "val" / "dev_tuning.jsonl"
    with open(val_path, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f]

    val_texts = [r["customer_text"] for r in val_records]
    val_labels = [r.get("silver_intent", assign_silver_training_intent(r["customer_text"])) for r in val_records]
    print(f"Quarantined Validation Split Size: {len(val_texts):,} records")

    # Fit Classifier
    clf = IntentClassifier()
    print("\nEncoding dense embeddings and fitting Multinomial Logistic Regression...")
    clf.fit(train_texts, train_labels, val_texts, val_labels)

    fitted_t = float(clf.scaler.temperature)
    print(f"\nMulticlass Temperature Scaling Completed:")
    print(f"  Fitted Temperature (T): {fitted_t:.4f}")
    print(f"  Number of Classes: {len(clf.classes_)}")
    print(f"  Classes: {clf.classes_}")

    out_path = project_root / "models" / "intent_classifier.pkl"
    clf.save(out_path)
    print(f"\nSaved calibrated classifier artifact to {out_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
