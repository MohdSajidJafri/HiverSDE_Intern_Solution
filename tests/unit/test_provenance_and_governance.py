"""
Unit tests for data provenance, governance, and configuration authority:
- Authoritative config enforcement and fail-loud detection
- Gold annotation queue integrity (no fabricated annotations)
- Human-vs-judge agreement governance
- Provider identity and execution honesty
"""

import os
import json
import pytest
from src.hiver_agent.config import AppConfig
from src.hiver_agent.generation.provider import DeterministicGroundedProvider, ExternalAPIProvider


def test_authoritative_config_enforcement():
    """Verify that AppConfig.load_authoritative() successfully validates matching config."""
    cfg = AppConfig.load_authoritative("config.yaml", "models/freeze_manifest.json")
    assert cfg.thresholds.intent_confidence_threshold == 0.45
    assert cfg.thresholds.evidence_quality_threshold == 0.45


def test_authoritative_config_fail_loud_on_divergence(tmp_path):
    """Verify that load_authoritative() fails loudly (RuntimeError) if config diverged from manifest."""
    bad_manifest = tmp_path / "bad_manifest.json"
    manifest_data = {
        "frozen_thresholds": {
            "intent_confidence_threshold": 0.88,  # diverged from 0.45 in config.yaml
            "evidence_quality_threshold": 0.45
        },
        "classifier_artifacts": {
            "calibrated_temperature_t": 0.7820
        }
    }
    with open(bad_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    with pytest.raises(RuntimeError, match="FATAL CONFIG DIVERGENCE"):
        AppConfig.load_authoritative("config.yaml", str(bad_manifest))


def test_gold_annotation_queue_integrity():
    """Verify that gold queue has 200 records with valid human annotations."""
    queue_path = "data/gold/gold_annotation_queue.jsonl"
    assert os.path.exists(queue_path), f"Gold queue missing at {queue_path}"

    valid_intents = {
        "playback_issues", "app_crash_technical", "offline_downloads", "device_connectivity",
        "playlist_library", "subscription_billing", "account_access_security",
        "feature_request_ui", "service_status_outage", "other_unsupported"
    }

    count = 0
    with open(queue_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert rec["gold_intent"] in valid_intents, f"Invalid intent: {rec['gold_intent']}"
            assert rec["ground_truth_decision"] in ("AUTO_HANDLE", "ESCALATE")
            assert isinstance(rec["is_sensitive"], bool)
            assert "customer_tweet_id" in rec and rec["customer_tweet_id"]
            assert "customer_text" in rec and len(rec["customer_text"]) > 0
            assert "annotator" in rec and len(rec["annotator"]) > 0
            count += 1

    assert count == 200, f"Expected 200 gold candidate items, got {count}"


def test_human_vs_judge_governance_report():
    """Verify that human vs judge agreement is truthfully marked PENDING_HUMAN_ANNOTATION with 0 fabricated stats."""
    report_path = "reports/results/human_vs_judge_agreement.json"
    assert os.path.exists(report_path), f"Report missing at {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PENDING_HUMAN_ANNOTATION"
    assert data["agreement_metrics"] is None
    assert data["sample_size"] == 50
    assert data["completed_human_annotations"] == 0
    assert "queue_path" in data


def test_provider_honesty_and_fallback():
    """Verify provider reporting and transparent fallback behavior."""
    det_provider = DeterministicGroundedProvider()
    reply = det_provider.generate_reply("help", "playback_issues", [])
    assert reply.provider == "deterministic_grounded"

    # External provider without valid key must explicitly fall back
    ext_provider = ExternalAPIProvider(api_key="invalid_or_missing_test_key")
    ext_reply = ext_provider.generate_reply("help", "playback_issues", [])
    assert ext_reply.provider == "deterministic_grounded_fallback"
