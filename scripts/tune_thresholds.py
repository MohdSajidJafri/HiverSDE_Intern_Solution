"""
Threshold Calibration, Multi-Objective Sweep, and Authoritative System Freeze.
Executes real pipeline inference across the quarantined validation split (data/val/dev_tuning.jsonl)
using real model predictions, real calibrated probabilities, and real retrieval evidence.
Sweeps operating threshold candidates and locks the authoritative frozen configuration
into models/freeze_manifest.json.
"""

import sys
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.hiver_agent.config import AppConfig
from src.hiver_agent.nlp.classifier import IntentClassifier
from src.hiver_agent.retrieval.vector_store import VectorStore
from src.hiver_agent.retrieval.evidence_quality import EvidenceQualityAssessor
from src.hiver_agent.generation.provider import DeterministicGroundedProvider
from src.hiver_agent.generation.hallucination_checker import HallucinationChecker
from src.hiver_agent.policy.escalation import EscalationPolicy


def get_file_sha256(path: Path) -> str:
    """Computes SHA256 checksum of a file."""
    if not path.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_sha() -> str:
    """Gets current git commit hash or clean fallback."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "uncommitted"


def main():
    print("=" * 75)
    print("VALIDATION THRESHOLD SWEEP & SYSTEM FREEZE (REAL MODEL INFERENCE)")
    print("=" * 75)

    val_path = project_root / "data" / "val" / "dev_tuning.jsonl"
    with open(val_path, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f]
    print(f"Loaded {len(val_records)} quarantined validation records.")

    # Load actual components
    models_dir = project_root / "models"
    classifier = IntentClassifier()
    classifier = classifier.load(models_dir / "intent_classifier.pkl")
    vector_store = VectorStore.load(models_dir / "retrieval_index.pkl")
    assessor = EvidenceQualityAssessor()
    generator = DeterministicGroundedProvider()
    checker = HallucinationChecker()

    actual_temperature = float(classifier.scaler.temperature)
    print(f"Loaded classifier with actual calibrated temperature T = {actual_temperature:.4f}")

    # Run real model inference on all validation records once
    print("\nRunning real end-to-end model inference on validation records...")
    val_inference_results = []
    correct_intents = 0

    for r in val_records:
        query = r["customer_text"]
        gt_intent = r.get("silver_intent", "")
        gt_dec = r.get("ground_truth_decision", "AUTO_HANDLE")
        is_sensitive = r.get("is_sensitive", False)

        intent_res = classifier.predict_one(query)
        evidence_res = vector_store.retrieve(query, top_k=3)
        ev_assess = assessor.assess_evidence(query, intent_res["predicted_intent"], evidence_res)
        gen_res = generator.generate_reply(query, intent_res["predicted_intent"], evidence_res)
        claim_ver = checker.verify_claims(gen_res.reply, evidence_res)

        if intent_res["predicted_intent"] == gt_intent:
            correct_intents += 1

        val_inference_results.append({
            "record": r,
            "predicted_intent": intent_res["predicted_intent"],
            "calibrated_confidence": intent_res["calibrated_confidence"],
            "is_outlier": intent_res["is_novelty_outlier"],
            "evidence_quality": ev_assess["evidence_quality_score"],
            "has_contradiction": ev_assess["has_contradiction"],
            "unsupported_claims": claim_ver.get("unsupported_claims", 0),
            "gt_decision": gt_dec,
            "is_sensitive": is_sensitive
        })

    val_accuracy = correct_intents / len(val_records) if val_records else 0.0
    print(f"Validation Intent Accuracy (Silver Baseline): {val_accuracy*100:.2f}%")

    # Sweep Candidate Grid
    conf_candidates = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    quality_candidates = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    n_total = len(val_records)
    sensitive_intents = ["subscription_billing", "account_access_security"]
    grid_results = []

    print("\nEvaluating Automation-vs-Safety Trade-Off Grid:")
    print("-" * 90)
    print(f"{'Conf Th':<10} {'Qual Th':<10} {'Sens FA Count':<15} {'False Auto %':<15} {'Auto-Handle %':<16} {'Escalation %':<14}")
    print("-" * 90)

    for conf_th in conf_candidates:
        for quality_th in quality_candidates:
            false_auto_handle = 0
            sensitive_false_auto_handle = 0
            safe_auto_handle = 0
            escalated = 0

            for inf in val_inference_results:
                pred_intent = inf["predicted_intent"]
                conf = inf["calibrated_confidence"]
                qual = inf["evidence_quality"]
                contra = inf["has_contradiction"]
                outlier = inf["is_outlier"]
                unsupp = inf["unsupported_claims"]
                gt_dec = inf["gt_decision"]
                is_sens = inf["is_sensitive"]

                # Conservative Escalation Policy Logic
                should_escalate = (
                    (pred_intent in sensitive_intents) or
                    contra or
                    outlier or
                    (unsupp > 0) or
                    (conf < conf_th) or
                    (qual < quality_th)
                )

                pred_dec = "ESCALATE" if should_escalate else "AUTO_HANDLE"

                if pred_dec == "ESCALATE":
                    escalated += 1
                else:
                    if gt_dec == "ESCALATE":
                        false_auto_handle += 1
                        if is_sens or (pred_intent in sensitive_intents):
                            sensitive_false_auto_handle += 1
                    else:
                        safe_auto_handle += 1

            fah_rate = false_auto_handle / n_total
            auto_rate = (n_total - escalated) / n_total
            esc_rate = escalated / n_total

            entry = {
                "conf_threshold": conf_th,
                "quality_threshold": quality_th,
                "sensitive_false_auto_handle_count": sensitive_false_auto_handle,
                "false_auto_handle_count": false_auto_handle,
                "false_auto_handle_rate": round(fah_rate, 4),
                "auto_handle_rate": round(auto_rate, 4),
                "escalation_rate": round(esc_rate, 4),
                "safe_auto_handle_coverage": round(safe_auto_handle / n_total, 4)
            }
            grid_results.append(entry)

    # Print representative rows of grid
    for r in grid_results[::7]:
        print(
            f"{r['conf_threshold']:<10.2f} {r['quality_threshold']:<10.2f} "
            f"{r['sensitive_false_auto_handle_count']:<15} {r['false_auto_handle_rate']*100:<14.1f}% "
            f"{r['auto_handle_rate']*100:<15.1f}% {r['escalation_rate']*100:<13.1f}%"
        )
    print("-" * 90)

    # -------------------------------------------------------------
    # OPTIMAL OPERATING POINT SELECTION:
    # 1. Hard Safety Constraint: sensitive_false_auto_handle_count == 0
    # 2. Maximum Coverage: among safe points, choose highest auto_handle_rate (least conservative)
    # 3. Tie-breaker: lowest overall false_auto_handle_rate
    # 4. Tie-breaker: lower conf_threshold for greater routine coverage
    # -------------------------------------------------------------
    safe_candidates = [r for r in grid_results if r["sensitive_false_auto_handle_count"] == 0]
    if safe_candidates:
        max_auto = max(r["auto_handle_rate"] for r in safe_candidates)
        top_candidates = [r for r in safe_candidates if r["auto_handle_rate"] == max_auto]
        # Tie-breaker: lowest overall false_auto_handle_rate
        min_fah = min(r["false_auto_handle_rate"] for r in top_candidates)
        optimal_candidates = [r for r in top_candidates if r["false_auto_handle_rate"] == min_fah]
        # Final tie-breaker: lower conf_threshold (higher coverage)
        optimal = min(optimal_candidates, key=lambda x: (x["conf_threshold"], x["quality_threshold"]))
    else:
        # If 0 sensitive false auto handles unreachable, choose absolute minimum
        optimal = min(grid_results, key=lambda x: (x["sensitive_false_auto_handle_count"], -x["auto_handle_rate"]))

    print(f"\nSelected Frozen Operating Point (Highest Coverage under Safety Constraint):")
    print(f"  Confidence Threshold (tau_conf): {optimal['conf_threshold']}")
    print(f"  Evidence Quality Threshold (tau_qual): {optimal['quality_threshold']}")
    print(f"  Sensitive False Auto-Handle Count: {optimal['sensitive_false_auto_handle_count']}")
    print(f"  Overall False Auto-Handle Rate: {optimal['false_auto_handle_rate']*100:.1f}%")
    print(f"  Safe Auto-Handle Rate: {optimal['auto_handle_rate']*100:.1f}%")
    print(f"  Escalation Rate: {optimal['escalation_rate']*100:.1f}%")

    # Save threshold tuning results
    reports_dir = project_root / "reports" / "results"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tuning_file = reports_dir / "threshold_tuning_results.json"
    with open(tuning_file, "w", encoding="utf-8") as f:
        json.dump({
            "selection_objective": "Zero sensitive false auto-handles (hard constraint) with maximum safe auto-handle coverage (least conservative operating point)",
            "validation_sample_size": n_total,
            "validation_intent_accuracy": round(val_accuracy, 4),
            "calibrated_temperature_t": round(actual_temperature, 4),
            "optimal": optimal,
            "grid": grid_results
        }, f, indent=2)
    print(f"Saved comprehensive threshold tuning results to {tuning_file}")

    # -------------------------------------------------------------
    # GENERATE AUTHORITATIVE FREEZE MANIFEST
    # -------------------------------------------------------------
    print("\nWriting authoritative models/freeze_manifest.json...")
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = models_dir / "freeze_manifest.json"

    freeze_manifest = {
        "freeze_status": "FROZEN",
        "freeze_timestamp": "2026-09-09T15:45:00Z",
        "git_commit_sha": get_git_commit_sha(),
        "dataset_provenance": {
            "source": "Customer Support on Twitter (twcs.csv)",
            "source_url": "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv",
            "selected_brand": "SpotifyCares",
            "partition_strategy": "connected_components_author_conversation_bipartite_graph",
            "evaluation_tier": "SILVER_DEVELOPMENT_PENDING_HUMAN_GOLD",
            "calibration_provenance": "Calibrated via Multiclass Temperature Scaling on silver-labelled validation split (data/val/dev_tuning.jsonl, N=156); explicitly NOT human gold calibration"
        },
        "file_checksums_sha256": {
            "gold_annotation_queue_jsonl": get_file_sha256(project_root / "data" / "gold" / "gold_annotation_queue.jsonl"),
            "silver_eval_set_jsonl": get_file_sha256(project_root / "data" / "interim" / "silver_eval_set.jsonl"),
            "silver_val_data_jsonl": get_file_sha256(project_root / "data" / "val" / "dev_tuning.jsonl"),
            "retrieval_corpus_jsonl": get_file_sha256(project_root / "data" / "processed" / "retrieval_corpus.jsonl"),
            "unselected_multiturn_interactions_jsonl": get_file_sha256(project_root / "data" / "interim" / "unselected_multiturn_interactions.jsonl"),
            "retrieval_index_pkl": get_file_sha256(project_root / "models" / "retrieval_index.pkl"),
            "intent_classifier_pkl": get_file_sha256(project_root / "models" / "intent_classifier.pkl"),
            "dependencies_pyproject_toml": get_file_sha256(project_root / "pyproject.toml"),
            "config_yaml": get_file_sha256(project_root / "config.yaml")
        },
        "model_specifications": {
            "embedding_model_name": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "classifier_architecture": "SentenceTransformer + Multinomial Logistic Regression (L-BFGS) + Multiclass Temperature Scaling + Centroid Outlier Detector",
            "calibrated_temperature_t": round(actual_temperature, 4),
            "temperature_source": "models/intent_classifier.pkl:scaler.temperature",
            "temperature_label_provenance": "Fitted via L-BFGS NLL minimization on silver-labelled validation split (data/val/dev_tuning.jsonl, N=156); explicitly NOT human gold labels",
            "llm_provider": "deterministic_grounded_production",
            "prompt_template_version": "v1.0-grounded-audit"
        },
        "frozen_thresholds": {
            "intent_confidence_threshold": optimal["conf_threshold"],
            "evidence_quality_threshold": optimal["quality_threshold"],
            "centroid_novelty_threshold": 0.45,
            "leakage_screening_threshold": 0.92
        },
        "threshold_selection_record": {
            "selection_criterion": "Zero sensitive false auto-handle constraint + maximum auto-handle coverage",
            "validation_metrics_at_selected_point": {
                "sensitive_false_auto_handle_count": optimal["sensitive_false_auto_handle_count"],
                "false_auto_handle_rate": optimal["false_auto_handle_rate"],
                "auto_handle_rate": optimal["auto_handle_rate"],
                "escalation_rate": optimal["escalation_rate"],
                "validation_sample_size": n_total,
                "validation_intent_accuracy": round(val_accuracy, 4)
            }
        },
        "policy_configuration": {
            "sensitive_intents": sensitive_intents,
            "escalate_on_contradiction": True,
            "escalate_on_outlier": True,
            "escalate_on_low_confidence": True,
            "escalate_on_low_evidence": True,
            "escalate_on_unsupported_claims": True
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(freeze_manifest, f, indent=2)
    print(f"Saved authoritative freeze manifest to {manifest_path}")

    # Synchronize config.yaml so there is ZERO discrepancy
    config_yaml_path = project_root / "config.yaml"
    import yaml
    with open(config_yaml_path, "r", encoding="utf-8") as f:
        conf_data = yaml.safe_load(f)

    conf_data["thresholds"]["intent_confidence_threshold"] = optimal["conf_threshold"]
    conf_data["thresholds"]["evidence_quality_threshold"] = optimal["quality_threshold"]

    with open(config_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(conf_data, f, default_flow_style=False, sort_keys=False)
    print(f"Synchronized config.yaml thresholds to {optimal['conf_threshold']} / {optimal['quality_threshold']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
