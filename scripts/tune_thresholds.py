"""
Threshold calibration, trade-off curve analysis, and System Freeze script.
Tunes operating thresholds on validation split and freezes all parameters into
models/freeze_manifest.json before final gold evaluation.
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
    return "initial_prototype_uncommitted"


def main():
    print("=" * 75)
    print("VALIDATION THRESHOLD TUNING & SYSTEM FREEZE STAGE")
    print("=" * 75)

    val_path = project_root / "data" / "val" / "dev_tuning.jsonl"
    with open(val_path, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f]
    print(f"Loaded {len(val_records)} validation tuning records.")

    # Sweep threshold grid
    conf_candidates = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    quality_candidates = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    n_total = len(val_records)
    grid_results = []

    print("\nEvaluating Automation-vs-Safety Trade-Off Grid:")
    print("-" * 80)
    print(f"{'Conf Th':<10} {'Qual Th':<10} {'False Auto %':<16} {'Safe Coverage %':<18} {'Escalation %':<14}")
    print("-" * 80)

    for conf_th in conf_candidates:
        for quality_th in quality_candidates:
            false_auto_handle = 0
            safe_auto_handle = 0
            escalated = 0

            for r in val_records:
                gt = r["ground_truth_decision"]
                conf = r["calibrated_confidence"]
                qual = r["evidence_quality"]
                contra = r["has_contradiction"]
                outlier = r["is_outlier"]
                sensitive = r["is_sensitive"]

                if sensitive or contra or outlier or (conf < conf_th) or (qual < quality_th):
                    pred_dec = "ESCALATE"
                else:
                    pred_dec = "AUTO_HANDLE"

                if pred_dec == "ESCALATE":
                    escalated += 1
                else:
                    if gt == "ESCALATE":
                        false_auto_handle += 1
                    else:
                        safe_auto_handle += 1

            fah_rate = false_auto_handle / n_total
            safe_cov = safe_auto_handle / n_total
            esc_rate = escalated / n_total

            entry = {
                "conf_threshold": conf_th,
                "quality_threshold": quality_th,
                "false_auto_handle_rate": round(fah_rate, 4),
                "safe_auto_handle_coverage": round(safe_cov, 4),
                "escalation_rate": round(esc_rate, 4)
            }
            grid_results.append(entry)

    # Print representative sample of grid
    for r in grid_results[::6]:
        print(
            f"{r['conf_threshold']:<10.2f} {r['quality_threshold']:<10.2f} "
            f"{r['false_auto_handle_rate']*100:<15.1f}% {r['safe_auto_handle_coverage']*100:<17.1f}% "
            f"{r['escalation_rate']*100:<13.1f}%"
        )
    print("-" * 80)

    # Optimal operating point selection:
    # Strict safety constraint: False Auto-Handle Rate must be 0.0% on validation set
    valid_pts = [r for r in grid_results if r["false_auto_handle_rate"] == 0.0]
    if valid_pts:
        optimal = max(valid_pts, key=lambda x: x["safe_auto_handle_coverage"])
    else:
        optimal = min(grid_results, key=lambda x: x["false_auto_handle_rate"])

    print(f"\nOptimal Frozen Operating Point:")
    print(f"  Confidence Threshold: {optimal['conf_threshold']}")
    print(f"  Evidence Quality Threshold: {optimal['quality_threshold']}")
    print(f"  False Auto-Handle Rate: {optimal['false_auto_handle_rate']*100:.1f}%")
    print(f"  Safe Auto-Handle Coverage: {optimal['safe_auto_handle_coverage']*100:.1f}%")
    print(f"  Escalation Rate: {optimal['escalation_rate']*100:.1f}%")

    # Save threshold tuning results
    reports_dir = project_root / "reports" / "results"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tuning_file = reports_dir / "threshold_tuning_results.json"
    with open(tuning_file, "w", encoding="utf-8") as f:
        json.dump({"grid": grid_results, "optimal": optimal}, f, indent=2)
    print(f"Saved threshold tuning results to {tuning_file}")

    # GENERATE COMPREHENSIVE FREEZE MANIFEST
    print("\nLocking system parameters and generating models/freeze_manifest.json...")
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = models_dir / "freeze_manifest.json"

    app_config = AppConfig()

    manifest_data = {
        "freeze_status": "FROZEN",
        "freeze_timestamp": "2026-09-09T14:55:00Z",
        "git_commit_sha": get_git_commit_sha(),
        "dataset_provenance": {
            "source": "Customer Support on Twitter (twcs.csv)",
            "source_url": app_config.data.raw_dataset_url,
            "selected_brand": "SpotifyCares"
        },
        "file_checksums_sha256": {
            "gold_messages_jsonl": get_file_sha256(project_root / "data" / "gold" / "gold_messages.jsonl"),
            "dev_tuning_jsonl": get_file_sha256(project_root / "data" / "val" / "dev_tuning.jsonl"),
            "retrieval_corpus_jsonl": get_file_sha256(project_root / "data" / "processed" / "retrieval_corpus.jsonl"),
            "retrieval_index_pkl": get_file_sha256(project_root / "models" / "retrieval_index.pkl"),
            "config_yaml": get_file_sha256(project_root / "config.yaml")
        },
        "model_specifications": {
            "embedding_model_name": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "classifier_architecture": "SentenceTransformer + Multinomial Logistic Regression (L-BFGS) + Multiclass Temperature Scaling + Centroid Outlier Detector",
            "calibrated_temperature_t": 1.15,
            "llm_provider": "deterministic_grounded_offline_and_gemini_api_dual",
            "prompt_template_version": "v1.0-grounded-audit"
        },
        "frozen_thresholds": {
            "intent_confidence_threshold": optimal["conf_threshold"],
            "evidence_quality_threshold": optimal["quality_threshold"],
            "centroid_novelty_threshold": 0.45,
            "leakage_screening_threshold": 0.92
        },
        "policy_configuration": {
            "sensitive_intents": app_config.escalation.sensitive_intents,
            "escalate_on_contradiction": True,
            "escalate_on_outlier": True,
            "escalate_on_low_confidence": True,
            "escalate_on_low_evidence": True
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"System permanently frozen. Manifest written to {manifest_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
