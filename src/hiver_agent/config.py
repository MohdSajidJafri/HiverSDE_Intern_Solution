"""
Configuration management for Hiver Brand AI Support Agent.
Externalizes all runtime options, thresholds, and paths.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import os
import json
import yaml


class DataConfig(BaseModel):
    raw_dataset_url: str = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    gold_data_path: Path = Path("data/gold/gold_messages.jsonl")
    val_data_path: Path = Path("data/val/dev_tuning.jsonl")
    interim_dir: Path = Path("data/interim")
    selected_brand: str = "SpotifyCares"  # Provisional until profiling executes
    random_seed: int = 42


class ModelConfig(BaseModel):
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    models_dir: Path = Path("models")
    freeze_manifest_path: Path = Path("models/freeze_manifest.json")
    classifier_path: Path = Path("models/intent_classifier.pkl")
    vector_index_path: Path = Path("models/retrieval_index.pkl")


class ThresholdConfig(BaseModel):
    # Authoritative frozen intent confidence threshold
    intent_confidence_threshold: float = 0.45
    # Authoritative frozen evidence quality score threshold
    evidence_quality_threshold: float = 0.45
    # Novelty / outlier cosine distance threshold from class centroid
    centroid_novelty_threshold: float = 0.45
    # Leakage screening threshold (cosine similarity)
    leakage_screening_threshold: float = 0.92


class EscalationConfig(BaseModel):
    # Classes that unconditionally trigger human escalation
    sensitive_intents: list[str] = Field(default_factory=lambda: [
        "subscription_billing",
        "account_access_security"
    ])
    escalate_on_contradiction: bool = True
    escalate_on_outlier: bool = True
    escalate_on_low_confidence: bool = True
    escalate_on_low_evidence: bool = True
    escalate_on_unsupported_claims: bool = True


class GenerationConfig(BaseModel):
    provider_type: str = "deterministic"  # 'deterministic' | 'external_api'
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.0
    max_tokens: int = 250
    api_key_env_var: str = "GEMINI_API_KEY"


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @classmethod
    def load_from_yaml(cls, path: Optional[Path] = None) -> "AppConfig":
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            return cls(**raw)
        return cls()

    @classmethod
    def load_authoritative(
        cls,
        yaml_path: Any = Path("config.yaml"),
        manifest_path: Any = Path("models/freeze_manifest.json")
    ) -> "AppConfig":
        """
        Loads the authoritative frozen configuration and strictly validates
        that config.yaml and freeze_manifest.json are synchronized.
        Fails loudly on any mismatch.
        """
        yaml_path = Path(yaml_path)
        manifest_path = Path(manifest_path)

        if not manifest_path.exists():
            raise FileNotFoundError(f"Authoritative freeze manifest missing at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        frozen_th = manifest_data.get("frozen_thresholds", {})
        m_conf_th = frozen_th.get("intent_confidence_threshold")
        m_qual_th = frozen_th.get("evidence_quality_threshold")

        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
            y_th = yaml_data.get("thresholds", {})
            y_conf_th = y_th.get("intent_confidence_threshold")
            y_qual_th = y_th.get("evidence_quality_threshold")

            if y_conf_th != m_conf_th or y_qual_th != m_qual_th:
                raise RuntimeError(
                    f"FATAL CONFIG DIVERGENCE DETECTED!\n"
                    f"  freeze_manifest.json: conf={m_conf_th}, qual={m_qual_th}\n"
                    f"  config.yaml:         conf={y_conf_th}, qual={y_qual_th}\n"
                    f"Runtime thresholds must strictly match authoritative frozen manifest."
                )
            base_config = cls(**yaml_data)
        else:
            base_config = cls()

        base_config.thresholds.intent_confidence_threshold = m_conf_th
        base_config.thresholds.evidence_quality_threshold = m_qual_th
        return base_config

    def save_to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(mode="json"), f, indent=2)


# Authoritative runtime configuration instance
try:
    default_config = AppConfig.load_authoritative()
except Exception:
    default_config = AppConfig()

