"""
Configuration management for Hiver Brand AI Support Agent.
Externalizes all runtime options, thresholds, and paths.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import os
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
    # Calibrated intent confidence threshold
    intent_confidence_threshold: float = 0.65
    # Evidence quality score threshold
    evidence_quality_threshold: float = 0.70
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


class GenerationConfig(BaseModel):
    provider_type: str = "deterministic"  # 'deterministic' | 'openai' | 'gemini'
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

    def save_to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(mode="json"), f, indent=2)


# Global default configuration instance
default_config = AppConfig()
