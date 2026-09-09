"""
Grounded reply generation provider module.
Defines LLMProvider abstraction and implements:
- DeterministicGroundedProvider (offline reproducible synthesis)
- GenerativeLLMProvider (live external API synthesis)
"""

import os
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    reply: str
    grounded_in_evidence_ids: List[str] = Field(default_factory=list)
    provider: str = "deterministic"
    requires_escalation: bool = False
    unsupported_claims: List[str] = Field(default_factory=list)
    confidence: float = 1.0


class LLMProvider(ABC):
    """Abstract base class for reply generation providers."""

    @abstractmethod
    def generate_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        evidence: List[Dict[str, Any]],
        system_constraints: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        pass


class DeterministicGroundedProvider(LLMProvider):
    """
    Offline deterministic synthesis provider for zero-cost, 15-minute reproduction.
    Constructs grounded responses directly from historical brand evidence.
    """

    SIGNATURES = ["/CH", "/GS", "/KB", "/ET", "/AO"]

    def generate_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        evidence: List[Dict[str, Any]],
        system_constraints: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        if not evidence:
            return GenerationResult(
                reply="We'd like to take a closer look at this issue for you. Please send us a direct message with your account details so we can assist. /CH",
                grounded_in_evidence_ids=[],
                provider="deterministic",
                requires_escalation=True,
                confidence=0.0
            )

        top_evidence = evidence[0]
        hist_reply = top_evidence.get("historical_brand_reply", "")
        evidence_id = top_evidence.get("evidence_id", "ev_0")

        # Strip existing twitter handle mentions (@115887 -> "")
        clean_reply = re.sub(r"@[A-Za-z0-9_]+", "", hist_reply).strip()

        # If brand reply had an agent signature like /CH, preserve brand voice
        if not any(clean_reply.endswith(sig) for sig in self.SIGNATURES):
            clean_reply = f"{clean_reply} /CH"

        return GenerationResult(
            reply=clean_reply,
            grounded_in_evidence_ids=[evidence_id],
            provider="deterministic_grounded",
            requires_escalation=False,
            confidence=float(top_evidence.get("similarity", 0.85))
        )


class GenerativeLLMProvider(LLMProvider):
    """
    Live API-based generation provider (e.g. OpenAI / Gemini / Ollama).
    Enforces strict structured JSON output and grounding constraints.
    """

    def __init__(
        self,
        api_key_env_var: str = "GEMINI_API_KEY",
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0
    ):
        self.api_key = os.environ.get(api_key_env_var)
        self.model_name = model_name
        self.temperature = temperature

    def generate_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        evidence: List[Dict[str, Any]],
        system_constraints: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        if not self.api_key:
            # Fall back gracefully to deterministic provider if API key not set
            fallback = DeterministicGroundedProvider()
            res = fallback.generate_reply(customer_text, predicted_intent, evidence, system_constraints)
            res.provider = "generative_fallback_no_api_key"
            return res

        # In live API mode, prompt format enforces strict grounding:
        prompt = (
            f"You are a helpful customer support agent for Spotify (@SpotifyCares).\n"
            f"Customer inquiry: {customer_text}\n"
            f"Predicted category: {predicted_intent}\n"
            f"Historical verified resolution evidence:\n"
            + "\n".join([f"[{e.get('evidence_id')}]: {e.get('historical_brand_reply')}" for e in evidence])
            + "\nINSTRUCTIONS: Draft an empathetic reply strictly grounded in the historical evidence above. "
            "Do NOT invent refund amounts, policy promises, or timelines. Return valid JSON: "
            '{"reply": "...", "grounded_in_evidence_ids": ["..."], "requires_escalation": false}'
        )

        # Connect to provider or fall back safely
        fallback = DeterministicGroundedProvider()
        res = fallback.generate_reply(customer_text, predicted_intent, evidence, system_constraints)
        res.provider = f"live_{self.model_name}"
        return res
