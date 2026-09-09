"""
Grounded reply generation provider module.
Defines LLMProvider abstraction and implements:
- DeterministicGroundedProvider (primary production/demo grounded synthesis provider)
- ExternalAPIProvider (real external HTTP API adapter with timeout, error handling, and explicit fallback)
"""

import os
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import httpx
from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    reply: str
    grounded_in_evidence_ids: List[str] = Field(default_factory=list)
    provider: str = "deterministic_grounded"
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
    Primary production/demo grounded synthesis provider.
    Constructs deterministic, evidence-grounded responses directly from historical brand precedent.
    Zero-cost, 100% offline reproducible without external API keys.
    """

    SIGNATURES = ["/CH", "/GS", "/KB", "/ET", "/AO", "/GK", "/PL", "/KM", "/JP", "/AR", "/LO", "/NS"]

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
                provider="deterministic_grounded",
                requires_escalation=True,
                confidence=0.0
            )

        top_evidence = evidence[0]
        hist_reply = top_evidence.get("historical_brand_reply", "")
        evidence_id = str(top_evidence.get("evidence_id", "ev_0"))

        # Strip existing twitter handle mentions (@115887 -> "")
        clean_reply = re.sub(r"@[A-Za-z0-9_]+", "", hist_reply).strip()

        # Preserve authentic Spotify customer support sign-off
        if not any(clean_reply.endswith(sig) for sig in self.SIGNATURES):
            clean_reply = f"{clean_reply} /CH"

        return GenerationResult(
            reply=clean_reply,
            grounded_in_evidence_ids=[evidence_id],
            provider="deterministic_grounded",
            requires_escalation=False,
            confidence=float(top_evidence.get("similarity", 0.85))
        )


class ExternalAPIProvider(LLMProvider):
    """
    Real external API generation adapter (e.g. Gemini / OpenAI).
    Performs real network HTTP requests when API key is configured.
    If API key is missing or request fails, falls back gracefully to
    DeterministicGroundedProvider and transparently reports 'deterministic_grounded_fallback'.
    Never reports 'live' if a live API call did not succeed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_key_env_var: str = "GEMINI_API_KEY",
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        timeout_seconds: float = 15.0
    ):
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env_var)
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.fallback_provider = DeterministicGroundedProvider()

    def generate_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        evidence: List[Dict[str, Any]],
        system_constraints: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        if not self.api_key:
            # Fall back to deterministic provider if API key not set
            res = self.fallback_provider.generate_reply(customer_text, predicted_intent, evidence, system_constraints)
            res.provider = "deterministic_grounded_fallback"
            return res

        # Format prompt with explicit evidence IDs
        prompt = (
            f"You are a helpful customer support agent for Spotify (@SpotifyCares).\n"
            f"Customer inquiry: {customer_text}\n"
            f"Predicted category: {predicted_intent}\n"
            f"Historical verified resolution evidence:\n"
            + "\n".join([f"[{e.get('evidence_id')}]: {e.get('historical_brand_reply')}" for e in evidence])
            + "\nINSTRUCTIONS: Draft an empathetic reply strictly grounded in the historical evidence above. "
            "Do NOT invent refund amounts, policy promises, or timelines. Return JSON: "
            '{"reply": "...", "grounded_in_evidence_ids": ["..."], "requires_escalation": false}'
        )

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "responseMimeType": "application/json"
                }
            }
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

            # Parse reply content
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(raw_text)

            return GenerationResult(
                reply=parsed.get("reply", ""),
                grounded_in_evidence_ids=parsed.get("grounded_in_evidence_ids", []),
                provider=f"live_{self.model_name}",
                requires_escalation=bool(parsed.get("requires_escalation", False)),
                confidence=1.0
            )
        except Exception:
            # On any network or parsing failure, fall back safely and label honestly
            res = self.fallback_provider.generate_reply(customer_text, predicted_intent, evidence, system_constraints)
            res.provider = "deterministic_grounded_fallback"
            return res


# Alias for backward compatibility
GenerativeLLMProvider = ExternalAPIProvider
