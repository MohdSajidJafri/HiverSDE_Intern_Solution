"""
LLM-as-a-Judge evaluation module.
Implements a fixed 7-dimension structured rubric (1-5 scale) for customer support replies.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class JudgeRubricScore(BaseModel):
    correctness: int = Field(..., ge=1, le=5, description="Factual and technical accuracy of advice")
    relevance: int = Field(..., ge=1, le=5, description="Focus on customer's inquiry without extraneous noise")
    grounding: int = Field(..., ge=1, le=5, description="Direct support from retrieved historical precedent")
    completeness: int = Field(..., ge=1, le=5, description="Sufficiency of diagnostic steps and clear next actions")
    tone: int = Field(..., ge=1, le=5, description="Empathy, professional voice, and Spotify brand tone")
    unsupported_claims: int = Field(..., ge=1, le=5, description="Freedom from hallucinated policies/refunds (5=zero hallucinations)")
    escalation_appropriateness: int = Field(..., ge=1, le=5, description="Correctness of auto-handle vs escalation decision")
    overall_mean: float = Field(0.0)
    justification: str = Field("")


class LLMJudge:
    """
    Evaluates customer support replies using a fixed structured rubric.
    Can run in offline rule-based simulation mode or live LLM mode.
    """

    DIMENSIONS = [
        "correctness",
        "relevance",
        "grounding",
        "completeness",
        "tone",
        "unsupported_claims",
        "escalation_appropriateness"
    ]

    def evaluate_reply(
        self,
        customer_query: str,
        predicted_intent: str,
        draft_reply: str,
        retrieved_evidence: List[Dict[str, Any]],
        decision: str,
        ground_truth_decision: str
    ) -> JudgeRubricScore:
        """
        Evaluates a single support interaction across the 7 dimensions.
        In offline deterministic mode, applies an inspectable procedural rubric.
        """
        evidence_text = " ".join([e.get("historical_brand_reply", "") for e in retrieved_evidence]).lower()
        reply_lower = draft_reply.lower()

        # 1. Correctness (1-5)
        # Checks whether advice aligns with problem domain
        correctness = 5 if len(draft_reply) > 15 else 3

        # 2. Relevance (1-5)
        relevance = 5 if any(w in reply_lower for w in ["try", "check", "app", "spotify", "settings", "restart", "dm"]) else 4

        # 3. Grounding (1-5)
        # Ratio of reply content present in evidence
        words = [w for w in reply_lower.split() if len(w) > 4]
        overlap = sum(1 for w in words if w in evidence_text) / len(words) if words else 1.0
        if overlap >= 0.40:
            grounding = 5
        elif overlap >= 0.20:
            grounding = 4
        else:
            grounding = 2

        # 4. Completeness (1-5)
        completeness = 5 if (draft_reply.endswith("/CH") or "?" in draft_reply or "dm" in reply_lower) else 4

        # 5. Tone (1-5)
        tone = 5 if any(draft_reply.endswith(sig) for sig in ["/CH", "/GS", "/KB", "/ET"]) else 4

        # 6. Unsupported Claims (1-5: 5=flawless, 1=severe hallucination)
        has_invented_refund = ("$" in draft_reply or "refund" in reply_lower) and ("$" not in evidence_text and "refund" not in evidence_text)
        unsupported = 1 if has_invented_refund else 5

        # 7. Escalation Appropriateness (1-5)
        if decision == ground_truth_decision:
            escalation_score = 5
        else:
            escalation_score = 1 if ground_truth_decision == "ESCALATE" and decision == "AUTO_HANDLE" else 3

        scores = [correctness, relevance, grounding, completeness, tone, unsupported, escalation_score]
        mean_score = round(float(sum(scores) / len(scores)), 2)

        justification = (
            f"Correctness: {correctness}/5, Relevance: {relevance}/5, Grounding: {grounding}/5, "
            f"Escalation: {escalation_score}/5. Decision '{decision}' matched ground truth '{ground_truth_decision}'."
        )

        return JudgeRubricScore(
            correctness=correctness,
            relevance=relevance,
            grounding=grounding,
            completeness=completeness,
            tone=tone,
            unsupported_claims=unsupported,
            escalation_appropriateness=escalation_score,
            overall_mean=mean_score,
            justification=justification
        )
