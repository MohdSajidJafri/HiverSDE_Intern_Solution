"""
Conservative Escalation Policy Engine.
Enforces auditable routing decisions: AUTO_HANDLE vs ESCALATE with explicit reason codes.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from src.hiver_agent.config import AppConfig, default_config


class EscalationDecision(BaseModel):
    action: str  # 'AUTO_HANDLE' | 'ESCALATE'
    reason_code: str
    reason: str
    risk_score: float
    supporting_signals: Dict[str, Any] = Field(default_factory=dict)


class EscalationPolicy:
    """
    Evaluates classification, retrieval quality, contradiction, and claim-level grounding signals
    against frozen operating thresholds.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or default_config

    def evaluate(
        self,
        intent_result: Dict[str, Any],
        evidence_assessment: Dict[str, Any],
        claim_verification: Optional[Dict[str, Any]] = None
    ) -> EscalationDecision:
        conf_th = self.config.thresholds.intent_confidence_threshold
        quality_th = self.config.thresholds.evidence_quality_threshold
        sensitive_intents = self.config.escalation.sensitive_intents

        predicted_intent = intent_result.get("predicted_intent", "")
        calibrated_conf = float(intent_result.get("calibrated_confidence", 0.0))
        is_outlier = bool(intent_result.get("is_novelty_outlier", False))

        evidence_quality = float(evidence_assessment.get("evidence_quality_score", 0.0))
        has_contradiction = bool(evidence_assessment.get("has_contradiction", False))
        contradiction_explanation = evidence_assessment.get("contradiction_explanation", "")

        unsupported_claims = 0
        if claim_verification:
            unsupported_claims = claim_verification.get("unsupported_claims", 0)

        signals = {
            "predicted_intent": predicted_intent,
            "calibrated_confidence": calibrated_conf,
            "is_outlier": is_outlier,
            "evidence_quality": evidence_quality,
            "has_contradiction": has_contradiction,
            "unsupported_claims": unsupported_claims
        }

        # 1. Critical Check: Sensitive Account / Security / Billing Domains
        if predicted_intent in sensitive_intents:
            return EscalationDecision(
                action="ESCALATE",
                reason_code="SENSITIVE_ACCOUNT_DOMAIN",
                reason=f"Inquiry belongs to sensitive category '{predicted_intent}' requiring human agent authority.",
                risk_score=0.90,
                supporting_signals=signals
            )

        # 2. Critical Check: Contradictory Historical Precedent
        if has_contradiction:
            return EscalationDecision(
                action="ESCALATE",
                reason_code="CONTRADICTORY_HISTORICAL_EVIDENCE",
                reason=f"Retrieved historical resolutions offer conflicting advice: {contradiction_explanation}",
                risk_score=0.85,
                supporting_signals=signals
            )

        # 3. Critical Check: Novelty / Out-of-Scope Anomaly
        if is_outlier:
            return EscalationDecision(
                action="ESCALATE",
                reason_code="OUT_OF_SCOPE_ANOMALY",
                reason="Query is far from all known support intent centroids in embedding space.",
                risk_score=0.80,
                supporting_signals=signals
            )

        # 4. Low Intent Confidence
        if calibrated_conf < conf_th:
            return EscalationDecision(
                action="ESCALATE",
                reason_code="LOW_INTENT_CONFIDENCE",
                reason=f"Calibrated intent confidence ({calibrated_conf:.2f}) is below threshold ({conf_th:.2f}).",
                risk_score=round(1.0 - calibrated_conf, 2),
                supporting_signals=signals
            )

        # 5. Low Evidence Quality
        if evidence_quality < quality_th:
            return EscalationDecision(
                action="ESCALATE",
                reason_code="LOW_EVIDENCE_QUALITY",
                reason=f"Retrieved evidence quality score ({evidence_quality:.2f}) is below threshold ({quality_th:.2f}).",
                risk_score=round(1.0 - evidence_quality, 2),
                supporting_signals=signals
            )

        # 6. Ungrounded Claim Risk
        if unsupported_claims > 0:
            return EscalationDecision(
                action="ESCALATE",
                reason_code="UNGROUNDED_CLAIM_RISK",
                reason=f"Drafted reply contains {unsupported_claims} unsupported or unverified factual claims.",
                risk_score=0.75,
                supporting_signals=signals
            )

        # All safety checks passed -> AUTO_HANDLE
        risk_score = round(max(0.05, 1.0 - (calibrated_conf * 0.5 + evidence_quality * 0.5)), 2)
        return EscalationDecision(
            action="AUTO_HANDLE",
            reason_code="HIGH_CONFIDENCE_GROUNDED",
            reason="Inquiry has high calibrated intent confidence, high-quality consistent evidence, and verified grounding.",
            risk_score=risk_score,
            supporting_signals=signals
        )
