"""
Claim-level support checking and hallucination detection module.
Evaluates drafted replies sentence-by-sentence against retrieved historical evidence,
classifying claims into SUPPORTED, UNSUPPORTED, or UNCERTAIN.
"""

import re
from enum import Enum
from typing import Dict, List, Any


class ClaimSupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNCERTAIN = "UNCERTAIN"


class HallucinationChecker:
    """
    Evaluates evidence grounding at the claim/sentence level.
    Treats uncertain claims conservatively as potential grounding risks.
    """

    # High-risk claim patterns that should NEVER appear unless explicitly in evidence
    HIGH_RISK_PATTERNS = [
        re.compile(r"\$\d+|\b\d+\s*dollars|\brefund\s+of\b", re.I),  # Specific dollar refund promises
        re.compile(r"within\s+\d+\s+(hours|days|minutes)", re.I),     # Specific SLA timeline guarantees
        re.compile(r"credited\s+your\s+account", re.I),              # Claiming money was credited
        re.compile(r"we\s+have\s+fixed\s+the\s+bug\b", re.I)         # Claiming bug is completely fixed
    ]

    def split_into_claims(self, text: str) -> List[str]:
        """Splits reply into discrete sentence-level claim units."""
        if not text:
            return []
        # Split on sentence boundaries (.!?) while handling URLs
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
        return claims

    def verify_claims(
        self,
        draft_reply: str,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verifies each sentence in draft_reply against historical evidence texts.
        Returns detailed classification and unsupported claim metrics.
        """
        claims = self.split_into_claims(draft_reply)
        if not claims:
            return {
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "uncertain_claims": 0,
                "unsupported_claim_rate": 0.0,
                "is_fully_grounded": True,
                "claims_detail": []
            }

        evidence_text = " ".join([
            e.get("historical_brand_reply", "") + " " + e.get("historical_customer", "")
            for e in retrieved_evidence
        ]).lower()

        claims_detail = []
        supported_count = 0
        unsupported_count = 0
        uncertain_count = 0

        for claim in claims:
            claim_lower = claim.lower()

            # 1. High risk check: does claim contain monetary or SLA assertions not in evidence?
            has_high_risk = any(p.search(claim) for p in self.HIGH_RISK_PATTERNS)
            if has_high_risk:
                evidence_matches_risk = any(p.search(evidence_text) for p in self.HIGH_RISK_PATTERNS)
                if not evidence_matches_risk:
                    claims_detail.append({
                        "claim": claim,
                        "status": ClaimSupportStatus.UNSUPPORTED.value,
                        "reason": "Contains specific monetary or SLA promise absent from historical evidence."
                    })
                    unsupported_count += 1
                    continue

            # 2. Conversational closers and polite signoffs
            if re.search(r"let\s+us\s+know|keep\s+us\s+posted|hope\s+this\s+helps|happy\s+to\s+help|give\s+us\s+a\s+shout|have\s+a\s+(great|nice)\s+day|feel\s+free|no\s+worries|cheers", claim_lower):
                claims_detail.append({
                    "claim": claim,
                    "status": ClaimSupportStatus.SUPPORTED.value,
                    "reason": "Standard conversational closer / pleasantry."
                })
                supported_count += 1
                continue

            # 3. Key content word overlap with evidence
            words = [w for w in re.findall(r"\b[a-z]{3,}\b", claim_lower) if w not in {"the", "and", "you", "for", "that", "with", "this", "can", "have"}]
            if not words:
                # Conversational pleasantry like "Thanks!" or "/CH"
                claims_detail.append({
                    "claim": claim,
                    "status": ClaimSupportStatus.SUPPORTED.value,
                    "reason": "Standard conversational pleasantry."
                })
                supported_count += 1
                continue

            matches = sum(1 for w in words if w in evidence_text)
            overlap_ratio = matches / len(words)

            if overlap_ratio >= 0.40:
                status = ClaimSupportStatus.SUPPORTED
                reason = f"Substantive lexical and procedural overlap ({overlap_ratio:.0%}) with evidence."
                supported_count += 1
            elif overlap_ratio >= 0.20:
                status = ClaimSupportStatus.UNCERTAIN
                reason = f"Partial overlap ({overlap_ratio:.0%}) with evidence; treated conservatively."
                uncertain_count += 1
            else:
                status = ClaimSupportStatus.UNSUPPORTED
                reason = f"Low overlap ({overlap_ratio:.0%}) with evidence; claim ungrounded."
                unsupported_count += 1

            claims_detail.append({
                "claim": claim,
                "status": status.value,
                "reason": reason
            })

        total = len(claims)
        unsupported_rate = (unsupported_count + uncertain_count * 0.5) / total

        return {
            "total_claims": total,
            "supported_claims": supported_count,
            "unsupported_claims": unsupported_count,
            "uncertain_claims": uncertain_count,
            "unsupported_claim_rate": round(float(unsupported_rate), 4),
            "is_fully_grounded": bool(unsupported_count == 0 and uncertain_count == 0),
            "claims_detail": claims_detail
        }
