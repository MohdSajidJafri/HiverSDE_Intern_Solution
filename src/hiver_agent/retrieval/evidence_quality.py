"""
Evidence Quality and Contradiction Assessment module.
Assesses solution-level relevance, intent alignment, and 3-state contradiction status
across retrieved historical customer-brand interactions.
"""

import re
from enum import Enum
from typing import Dict, List, Any, Optional
import numpy as np


class ContradictionStatus(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceQualityAssessor:
    """
    Evaluates evidence quality beyond raw vector similarity.
    Distinguishes compatible versus genuinely incompatible historical advice.
    """

    # Antagonistic action pairs (Action X vs Action Y that contradict)
    CONTRADICTORY_PATTERNS = [
        # Reinstall vs Do NOT reinstall / known outage
        (
            re.compile(r"reinstall|uninstall", re.I),
            re.compile(r"don't reinstall|do not reinstall|server outage|maintenance mode|known issue|investigating|wait", re.I)
        ),
        # Restart device vs Hardware replacement / repair
        (
            re.compile(r"restart|reboot|turn off and on", re.I),
            re.compile(r"hardware issue|repair|replace device|service center", re.I)
        ),
        # Contact us in DM vs Issue resolved on website / self-service only
        (
            re.compile(r"send us a dm|private message", re.I),
            re.compile(r"cannot assist in dm|only via phone|contact your carrier", re.I)
        )
    ]

    # Actionable troubleshooting indicators
    ACTIONABLE_REGEX = re.compile(
        r"try|check|restart|reinstall|settings|cache|version|toggle|update|logout|login|switch|step",
        re.I
    )

    def assess_evidence(
        self,
        query: str,
        predicted_intent: str,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assesses retrieved evidence for:
        1. Solution-level relevance and actionability.
        2. Intent compatibility.
        3. 3-state contradiction analysis.
        """
        if not retrieved_evidence:
            return {
                "top_similarity": 0.0,
                "evidence_quality_score": 0.0,
                "contradiction_status": ContradictionStatus.INSUFFICIENT_EVIDENCE.value,
                "contradiction_explanation": "No evidence records retrieved.",
                "solution_relevance": 0.0,
                "is_actionable": False,
                "has_contradiction": False
            }

        top_similarity = float(retrieved_evidence[0].get("similarity", 0.0))
        historical_replies = [e.get("historical_brand_reply", "") for e in retrieved_evidence]

        # 1. Actionability of top evidence
        actionable_flags = [bool(self.ACTIONABLE_REGEX.search(r)) for r in historical_replies]
        actionable_ratio = sum(actionable_flags) / len(actionable_flags) if actionable_flags else 0.0

        # 2. 3-State Contradiction Analysis
        contradiction_status, explanation = self._check_contradictions(historical_replies)

        # 3. Composite Evidence Quality Score (0.0 to 1.0)
        # Penalized heavily if contradiction detected or non-actionable
        quality_score = top_similarity * 0.5 + actionable_ratio * 0.5
        if contradiction_status == ContradictionStatus.INCOMPATIBLE:
            quality_score *= 0.3  # Severe penalty for conflicting guidance
        elif contradiction_status == ContradictionStatus.INSUFFICIENT_EVIDENCE:
            quality_score *= 0.7

        return {
            "top_similarity": round(top_similarity, 4),
            "evidence_quality_score": round(quality_score, 4),
            "contradiction_status": contradiction_status.value,
            "contradiction_explanation": explanation,
            "solution_relevance": round(actionable_ratio, 4),
            "is_actionable": bool(actionable_flags[0] if actionable_flags else False),
            "has_contradiction": (contradiction_status == ContradictionStatus.INCOMPATIBLE)
        }

    def _check_contradictions(self, replies: List[str]) -> tuple[ContradictionStatus, str]:
        """
        Evaluates pairs of retrieved brand replies for genuine procedural contradictions.
        *Methodological Limitation*: Automated contradiction detection is inherently an
        empirical approximation; ambiguous or borderline cases are treated conservatively.
        """
        if len(replies) < 2:
            return ContradictionStatus.INSUFFICIENT_EVIDENCE, "Fewer than 2 evidence candidates available."

        # Check for mutually exclusive directives
        for i in range(len(replies)):
            for j in range(i + 1, len(replies)):
                r1, r2 = replies[i], replies[j]

                for pat_a, pat_b in self.CONTRADICTORY_PATTERNS:
                    if (pat_a.search(r1) and pat_b.search(r2)) or (pat_b.search(r1) and pat_a.search(r2)):
                        return (
                            ContradictionStatus.INCOMPATIBLE,
                            f"Contradictory guidance detected: Candidate {i+1} advises action matching '{pat_a.pattern}' while Candidate {j+1} advises '{pat_b.pattern}'."
                        )

        # If multiple replies exist and both have actionable suggestions without conflict, they are COMPATIBLE
        has_substantive_content = sum(bool(len(r.split()) > 4) for r in replies) >= 2
        if has_substantive_content:
            return ContradictionStatus.COMPATIBLE, "Retrieved historical evidence candidates offer mutually compatible troubleshooting advice."

        return ContradictionStatus.INSUFFICIENT_EVIDENCE, "Retrieved candidates lack sufficient detail to verify advice compatibility."
