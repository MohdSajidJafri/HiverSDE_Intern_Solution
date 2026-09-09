"""
Brand profiling and evidence evaluation module.
Empirically assesses candidate brands for conversational quality, language purity,
and actionable troubleshooting precedent.
"""

import re
from typing import Dict, List, Any
import pandas as pd


class BrandProfiler:
    """Profiles and scores brands in customer support datasets."""

    CANDIDATE_BRANDS = [
        "SpotifyCares",
        "AppleSupport",
        "AmazonHelp",
        "Uber_Support",
        "British_Airways",
        "Delta",
        "Tesco"
    ]

    # Regex detecting concrete troubleshooting instructions (restart, cache, reinstall, toggle, settings, steps)
    TROUBLESHOOT_PATTERN = re.compile(
        r"restart|reinstall|cache|uninstall|sleep/wake|hold|toggle|switch|backstage|reboot|troubleshoot|check\s+from\s+settings|clean\s+install",
        re.IGNORECASE
    )

    # Regex detecting canned boilerplate redirects to private DMs / tickets
    CANNED_DM_PATTERN = re.compile(
        r"send\s+us\s+a\s+dm|join\s+us\s+in\s+a\s+dm|in\s+a\s+dm|private\s+message|dm\s+us|direct\s+message|contact\s+us\s+via",
        re.IGNORECASE
    )

    def profile_brands(self, df: pd.DataFrame, candidates: List[str] = None) -> List[Dict[str, Any]]:
        """
        Profiles candidate brands and computes empirical metrics.
        Returns ranked list of candidate metrics.
        """
        brands = candidates or self.CANDIDATE_BRANDS
        profiles = []

        # Convert IDs to clean strings
        df_clean = df.copy()
        df_clean["tweet_id"] = (
            pd.to_numeric(df_clean["tweet_id"], errors="coerce")
            .astype("Int64")
            .astype(str)
        )
        df_clean["in_response_to_tweet_id"] = (
            pd.to_numeric(df_clean["in_response_to_tweet_id"], errors="coerce")
            .astype("Int64")
            .astype(str)
        )

        for brand in brands:
            # Outbound replies authored by this brand
            brand_outbound = df_clean[(df_clean["author_id"] == brand) & (df_clean["inbound"] == False)]
            outbound_count = len(brand_outbound)
            if outbound_count == 0:
                continue

            # Join with parent customer tweets
            merged = brand_outbound.merge(
                df_clean[["tweet_id", "text", "author_id", "created_at"]],
                left_on="in_response_to_tweet_id",
                right_on="tweet_id",
                suffixes=("_brand", "_cust")
            )
            paired_count = len(merged)

            # 1. Non-English character estimation
            non_ascii_frac = (
                merged["text_cust"].apply(lambda x: len(re.findall(r"[^\x00-\x7F]", str(x))) > 5).mean()
                if paired_count > 0 else 0.0
            )

            # 2. Average lengths
            avg_cust_len = merged["text_cust"].str.len().mean() if paired_count > 0 else 0.0
            avg_brand_len = merged["text_brand"].str.len().mean() if paired_count > 0 else 0.0

            # 3. Diversity ratio
            unique_brand_replies = merged["text_brand"].nunique() if paired_count > 0 else 0
            diversity_ratio = unique_brand_replies / paired_count if paired_count > 0 else 0.0

            # 4. Concrete Troubleshooting Resolution Rate
            troubleshoot_ratio = (
                merged["text_brand"].str.contains(self.TROUBLESHOOT_PATTERN).mean()
                if paired_count > 0 else 0.0
            )

            # 5. Canned DM Redirection Rate
            canned_dm_ratio = (
                merged["text_brand"].str.contains(self.CANNED_DM_PATTERN).mean()
                if paired_count > 0 else 0.0
            )

            # 6. Composite Usable Support Evidence Score (0 to 100):
            # Volume factor (max 20 points)
            vol_score = min(20.0, (paired_count / 2000.0) * 20.0)
            # English purity (max 25 points)
            eng_score = (1.0 - non_ascii_frac) * 25.0
            # Reply diversity (max 25 points)
            div_score = diversity_ratio * 25.0
            # Resolution quality: rewards concrete troubleshooting, penalizes excessive pure canned DM redirects
            # (max 30 points)
            resolution_score = max(0.0, min(30.0, (troubleshoot_ratio * 100.0 * 1.5) - (canned_dm_ratio * 100.0 * 0.15)))

            evidence_score = vol_score + eng_score + div_score + resolution_score

            profiles.append({
                "brand": brand,
                "outbound_count": outbound_count,
                "paired_in_sample": paired_count,
                "non_english_est": round(float(non_ascii_frac), 4),
                "avg_cust_char_len": round(float(avg_cust_len), 1),
                "avg_brand_char_len": round(float(avg_brand_len), 1),
                "diversity_ratio": round(float(diversity_ratio), 4),
                "troubleshoot_ratio": round(float(troubleshoot_ratio), 4),
                "canned_dm_ratio": round(float(canned_dm_ratio), 4),
                "usable_support_evidence_score": round(float(evidence_score), 2)
            })

        # Rank by composite score descending
        profiles.sort(key=lambda x: x["usable_support_evidence_score"], reverse=True)
        return profiles
