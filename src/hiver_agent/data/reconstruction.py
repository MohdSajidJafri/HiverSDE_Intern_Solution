"""
Conversation reconstruction module for customer support interactions.
Reconstructs multi-turn conversational threads, separates inbounds from outbounds,
and preserves strict provenance metadata.
"""

from typing import Dict, List, Any, Optional, Set
import pandas as pd
from datetime import datetime


class ConversationReconstructor:
    """Reconstructs customer support conversation pairs and multi-turn threads."""

    def __init__(self, brand: str = "SpotifyCares"):
        self.brand = brand

    def reconstruct_pairs(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extracts canonical customer inquiry -> brand response interaction pairs.
        Preserves provenance, timestamps, and thread context.
        """
        # 1. Filter outbound replies authored by the target brand
        brand_replies = df[(df["author_id"] == self.brand) & (df["inbound"] == False)].copy()
        
        # 2. Drop rows with missing parent pointer
        valid_replies = brand_replies.dropna(subset=["in_response_to_tweet_id"]).copy()
        
        # Ensure ID types match cleanly (handling float conversion artifacts like '115712.0')
        valid_replies["in_response_to_tweet_id"] = (
            pd.to_numeric(valid_replies["in_response_to_tweet_id"], errors="coerce")
            .astype("Int64")
            .astype(str)
        )
        df_lookup = df.copy()
        df_lookup["tweet_id"] = (
            pd.to_numeric(df_lookup["tweet_id"], errors="coerce")
            .astype("Int64")
            .astype(str)
        )
        
        # 3. Join with parent customer tweets
        merged = valid_replies.merge(
            df_lookup[["tweet_id", "author_id", "text", "created_at", "inbound", "in_response_to_tweet_id"]],
            left_on="in_response_to_tweet_id",
            right_on="tweet_id",
            suffixes=("_brand", "_cust")
        )

        pairs = []
        seen_pairs: Set[str] = set()

        for _, row in merged.iterrows():
            cust_id = str(row["tweet_id_cust"])
            brand_id = str(row["tweet_id_brand"])
            pair_key = f"{cust_id}->{brand_id}"

            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            cust_text = str(row["text_cust"]).strip()
            brand_text = str(row["text_brand"]).strip()

            # Skip empty or degenerate messages
            if not cust_text or not brand_text:
                continue

            # Root conversation ID: if customer tweet responded to an earlier tweet, follow pointer if available
            root_conv_id = str(row.get("in_response_to_tweet_id_cust", cust_id))
            if pd.isna(root_conv_id) or root_conv_id == "nan":
                root_conv_id = cust_id

            pair_record = {
                "pair_id": f"pair_{brand_id}_{cust_id}",
                "conversation_id": root_conv_id,
                "customer_tweet_id": cust_id,
                "brand_tweet_id": brand_id,
                "customer_author_id": str(row["author_id_cust"]),
                "brand_author_id": self.brand,
                "customer_text": cust_text,
                "brand_reply": brand_text,
                "customer_created_at": str(row["created_at_cust"]),
                "brand_created_at": str(row["created_at_brand"]),
                "metadata": {
                    "inbound": bool(row["inbound_cust"]),
                    "brand": self.brand
                }
            }
            pairs.append(pair_record)

        return pairs

    def build_thread_graph(self, pairs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Groups pairs into conversation threads indexed by root conversation_id."""
        threads: Dict[str, List[Dict[str, Any]]] = {}
        for pair in pairs:
            conv_id = pair["conversation_id"]
            if conv_id not in threads:
                threads[conv_id] = []
            threads[conv_id].append(pair)
        return threads
