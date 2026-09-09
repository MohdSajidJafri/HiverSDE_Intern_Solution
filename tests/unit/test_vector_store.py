"""
Unit tests for VectorStore.
"""

import pytest
from src.hiver_agent.retrieval.vector_store import VectorStore


def test_vector_store_build_and_retrieve():
    pairs = [
        {
            "pair_id": "pair_1",
            "conversation_id": "conv_1",
            "customer_tweet_id": "101",
            "brand_tweet_id": "102",
            "customer_text": "Desktop app crashing on Windows after update",
            "brand_reply": "Try a clean reinstall of the app from spotify.com/download /CH",
            "metadata": {"brand": "SpotifyCares"}
        },
        {
            "pair_id": "pair_2",
            "conversation_id": "conv_2",
            "customer_tweet_id": "201",
            "brand_tweet_id": "202",
            "customer_text": "I was charged twice for Premium this month",
            "brand_reply": "Send us a DM with your account email and we'll check backstage /CH",
            "metadata": {"brand": "SpotifyCares"}
        },
        {
            "pair_id": "pair_3",
            "conversation_id": "conv_3",
            "customer_tweet_id": "301",
            "brand_tweet_id": "302",
            "customer_text": "Music stops playing after 5 seconds on iOS",
            "brand_reply": "Restart your phone by holding sleep/wake and volume down /GS",
            "metadata": {"brand": "SpotifyCares"}
        }
    ]

    store = VectorStore()
    store.build_index(pairs)

    assert store.is_built is True
    assert store.embeddings.shape == (3, 384)

    # Retrieve for crashing query
    results = store.retrieve("Spotify desktop app crashes on launch", top_k=2)
    assert len(results) == 2
    top = results[0]
    assert "reinstall" in top["historical_brand_reply"]
    assert top["evidence_id"] == "ev_102"
    assert top["similarity"] > 0.50
