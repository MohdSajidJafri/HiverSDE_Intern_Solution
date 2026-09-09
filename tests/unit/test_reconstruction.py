"""
Unit tests for ConversationReconstructor.
"""

import pytest
import pandas as pd
from src.hiver_agent.data.reconstruction import ConversationReconstructor


@pytest.fixture
def sample_tweets_df():
    data = [
        # Customer tweet 1
        {"tweet_id": "101", "author_id": "user_a", "inbound": True, "created_at": "Tue Oct 31 10:00:00 2017", "text": "App crashed", "response_tweet_id": "102", "in_response_to_tweet_id": None},
        # Brand response 1
        {"tweet_id": "102", "author_id": "SpotifyCares", "inbound": False, "created_at": "Tue Oct 31 10:05:00 2017", "text": "Try reinstalling", "response_tweet_id": None, "in_response_to_tweet_id": "101"},
        # Customer follow-up 2
        {"tweet_id": "103", "author_id": "user_a", "inbound": True, "created_at": "Tue Oct 31 10:10:00 2017", "text": "Reinstalling worked thanks", "response_tweet_id": "104", "in_response_to_tweet_id": "102"},
        # Brand response 2
        {"tweet_id": "104", "author_id": "SpotifyCares", "inbound": False, "created_at": "Tue Oct 31 10:15:00 2017", "text": "Glad to hear that!", "response_tweet_id": None, "in_response_to_tweet_id": "103"},
        # Orphan customer tweet (no brand response)
        {"tweet_id": "201", "author_id": "user_b", "inbound": True, "created_at": "Tue Oct 31 11:00:00 2017", "text": "Is spotify down?", "response_tweet_id": None, "in_response_to_tweet_id": None},
        # Orphan brand tweet (missing parent)
        {"tweet_id": "301", "author_id": "SpotifyCares", "inbound": False, "created_at": "Tue Oct 31 12:00:00 2017", "text": "Welcome back!", "response_tweet_id": None, "in_response_to_tweet_id": None},
        # Different brand tweet
        {"tweet_id": "401", "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 31 12:00:00 2017", "text": "Hello Amazon", "response_tweet_id": None, "in_response_to_tweet_id": "101"}
    ]
    return pd.DataFrame(data)


def test_reconstruction_pairs(sample_tweets_df):
    reconstructor = ConversationReconstructor(brand="SpotifyCares")
    pairs = reconstructor.reconstruct_pairs(sample_tweets_df)
    
    # Should reconstruct exactly 2 Spotify pairs (101->102 and 103->104)
    assert len(pairs) == 2
    
    pair_1 = [p for p in pairs if p["customer_tweet_id"] == "101"][0]
    assert pair_1["brand_tweet_id"] == "102"
    assert pair_1["customer_text"] == "App crashed"
    assert pair_1["brand_reply"] == "Try reinstalling"
    assert pair_1["metadata"]["brand"] == "SpotifyCares"


def test_reconstruction_ignores_orphans_and_other_brands(sample_tweets_df):
    reconstructor = ConversationReconstructor(brand="SpotifyCares")
    pairs = reconstructor.reconstruct_pairs(sample_tweets_df)
    
    tweet_ids = [p["customer_tweet_id"] for p in pairs]
    assert "201" not in tweet_ids  # orphan customer
    assert "301" not in [p["brand_tweet_id"] for p in pairs]  # orphan brand
    assert "401" not in [p["brand_tweet_id"] for p in pairs]  # other brand


def test_build_thread_graph(sample_tweets_df):
    reconstructor = ConversationReconstructor(brand="SpotifyCares")
    pairs = reconstructor.reconstruct_pairs(sample_tweets_df)
    threads = reconstructor.build_thread_graph(pairs)
    
    assert len(threads) > 0
    for conv_id, thread_pairs in threads.items():
        assert len(thread_pairs) >= 1
