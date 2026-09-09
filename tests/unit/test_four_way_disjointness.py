"""
Unit and regression tests verifying four-way dataset quarantine and disjointness:
- Exact counts for Gold (200), Silver Dev (200), Validation (156), and Retrieval (1,427)
- 0 pairwise overlap across tweet IDs, thread IDs, and author IDs for all 6 pairs
- Strict quarantine of Gold candidates (unlabelled, absent from development/evaluation inputs)
- Human retrieval relevance queue integrity (50 queries x top-3, status PENDING_HUMAN_ANNOTATION)
"""

import json
from pathlib import Path
import pytest

project_root = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def dataset_partitions():
    gold_path = project_root / "data" / "gold" / "gold_annotation_queue.jsonl"
    silver_path = project_root / "data" / "interim" / "silver_eval_set.jsonl"
    val_path = project_root / "data" / "val" / "dev_tuning.jsonl"
    retrieval_path = project_root / "data" / "processed" / "retrieval_corpus.jsonl"

    assert gold_path.exists(), f"Missing {gold_path}"
    assert silver_path.exists(), f"Missing {silver_path}"
    assert val_path.exists(), f"Missing {val_path}"
    assert retrieval_path.exists(), f"Missing {retrieval_path}"

    with open(gold_path, "r", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f]
    with open(silver_path, "r", encoding="utf-8") as f:
        silver = [json.loads(line) for line in f]
    with open(val_path, "r", encoding="utf-8") as f:
        val = [json.loads(line) for line in f]
    with open(retrieval_path, "r", encoding="utf-8") as f:
        retrieval = [json.loads(line) for line in f]

    return {
        "gold": gold,
        "silver": silver,
        "val": val,
        "retrieval": retrieval
    }


def test_four_way_exact_counts(dataset_partitions):
    """Verify exact deterministic partition record counts."""
    assert len(dataset_partitions["gold"]) == 200, "Gold Candidate Queue must have exactly 200 queries"
    assert len(dataset_partitions["silver"]) == 200, "Silver Dev Benchmark must have exactly 200 queries"
    assert len(dataset_partitions["val"]) == 156, "Quarantined Validation Split must have exactly 156 pairs"
    assert len(dataset_partitions["retrieval"]) == 1427, "Clean Retrieval Corpus must have exactly 1,427 pairs"


def test_four_way_zero_pairwise_leakage(dataset_partitions):
    """Verify zero tweet, thread, or author overlap across all 6 pairs."""
    splits = {}
    for name, records in dataset_partitions.items():
        tweets = {str(r["customer_tweet_id"]) for r in records}
        threads = {str(r["conversation_id"]) for r in records}
        authors = {str(r["customer_author_id"]) for r in records if str(r.get("customer_author_id")) not in ["", "nan"]}
        splits[name] = {"tweets": tweets, "threads": threads, "authors": authors}

    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            tweet_overlap = splits[s1]["tweets"].intersection(splits[s2]["tweets"])
            thread_overlap = splits[s1]["threads"].intersection(splits[s2]["threads"])
            author_overlap = splits[s1]["authors"].intersection(splits[s2]["authors"])

            assert len(tweet_overlap) == 0, f"Tweet ID leakage between {s1} and {s2}: {tweet_overlap}"
            assert len(thread_overlap) == 0, f"Thread ID leakage between {s1} and {s2}: {thread_overlap}"
            assert len(author_overlap) == 0, f"Author ID leakage between {s1} and {s2}: {author_overlap}"


def test_gold_candidates_are_strictly_quarantined(dataset_partitions):
    """Verify Gold candidate records have valid human annotations and remain strictly quarantined from other splits."""
    gold = dataset_partitions["gold"]
    valid_intents = {
        "playback_issues", "app_crash_technical", "offline_downloads", "device_connectivity",
        "playlist_library", "subscription_billing", "account_access_security",
        "feature_request_ui", "service_status_outage", "other_unsupported"
    }
    for r in gold:
        assert r["gold_intent"] in valid_intents, f"Gold intent must be valid, got {r['gold_intent']}"
        assert r["ground_truth_decision"] in ("AUTO_HANDLE", "ESCALATE"), f"Gold decision must be valid, got {r['ground_truth_decision']}"
        assert r["annotator"] != "", "Gold annotator must be recorded"

    # Verify Gold tweet IDs never appear in training, validation, or silver dev
    gold_tweets = {str(r["customer_tweet_id"]) for r in gold}
    dev_tweets = {str(r["customer_tweet_id"]) for r in dataset_partitions["silver"]}
    val_tweets = {str(r["customer_tweet_id"]) for r in dataset_partitions["val"]}
    ret_tweets = {str(r["customer_tweet_id"]) for r in dataset_partitions["retrieval"]}

    assert len(gold_tweets.intersection(dev_tweets)) == 0
    assert len(gold_tweets.intersection(val_tweets)) == 0
    assert len(gold_tweets.intersection(ret_tweets)) == 0


def test_retrieval_corpus_has_intent_metadata(dataset_partitions):
    """Verify all records in retrieval corpus carry intent metadata for relevance proxy evaluation."""
    for r in dataset_partitions["retrieval"]:
        assert "intent" in r, "Retrieval record missing 'intent' field"
        assert r["intent"] != "", "Retrieval record has empty intent"


def test_human_retrieval_annotation_queue_structure():
    """Verify the future human retrieval annotation queue schema and pending status."""
    queue_path = project_root / "reports" / "annotations" / "retrieval_relevance_annotation_queue.jsonl"
    status_path = project_root / "reports" / "annotations" / "retrieval_annotation_status.json"

    assert queue_path.exists(), f"Missing {queue_path}"
    assert status_path.exists(), f"Missing {status_path}"

    with open(queue_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    assert len(items) == 150, "Human retrieval queue must contain exactly 150 candidate pairs (50 queries x 3)"

    for it in items:
        assert it["human_relevance_label"] == "", "Human label must be unlabelled (not fabricated)"
        assert it["status"] == "PENDING_HUMAN_ANNOTATION"
        assert it["retrieved_rank"] in [1, 2, 3]

    with open(status_path, "r", encoding="utf-8") as f:
        status_data = json.load(f)

    assert status_data["status"] == "PENDING_HUMAN_ANNOTATION"
    assert status_data["labeled_count"] == 0
    assert status_data["pending_count"] == 150


def test_exact_dataset_accounting_reconciliation(dataset_partitions):
    """
    Verify exact mathematical reconciliation:
    sum(Gold + Silver + Validation + Retrieval + explicitly excluded) == reconstructed interaction count (2,328)
    """
    unselected_path = project_root / "data" / "interim" / "unselected_multiturn_interactions.jsonl"
    assert unselected_path.exists(), f"Missing {unselected_path}"

    with open(unselected_path, "r", encoding="utf-8") as f:
        unselected = [json.loads(line) for line in f]

    gold_count = len(dataset_partitions["gold"])
    silver_count = len(dataset_partitions["silver"])
    val_count = len(dataset_partitions["val"])
    retrieval_count = len(dataset_partitions["retrieval"])
    unselected_count = len(unselected)

    # 1. Exact partition counts
    assert gold_count == 200
    assert silver_count == 200
    assert val_count == 156
    assert retrieval_count == 1427
    assert unselected_count == 345

    # 2. Excluded group counts
    gold_unselected = [r for r in unselected if r.get("exclusion_group") == "gold_component_secondary_turn"]
    silver_unselected = [r for r in unselected if r.get("exclusion_group") == "silver_component_secondary_turn"]
    assert len(gold_unselected) == 190, f"Expected 190 gold secondary turns, got {len(gold_unselected)}"
    assert len(silver_unselected) == 155, f"Expected 155 silver secondary turns, got {len(silver_unselected)}"

    # 3. Sum matches reconstructed pairs exactly
    total_reconciled = gold_count + silver_count + val_count + retrieval_count + unselected_count
    assert total_reconciled == 2328, f"Reconciliation mismatch: {total_reconciled} != 2328"

    # 4. Zero overlap of unselected turns with opposing partitions
    silver_tweets = {str(r["customer_tweet_id"]) for r in dataset_partitions["silver"]}
    gold_tweets = {str(r["customer_tweet_id"]) for r in dataset_partitions["gold"]}
    val_tweets = {str(r["customer_tweet_id"]) for r in dataset_partitions["val"]}
    ret_tweets = {str(r["customer_tweet_id"]) for r in dataset_partitions["retrieval"]}

    unsel_gold_tweets = {str(r["customer_tweet_id"]) for r in gold_unselected}
    unsel_silver_tweets = {str(r["customer_tweet_id"]) for r in silver_unselected}

    assert len(unsel_gold_tweets.intersection(silver_tweets)) == 0, "Leakage: unselected gold turn overlaps with silver dev!"
    assert len(unsel_gold_tweets.intersection(val_tweets)) == 0, "Leakage: unselected gold turn overlaps with validation!"
    assert len(unsel_gold_tweets.intersection(ret_tweets)) == 0, "Leakage: unselected gold turn overlaps with retrieval!"

    assert len(unsel_silver_tweets.intersection(gold_tweets)) == 0, "Leakage: unselected silver turn overlaps with gold queue!"
    assert len(unsel_silver_tweets.intersection(val_tweets)) == 0, "Leakage: unselected silver turn overlaps with validation!"
    assert len(unsel_silver_tweets.intersection(ret_tweets)) == 0, "Leakage: unselected silver turn overlaps with retrieval!"
