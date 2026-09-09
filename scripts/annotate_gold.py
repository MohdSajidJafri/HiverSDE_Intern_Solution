"""
Interactive Terminal Annotation Tool for Human Gold Evaluation.
Allows human annotators to quickly review and label the 200 held-out customer inquiries.
Supports auto-saving, quick numeric shortcuts, progress tracking, and CSV synchronization.
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent.parent

JSONL_PATH = project_root / "data" / "gold" / "gold_annotation_queue.jsonl"
CSV_PATH = project_root / "data" / "gold" / "gold_annotation_queue.csv"

VALID_INTENTS = [
    "playback_issues",
    "app_crash_technical",
    "offline_downloads",
    "device_connectivity",
    "playlist_library",
    "subscription_billing",
    "account_access_security",
    "feature_request_ui",
    "service_status_outage",
    "other_unsupported"
]

DEFAULT_ESCALATE_INTENTS = {
    "subscription_billing",
    "account_access_security"
}


def load_queue_jsonl() -> List[Dict[str, Any]]:
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"Missing annotation queue: {JSONL_PATH}")
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_queue_jsonl(records: List[Dict[str, Any]]) -> None:
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sync_from_csv() -> None:
    """Syncs manual edits from gold_annotation_queue.csv to gold_annotation_queue.jsonl."""
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
    
    print(f"Reading manual annotations from {CSV_PATH}...")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_rows = {row["id"]: row for row in reader}

    records = load_queue_jsonl()
    updated_count = 0
    for r in records:
        cid = r["id"]
        if cid in csv_rows:
            crow = csv_rows[cid]
            if crow.get("gold_intent", "").strip():
                r["gold_intent"] = crow["gold_intent"].strip()
            if crow.get("ground_truth_decision", "").strip():
                r["ground_truth_decision"] = crow["ground_truth_decision"].strip().upper()
            if crow.get("annotator", "").strip():
                r["annotator"] = crow["annotator"].strip()
            if crow.get("annotation_notes", "").strip():
                r["annotation_notes"] = crow["annotation_notes"].strip()
            if crow.get("is_sensitive", "").strip():
                val = crow["is_sensitive"].strip().lower()
                r["is_sensitive"] = True if val in ("true", "1", "yes") else False
            updated_count += 1

    save_queue_jsonl(records)
    print(f"Successfully synchronized {updated_count} records to {JSONL_PATH}!")


def export_to_csv() -> None:
    """Exports current jsonl annotations to CSV format for spreadsheet inspection."""
    records = load_queue_jsonl()
    fieldnames = [
        "id", "customer_tweet_id", "brand_tweet_id", "conversation_id",
        "customer_author_id", "customer_created_at", "customer_text",
        "historical_brand_reply", "gold_intent", "annotator",
        "annotation_notes", "is_sensitive", "ground_truth_decision"
    ]
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Exported {len(records)} records to {CSV_PATH}")


def run_interactive_annotator(annotator_name: str = "") -> None:
    records = load_queue_jsonl()
    total = len(records)
    
    print("=" * 80)
    print("SPOTIFY CUSTOMER SUPPORT: HUMAN GOLD ANNOTATOR")
    print(f"Total Records: {total}")
    print("=" * 80)

    if not annotator_name:
        annotator_name = input("Enter your name or annotator ID (e.g. 'human_reviewer'): ").strip()
        if not annotator_name:
            annotator_name = "human_reviewer"

    annotated = sum(1 for r in records if r.get("gold_intent", "").strip() != "")
    print(f"Current Progress: {annotated}/{total} annotated ({total - annotated} remaining)\n")

    for idx, r in enumerate(records):
        # Skip already annotated records unless explicitly navigating
        if r.get("gold_intent", "").strip():
            continue

        cid = r["id"]
        print("-" * 80)
        print(f"Record [{idx+1}/{total}] | ID: {cid} | Created: {r.get('customer_created_at', 'N/A')}")
        print(f"Author ID: {r.get('customer_author_id', 'N/A')}")
        print("\nCUSTOMER MESSAGE:")
        print(f"  \"{r.get('customer_text', '')}\"")
        print("\nHISTORICAL BRAND REPLY (Context):")
        print(f"  \"{r.get('historical_brand_reply', '')}\"")
        print("-" * 80)

        # 1. Intent Selection
        print("Select Intent:")
        for i, intent in enumerate(VALID_INTENTS, 1):
            print(f"  [{i:2d}] {intent:<25}", end="\n" if i % 2 == 0 else " ")
        print("  [ q] Save & Quit")

        chosen_intent = ""
        while not chosen_intent:
            choice = input("\nEnter intent choice (1-10 or 'q'): ").strip().lower()
            if choice == "q":
                save_queue_jsonl(records)
                export_to_csv()
                print("\nProgress saved. Exiting annotator. You can resume anytime!")
                return
            if choice.isdigit() and 1 <= int(choice) <= 10:
                chosen_intent = VALID_INTENTS[int(choice) - 1]
            elif choice in VALID_INTENTS:
                chosen_intent = choice
            else:
                print("Invalid input. Please enter 1-10, an exact intent name, or 'q'.")

        # 2. Decision Selection
        default_decision = "ESCALATE" if chosen_intent in DEFAULT_ESCALATE_INTENTS else "AUTO_HANDLE"
        print(f"\nSelect Decision (Default: {default_decision}):")
        print(f"  [1] AUTO_HANDLE (standard safe technical troubleshooting)")
        print(f"  [2] ESCALATE   (sensitive billing, security, or out-of-scope)")
        dec_input = input(f"Enter decision [1=AUTO_HANDLE, 2=ESCALATE, Enter={default_decision}]: ").strip()
        if dec_input == "1":
            chosen_decision = "AUTO_HANDLE"
        elif dec_input == "2":
            chosen_decision = "ESCALATE"
        else:
            chosen_decision = default_decision

        # 3. Sensitivity Flag
        default_sensitive = "y" if chosen_intent in DEFAULT_ESCALATE_INTENTS else "n"
        sens_input = input(f"Is this sensitive/PII/billing? [y/n, Enter={default_sensitive}]: ").strip().lower()
        if sens_input in ("y", "yes"):
            is_sensitive = True
        elif sens_input in ("n", "no"):
            is_sensitive = False
        else:
            is_sensitive = (default_sensitive == "y")

        # 4. Optional notes
        notes = input("Optional annotation notes (Enter to skip): ").strip()

        # Update record
        r["gold_intent"] = chosen_intent
        r["ground_truth_decision"] = chosen_decision
        r["is_sensitive"] = is_sensitive
        r["annotator"] = annotator_name
        r["annotation_notes"] = notes

        # Save immediately to disk
        save_queue_jsonl(records)
        print(f"--> Saved {cid}: {chosen_intent} | {chosen_decision}\n")

    # Reached end
    save_queue_jsonl(records)
    export_to_csv()
    print("=" * 80)
    print("CONGRATULATIONS! All 200 records in the Gold Queue are fully annotated!")
    print("You can now run the master final evaluation command:")
    print("    python scripts/run_final_gold_evaluation.py")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Spotify Customer Support: Human Gold Annotator")
    parser.add_argument("--annotator", type=str, default="", help="Annotator name or ID")
    parser.add_argument("--sync-from-csv", action="store_true", help="Sync edits from CSV to JSONL")
    parser.add_argument("--export-to-csv", action="store_true", help="Export JSONL to CSV")
    args = parser.parse_args()

    if args.sync_from_csv:
        sync_from_csv()
    elif args.export_to_csv:
        export_to_csv()
    else:
        run_interactive_annotator(annotator_name=args.annotator)


if __name__ == "__main__":
    main()
