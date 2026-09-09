# Human Gold Annotation Guidelines: @SpotifyCares Inquiries

## 1. Overview & Objective
This document defines the official annotation protocol for the **200 held-out customer inquiries** in `data/gold/gold_annotation_queue.jsonl`. 

Once annotated, this dataset becomes the **final, frozen Human Gold Benchmark (`GOLD_HUMAN`)** used strictly for held-out evaluation.
- **Strict Quarantine Rule**: Gold labels are **never** used for model training, threshold tuning, temperature calibration, or vector search indexing.
- **Zero Fabrication**: Every label must reflect actual human judgment.

---

## 2. Annotation Methods (Choose Any)

### Option A (Recommended): Interactive Terminal Annotator
Run the interactive CLI tool to review and label inquiries one-by-one with keyboard shortcuts and auto-save:
```powershell
python scripts/annotate_gold.py
```
- Prompts for Intent (keys `1`–`10`) and Decision (`1` for `AUTO_HANDLE`, `2` for `ESCALATE`).
- Shows customer text, timestamp, customer ID, and historical brand reply.
- Automatically saves progress after each record; resume anytime.

### Option B: Direct JSONL Editing
Open `data/gold/gold_annotation_queue.jsonl` in your IDE or text editor and fill in the blank fields:
```json
{
  "id": "cand_001",
  ...
  "gold_intent": "device_connectivity",
  "ground_truth_decision": "AUTO_HANDLE",
  "is_sensitive": false,
  "annotator": "your_name",
  "annotation_notes": "Sonos speaker connection issue"
}
```

### Option C: Spreadsheet Editing (CSV)
Open `data/gold/gold_annotation_queue.csv` in Excel or Google Sheets, fill in the columns (`gold_intent`, `ground_truth_decision`, `is_sensitive`, `annotator`), save as CSV, and sync to JSONL:
```powershell
python scripts/annotate_gold.py --sync-from-csv
```

---

## 3. The 10 Operational Intents Reference

Every customer message must be assigned **exactly one** of the following 10 intents:

| # | Intent Identifier | Core Meaning & Scope | Key Indicators | Default Decision | Example Message |
|---|---|---|---|---|---|
| **1** | `playback_issues` | Music stops, pauses, skips, stuttering, repeat/shuffle broken, distorted sound. | *stops after 10s, keeps pausing, skips track, repeat broken, audio cuts out* | `AUTO_HANDLE` | *"Why does Spotify stop playing after 5 seconds on every song?"* |
| **2** | `app_crash_technical` | Desktop/mobile app crashes on launch, freezes, black/blank screen, won't open. | *crash, freezes, black screen, closes immediately, won't launch, error code* | `AUTO_HANDLE` | *"Spotify desktop app crashes on Windows 10 launch every single time."* |
| **3** | `offline_downloads` | Downloaded songs greyed out, offline sync failing, airplane mode playback fails, download limits. | *offline mode, download greyed out, sync failed, storage limit, won't download* | `AUTO_HANDLE` | *"My offline playlists won't play when I put my iPhone in airplane mode."* |
| **4** | `device_connectivity` | External audio hardware: Bluetooth, CarPlay, Android Auto, Sonos, Chromecast, Echo, Spotify Connect. | *bluetooth, carplay, sonos, alexa, echo, chromecast, connect to speaker* | `AUTO_HANDLE` | *"My Sonos speaker is visible in Spotify Connect but won't connect when tapped."* |
| **5** | `playlist_library` | Playlists disappeared, songs unliked, library sorting broken, duplicate tracks, missing albums. | *lost playlists, liked songs gone, library empty, restore playlist, missing album* | `AUTO_HANDLE` | *"All my saved playlists from the last 3 years disappeared after the update!"* |
| **6** | `subscription_billing` | Charged twice, cancel renewal, student discount validation, refund request, invoice inquiry. | *charged twice, double bill, refund, cancel premium, student discount, receipt* | **`ESCALATE`** | *"I was charged $9.99 twice for Spotify Premium this month. I want a refund."* |
| **7** | `account_access_security` | Hacked account, unauthorized email/password change, locked out, lost credentials. | *hacked, unauthorized email, password reset link not working, account compromised* | **`ESCALATE`** | *"Someone changed my account email to a Russian domain and I can't log in."* |
| **8** | `feature_request_ui` | Request for new product features, UI layout complaints, explicit filter toggle, lyrics display. | *bring back lyrics, add search filter, UI redesign, dark mode, feature request* | `AUTO_HANDLE` | *"Please add a feature to filter clean songs from search results."* |
| **9** | `service_status_outage` | Inquiries asking if Spotify servers are down, widespread service outage, HTTP 500/503 errors. | *is spotify down, server error 500, service status, outage, down for everyone* | `AUTO_HANDLE` | *"Is Spotify down for everyone right now? Getting server error 500."* |
| **10** | `other_unsupported` | Social banter, greetings, non-English tweets, vague statements, out-of-scope non-Spotify queries. | *thanks, hello, emoji only, non-English, off-topic, DM sent* | **`ESCALATE`** / `AUTO_HANDLE`* | *"Good morning @SpotifyCares hope you have a nice day"* |

*\*Note on `other_unsupported`: If the customer message is purely conversational/greeting or non-English, `AUTO_HANDLE` with a friendly redirect/acknowledgment is acceptable. If it is an ungrounded or out-of-scope question (e.g. asking for legal advice or competitor products), label `ESCALATE`.*

---

## 4. Ground-Truth Decision Rules (`ground_truth_decision`)

Assign either `AUTO_HANDLE` or `ESCALATE` based on risk:

### When to assign `AUTO_HANDLE`:
The issue can be safely and definitively addressed by standard automated technical troubleshooting or publicly documented product guidance:
- Standard technical fixes (reinstalling app, clearing local cache, toggling offline mode, restarting device).
- Hardware pairing instructions (checking Bluetooth settings, verifying Spotify Connect permissions).
- Product usage guidance (how to restore deleted playlists via web account page, explaining the 3,333 track download limit).
- Feedback acknowledgment (thanking customer for a feature suggestion, noting it has been shared with the product team).

### When to assign `ESCALATE`:
The issue involves financial risk, identity security, legal exposure, or requires backend database modifications:
1. **Financial & Billing (`subscription_billing`)**: Any dispute over charges, refund demands, billing frequency, or credit card updates. An automated AI must never commit financial refunds autonomously.
2. **Account Takeover & Security (`account_access_security`)**: Account theft, compromised credentials, or email address changes. Requires human security verification.
3. **Legal Threats or Abuse**: Inquiries mentioning lawsuits, lawyers, harassment, or severe escalation.
4. **Ungrounded or Out-of-Scope Queries**: Questions where no safe standard troubleshooting procedure exists.

---

## 5. Sensitivity Flag (`is_sensitive`)

- Set `is_sensitive = true` if the message mentions:
  - Personal identifiable information (PII), passwords, email changes, login credentials.
  - Credit card charges, bank statements, refunds, money disputes.
  - Account compromise, hacking, unauthorized access.
  *(Rule of thumb: All `subscription_billing` and `account_access_security` records have `is_sensitive = true`).*
- Set `is_sensitive = false` for all standard technical troubleshooting, feature feedback, and general public inquiries.

---

## 6. Disambiguation & Edge Cases

1. **Multi-Intent / Compound Messages**:
   - *Example*: *"Spotify desktop app keeps crashing and it also charged me twice."*
   - *Rule*: Prioritize the higher-risk intent $\longrightarrow$ `subscription_billing` + `ESCALATE`.
2. **Follow-Up / Ambiguous Messages**:
   - *Example*: *"I tried that link and it didn't work."*
   - *Rule*: Check `historical_brand_reply` to understand context. If it refers to an app crash troubleshooting link, label `app_crash_technical`.
3. **Non-English Messages**:
   - *Rule*: Label `other_unsupported`. Decision: `AUTO_HANDLE` (standard multi-lingual routing or English-only disclaimer).
4. **Short / Noisy Tweets**:
   - *Example*: *"ugh spotify"* $\longrightarrow$ `other_unsupported` + `AUTO_HANDLE`.
   - *Example*: *"broken again"* $\longrightarrow$ `playback_issues` (or `other_unsupported` if completely ambiguous).

---

## 7. Quality Checklist Before Final Evaluation

Before running the final evaluation, ensure:
1. All **200 records** have non-empty `gold_intent`.
2. All **200 records** have `ground_truth_decision` set to `AUTO_HANDLE` or `ESCALATE`.
3. All **200 records** have `is_sensitive` set to `true` or `false`.
4. The `annotator` field contains your identifier (e.g. `"human_reviewer"`).
5. No duplicate IDs exist (`cand_001` through `cand_200` are unique).

---

## 8. Executing the Final Gold Evaluation

Once annotation is complete, run the automated master command:
```powershell
python scripts/run_final_gold_evaluation.py
```
This single command will:
1. Validate all 200 records for strict completeness, valid taxonomy intents, and absence of duplicates.
2. Execute the frozen Primary Agent and Baselines against the Human Gold Benchmark.
3. Compute Stratified and Natural-Distribution metrics, ECE, Brier score, and safety metrics.
4. Regenerate `docs/FINAL_REPORT.md`, `README.md`, and `reports/results/evaluation_results.json` with official Human Gold metrics.
