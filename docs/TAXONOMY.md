# Empirical Intent Taxonomy: SpotifyCares Support Domain

## 1. Provenance & Discovery Methodology
The intent taxonomy was derived through unsupervised semantic clustering of real customer queries directed to `@SpotifyCares`. Raw customer tweets were normalized, transformed into TF-IDF and dense embeddings, and clustered via KMeans into 12 empirical subclusters. The intermediate cluster output is preserved in `data/interim/intent_clusters.json` for auditing.

The 12 empirical subclusters were consolidated into **10 business-oriented operational intents** based on shared operational resolution paths and escalation boundaries.

## 2. Taxonomy Summary Table

| # | Intent Name | Discovered Clusters | Escalation Default | Operational Rationale |
|---|---|---|---|---|
| 1 | `playback_issues` | Clusters 0, 2 | **AUTO_HANDLE** | Audio glitches, music unexpectedly pausing, skipping tracks,... |
| 2 | `app_crash_technical` | Clusters 1, 7 | **AUTO_HANDLE** | Desktop or mobile application crashing immediately on launch... |
| 3 | `offline_downloads` | Clusters 3 | **AUTO_HANDLE** | Downloaded playlists greyed out, sync between desktop and ph... |
| 4 | `device_connectivity` | Clusters 4 | **AUTO_HANDLE** | Bluetooth disconnects, CarPlay/Android Auto failures, Chrome... |
| 5 | `playlist_library` | Clusters 5 | **AUTO_HANDLE (or ESCALATE if permanent account data loss is suspected)** | Disappeared playlists, missing liked songs, unrecoverable de... |
| 6 | `subscription_billing` | Clusters 6 | **ESCALATE (Sensitive Account/Financial Domain)** | Unauthorized or duplicate charges, student discount verifica... |
| 7 | `account_access_security` | Clusters 8 | **ESCALATE (Critical Security Risk)** | Account takeover, unauthorized email/password changes, locke... |
| 8 | `feature_request_ui` | Clusters 9 | **AUTO_HANDLE** | Feedback on UI redesigns, requesting real-time lyrics, expli... |
| 9 | `service_status_outage` | Clusters 10 | **AUTO_HANDLE** | Global or regional Spotify server outages, HTTP 500/502/503 ... |
| 10 | `other_unsupported` | Clusters 11 | **ESCALATE (Ambiguity / Out-of-scope fallback)** | Chit-chat, artist compliments, insults, spam, non-English no... |

## 3. Granular Intent Definitions & Operational Boundaries

### 1. `playback_issues`
- **Definition**: Audio glitches, music unexpectedly pausing, skipping tracks, or shuffle/repeat controls not functioning.
- **Discovered Subclusters**: [0, 2]
- **Inclusion Criteria**: Playback stopped, songs pause after seconds, shuffle repeats same songs, won't play audio.
- **Exclusion Criteria**: Complete app crash on launch (app_crash_technical), local files or offline download issues (offline_downloads).
- **Representative Sample**: *"Why does my music keep pausing every 10 seconds on my iPhone?"*
- **Escalation Policy Default**: **AUTO_HANDLE**
- **Downstream Handling**: Provide device restart steps, audio quality toggles, or cache clearing guidance.

### 2. `app_crash_technical`
- **Definition**: Desktop or mobile application crashing immediately on launch, freezing, black screens, or installation errors.
- **Discovered Subclusters**: [1, 7]
- **Inclusion Criteria**: App won't open, crashes after update, desktop app shuts down immediately.
- **Exclusion Criteria**: Songs pausing while app remains open (playback_issues), server outage 500 error (service_status_outage).
- **Representative Sample**: *"Spotify desktop app crashes instantly on Windows 10 after today's update."*
- **Escalation Policy Default**: **AUTO_HANDLE**
- **Downstream Handling**: Recommend clean reinstall procedure with official download link.

### 3. `offline_downloads`
- **Definition**: Downloaded playlists greyed out, sync between desktop and phone failing, or storage/SD card errors.
- **Discovered Subclusters**: [3]
- **Inclusion Criteria**: Offline playlist won't play on airplane mode, waiting to download status, greyed out songs.
- **Exclusion Criteria**: General audio playback stutter while streaming online (playback_issues).
- **Representative Sample**: *"My downloaded offline songs are all greyed out and won't play without wifi."*
- **Escalation Policy Default**: **AUTO_HANDLE**
- **Downstream Handling**: Check offline mode toggle in Settings, verify device storage, or advise re-downloading playlist.

### 4. `device_connectivity`
- **Definition**: Bluetooth disconnects, CarPlay/Android Auto failures, Chromecast, Echo, PS4, or Spotify Connect errors.
- **Discovered Subclusters**: [4]
- **Inclusion Criteria**: Won't connect to car bluetooth, Echo speaker not showing up in devices menu, Spotify Connect drops.
- **Exclusion Criteria**: Phone app freezing on launch (app_crash_technical).
- **Representative Sample**: *"Spotify Connect cannot discover my Amazon Echo speaker anymore."*
- **Escalation Policy Default**: **AUTO_HANDLE**
- **Downstream Handling**: Verify same Wi-Fi network, power-cycle peripheral device, toggle Bluetooth.

### 5. `playlist_library`
- **Definition**: Disappeared playlists, missing liked songs, unrecoverable deleted music, or library sorting problems.
- **Discovered Subclusters**: [5]
- **Inclusion Criteria**: All my playlists vanished, missing songs from library, cannot sort by recently added.
- **Exclusion Criteria**: Single song playback stutter (playback_issues).
- **Representative Sample**: *"Logged into my account and all my playlists from the past 4 years are gone!"*
- **Escalation Policy Default**: **AUTO_HANDLE (or ESCALATE if permanent account data loss is suspected)**
- **Downstream Handling**: Guide user to web account page for 'Recover playlists' tool; escalate if multi-account confusion.

### 6. `subscription_billing`
- **Definition**: Unauthorized or duplicate charges, student discount verification, family plan billing, cancellation requests.
- **Discovered Subclusters**: [6]
- **Inclusion Criteria**: Charged $9.99 twice, cancel premium, renewal charge issue, student discount rejected.
- **Exclusion Criteria**: Free user asking how to shuffle (playback_issues).
- **Representative Sample**: *"I was charged twice on my credit card for Premium this month. I want a refund."*
- **Escalation Policy Default**: **ESCALATE (Sensitive Account/Financial Domain)**
- **Downstream Handling**: Do not fabricate refunds or timelines. Immediately escalate to human support via secure channel.

### 7. `account_access_security`
- **Definition**: Account takeover, unauthorized email/password changes, locked credentials, or login failures.
- **Discovered Subclusters**: [8]
- **Inclusion Criteria**: Someone hacked my account, email was changed without permission, password reset email not arriving.
- **Exclusion Criteria**: Trouble finding a song in library (playlist_library).
- **Representative Sample**: *"Someone in Russia changed the email on my Spotify account and I can't log in."*
- **Escalation Policy Default**: **ESCALATE (Critical Security Risk)**
- **Downstream Handling**: Do not request passwords. Escalate immediately to specialized account security team.

### 8. `feature_request_ui`
- **Definition**: Feedback on UI redesigns, requesting real-time lyrics, explicit song filter toggle, or feature suggestions.
- **Discovered Subclusters**: [9]
- **Inclusion Criteria**: Please bring back the old layout, add lyrics feature, explicit content search filter.
- **Exclusion Criteria**: Software bugs that break existing features (app_crash_technical).
- **Representative Sample**: *"Is there a way to filter search results to only show clean non-explicit versions?"*
- **Escalation Policy Default**: **AUTO_HANDLE**
- **Downstream Handling**: Acknowledge request politely, state current feature availability, share community ideas link.

### 9. `service_status_outage`
- **Definition**: Global or regional Spotify server outages, HTTP 500/502/503 errors, or platform maintenance mode.
- **Discovered Subclusters**: [10]
- **Inclusion Criteria**: Is Spotify down for everyone right now, server error 500 on all tracks, status page check.
- **Exclusion Criteria**: Single device connectivity failure on local Wi-Fi (device_connectivity).
- **Representative Sample**: *"Is Spotify completely down? All my friends say it's giving server error."*
- **Escalation Policy Default**: **AUTO_HANDLE**
- **Downstream Handling**: Confirm platform status, advise against reinstalling during known outage, link to @SpotifyStatus.

### 10. `other_unsupported`
- **Definition**: Chit-chat, artist compliments, insults, spam, non-English noise, or ambiguous questions.
- **Discovered Subclusters**: [11]
- **Inclusion Criteria**: Hello, nice music, what is your name, random emojis, non-support queries.
- **Exclusion Criteria**: Any actionable customer support inquiry with identifiable intent.
- **Representative Sample**: *"Hey you guys are really cool have a good day"*
- **Escalation Policy Default**: **ESCALATE (Ambiguity / Out-of-scope fallback)**
- **Downstream Handling**: If pleasantry, thank politely. If ambiguous, ask clarifying question or escalate.

## 4. Common Intent Confusions & Disambiguation Rules

| Intent A | Intent B | Disambiguation Boundary |
|---|---|---|
| `playback_issues` | `app_crash_technical` | If the app remains running but music stutters/stops, classify as `playback_issues`. If the app process terminates or freezes completely, classify as `app_crash_technical`. |
| `playback_issues` | `offline_downloads` | If songs fail specifically in offline mode or display greyed out icons, classify as `offline_downloads`. If streaming online, classify as `playback_issues`. |
| `device_connectivity` | `playback_issues` | If the failure involves peripheral hardware (Bluetooth, CarPlay, Echo, Chromecast), classify as `device_connectivity`. |
| `subscription_billing` | `account_access_security` | If the query involves charges, invoices, or payment methods, classify as `subscription_billing`. If it involves hacked credentials or locked logins, classify as `account_access_security`. |

## 5. Intermediate Artifact Reference
- Intermediate Cluster Data: `data/interim/intent_clusters.json`
- Reproduction Command: `python scripts/run_intent_discovery.py`
