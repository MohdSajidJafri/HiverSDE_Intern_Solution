# Annotation Guidelines: Hand-Labelled Gold Dataset

## 1. Purpose & Standards
These guidelines define the protocol used by human annotators to label customer inquiries for the Spotify customer support agent evaluation.

## 2. Labelling Dimensions
Each customer message is annotated with:
1. **`true_intent`**: One of the 10 consolidated intents discovered from data.
2. **`ground_truth_decision`**: `AUTO_HANDLE` or `ESCALATE`.
   - `AUTO_HANDLE`: Technical troubleshooting issues with actionable, safe standard procedures (cache, restart, settings).
   - `ESCALATE`: Billing disputes, unauthorized charges, hacked accounts, legal threats, or ungrounded bugs.
3. **`edge_case_type`**: `none`, `short`, `ambiguous`, `multi_intent`, `contradiction`, `sensitive`, or `out_of_scope`.
