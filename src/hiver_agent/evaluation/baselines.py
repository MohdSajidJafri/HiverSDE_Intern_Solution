"""
Baseline models for comparative benchmarking.
- Baseline 1 (Trivial): Majority Intent + Most Frequent Macro + Always Auto-Handle
- Baseline 2 (Simple): TF-IDF + Logistic Regression + Lexical Retrieval + Rule Escalation
"""

from typing import Dict, List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Baseline1Trivial:
    """
    Trivial Baseline:
    - Intent: Always predicts the majority class ('playback_issues')
    - Reply: Emits the most frequent canned brand response
    - Decision: Always AUTO_HANDLE (0% escalation)
    """

    MAJORITY_INTENT = "playback_issues"
    MOST_FREQUENT_REPLY = "Try restarting your device by holding the sleep/wake button. Keep us posted! /CH"

    def predict(self, text: str) -> Dict[str, Any]:
        return {
            "intent": {
                "predicted": self.MAJORITY_INTENT,
                "confidence": 1.0,
                "alternatives": []
            },
            "retrieval": {
                "evidence_count": 1,
                "top_similarity": 0.50,
                "evidence": [
                    {
                        "evidence_id": "ev_trivial",
                        "similarity": 0.50,
                        "historical_customer": "Music stops playing",
                        "historical_brand_reply": self.MOST_FREQUENT_REPLY
                    }
                ]
            },
            "decision": {
                "action": "AUTO_HANDLE",
                "reason_code": "TRIVIAL_ALWAYS_AUTO_HANDLE",
                "reason": "Trivial baseline always auto-handles."
            },
            "reply": {
                "draft": self.MOST_FREQUENT_REPLY,
                "grounded_in_evidence_ids": ["ev_trivial"],
                "unsupported_claims": []
            }
        }


class Baseline2Simple:
    """
    Simple Baseline:
    - Intent: Uncalibrated TF-IDF + Multinomial Logistic Regression
    - Retrieval: Lexical TF-IDF cosine matching
    - Decision: Simple keyword heuristic ('refund', 'cancel', 'hacked', 'stolen' -> ESCALATE)
    """

    SENSITIVE_KEYWORDS = ["refund", "cancel", "hacked", "stolen", "charge", "charged", "bill", "billing", "password"]

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.clf = LogisticRegression(solver="lbfgs", max_iter=500, random_state=42)
        self.retrieval_corpus: List[Dict[str, Any]] = []
        self.retrieval_tfidf = None
        self.is_trained = False

    def fit(self, train_pairs: List[Dict[str, Any]], intent_map: Dict[str, str]):
        """
        Fits TF-IDF and Logistic Regression on training pairs with assigned intents.
        """
        self.retrieval_corpus = train_pairs
        texts = [p["customer_text"] for p in train_pairs]
        labels = [intent_map.get(p["customer_text"], "other_unsupported") for p in train_pairs]

        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self.retrieval_tfidf = X
        self.is_trained = True
        return self

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_trained:
            # Fallback if not explicitly trained
            return Baseline1Trivial().predict(text)

        vec = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(vec)[0]
        pred_idx = int(np.argmax(probs))
        pred_intent = self.clf.classes_[pred_idx]
        conf = float(probs[pred_idx])

        # Lexical retrieval via TF-IDF cosine similarity
        sims = (self.retrieval_tfidf * vec.T).toarray().flatten()
        top_idx = int(np.argmax(sims)) if len(sims) > 0 else 0
        top_sim = float(sims[top_idx]) if len(sims) > 0 else 0.0
        top_pair = self.retrieval_corpus[top_idx] if self.retrieval_corpus else {}

        # Keyword escalation heuristic
        text_lower = text.lower()
        should_escalate = any(k in text_lower for k in self.SENSITIVE_KEYWORDS) or (conf < 0.50)

        action = "ESCALATE" if should_escalate else "AUTO_HANDLE"
        reason = "Keyword rule matched sensitive query" if should_escalate else "Simple baseline auto-handle"

        reply = top_pair.get("brand_reply", "Thanks for reaching out! /CH")

        return {
            "intent": {
                "predicted": pred_intent,
                "confidence": round(conf, 4),
                "alternatives": []
            },
            "retrieval": {
                "evidence_count": 1,
                "top_similarity": round(top_sim, 4),
                "evidence": [
                    {
                        "evidence_id": f"ev_{top_pair.get('brand_tweet_id', 0)}",
                        "similarity": round(top_sim, 4),
                        "historical_customer": top_pair.get("customer_text", ""),
                        "historical_brand_reply": reply
                    }
                ]
            },
            "decision": {
                "action": action,
                "reason_code": "KEYWORD_RULE_ESCALATION" if should_escalate else "SIMPLE_AUTO_HANDLE",
                "reason": reason
            },
            "reply": {
                "draft": reply,
                "grounded_in_evidence_ids": [f"ev_{top_pair.get('brand_tweet_id', 0)}"],
                "unsupported_claims": []
            }
        }
