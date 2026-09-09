"""
Calibrated Intent Classifier module.
Combines Sentence Transformer embeddings with Multinomial Logistic Regression,
calibrated via Multiclass Temperature Scaling, and an auxiliary geometric centroid
detector for out-of-scope / novelty anomaly identification.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from src.hiver_agent.nlp.normalizer import TextNormalizer
from src.hiver_agent.nlp.calibration import MulticlassTemperatureScaler


class IntentClassifier:
    """
    Calibrated intent classifier for customer support inquiries.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        random_state: int = 42,
        novelty_threshold: float = 0.45
    ):
        self.embedding_model_name = embedding_model_name
        self.random_state = random_state
        self.novelty_threshold = novelty_threshold

        self.normalizer = TextNormalizer()
        self._embedder: Optional[SentenceTransformer] = None
        self.clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        )
        self.scaler = MulticlassTemperatureScaler()
        self.classes_: List[str] = []
        self.centroids_: Dict[str, np.ndarray] = {}
        self.is_trained: bool = False

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy load SentenceTransformer model."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generates normalized dense embeddings for a list of texts."""
        cleaned = [self.normalizer.normalize(t) for t in texts]
        embeddings = self.embedder.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None
    ) -> "IntentClassifier":
        """
        Trains Multinomial Logistic Regression on training embeddings,
        computes intent geometric centroids, and fits temperature scaling on validation set.
        """
        self.classes_ = sorted(list(set(train_labels)))
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_train = np.array([label_to_idx[l] for l in train_labels])

        # 1. Embed training texts
        X_train = self.encode(train_texts)

        # 2. Fit Multinomial Logistic Regression
        self.clf.fit(X_train, y_train)

        # 3. Compute geometric centroids per class for novelty detection
        self.centroids_ = {}
        for c in self.classes_:
            idx = label_to_idx[c]
            class_vecs = X_train[y_train == idx]
            if len(class_vecs) > 0:
                mean_vec = np.mean(class_vecs, axis=0)
                # Normalize centroid vector
                norm = np.linalg.norm(mean_vec)
                self.centroids_[c] = mean_vec / norm if norm > 0 else mean_vec
            else:
                self.centroids_[c] = np.zeros(X_train.shape[1])

        # 4. Calibrate via Temperature Scaling
        if val_texts and val_labels:
            X_val = self.encode(val_texts)
            val_logits = self.clf.decision_function(X_val)
            y_val = np.array([label_to_idx.get(l, 0) for l in val_labels])
            self.scaler.fit(val_logits, y_val)
        else:
            # Fit on training logits if separate validation split is not supplied
            train_logits = self.clf.decision_function(X_train)
            self.scaler.fit(train_logits, y_train)

        self.is_trained = True
        return self

    def predict_one(self, text: str) -> Dict[str, Any]:
        """
        Predicts intent for a single message with calibrated confidence,
        centroid distance, and alternative predictions.
        """
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained yet.")

        vec = self.encode([text])  # shape: (1, dim)
        raw_logits = self.clf.decision_function(vec)  # shape: (1, n_classes)
        calibrated_probs = self.scaler.predict_proba(raw_logits)[0]  # shape: (n_classes,)

        pred_idx = int(np.argmax(calibrated_probs))
        predicted_intent = self.classes_[pred_idx]
        confidence = float(calibrated_probs[pred_idx])
        raw_logit = float(raw_logits[0, pred_idx])

        # Compute cosine distance to class geometric centroids
        # Distance = 1 - cos(vec, centroid)
        centroid_distances = {}
        for c, centroid in self.centroids_.items():
            cos_sim = float(np.dot(vec[0], centroid))
            centroid_distances[c] = round(1.0 - cos_sim, 4)

        min_centroid_distance = min(centroid_distances.values()) if centroid_distances else 0.0
        is_novelty_outlier = bool(min_centroid_distance > self.novelty_threshold)

        # Ranked alternatives
        ranked_indices = np.argsort(calibrated_probs)[::-1]
        alternatives = [
            {
                "intent": self.classes_[idx],
                "confidence": round(float(calibrated_probs[idx]), 4)
            }
            for idx in ranked_indices[1:4]  # Top 3 runner-ups
        ]

        # Map full calibrated probability distribution
        calibrated_prob_map = {
            self.classes_[i]: round(float(calibrated_probs[i]), 6)
            for i in range(len(self.classes_))
        }

        return {
            "predicted_intent": predicted_intent,
            "calibrated_confidence": round(confidence, 4),
            "raw_logit": round(raw_logit, 4),
            "temperature": round(float(self.scaler.temperature), 4),
            "is_novelty_outlier": is_novelty_outlier,
            "min_centroid_distance": round(min_centroid_distance, 4),
            "centroid_distances": centroid_distances,
            "calibrated_probabilities": calibrated_prob_map,
            "prob_vector": [float(p) for p in calibrated_probs],
            "alternatives": alternatives
        }

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Computes calibrated multiclass probability distribution matrix for a list of texts.
        Returns shape (N, n_classes).
        """
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained yet.")
        vecs = self.encode(texts)
        raw_logits = self.clf.decision_function(vecs)
        return self.scaler.predict_proba(raw_logits)

    def save(self, path: Path) -> None:
        """Serializes classifier state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "embedding_model_name": self.embedding_model_name,
            "random_state": self.random_state,
            "novelty_threshold": self.novelty_threshold,
            "clf": self.clf,
            "scaler": self.scaler,
            "classes_": self.classes_,
            "centroids_": self.centroids_,
            "is_trained": self.is_trained
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "IntentClassifier":
        """Deserializes classifier state from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            embedding_model_name=state["embedding_model_name"],
            random_state=state["random_state"],
            novelty_threshold=state["novelty_threshold"]
        )
        obj.clf = state["clf"]
        obj.scaler = state["scaler"]
        obj.classes_ = state["classes_"]
        obj.centroids_ = state["centroids_"]
        obj.is_trained = state["is_trained"]
        return obj
