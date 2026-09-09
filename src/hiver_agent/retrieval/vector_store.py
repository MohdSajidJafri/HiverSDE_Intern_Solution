"""
Semantic evidence retrieval module.
Indexes historical customer-brand interactions and retrieves top-K evidence
with cosine similarity scoring and strict provenance metadata.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.hiver_agent.nlp.normalizer import TextNormalizer


class VectorStore:
    """
    Local semantic vector store for historical customer support resolutions.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.normalizer = TextNormalizer()
        self._embedder: Optional[SentenceTransformer] = None

        self.corpus_records: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, dim)
        self.is_built: bool = False

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes texts into L2-normalized dense embeddings."""
        cleaned = [self.normalizer.normalize(t) for t in texts]
        return self.embedder.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)

    def build_index(self, pairs: List[Dict[str, Any]]) -> "VectorStore":
        """
        Builds vector index from a list of reconstructed conversation pairs.
        Each pair must have 'customer_text' and 'brand_reply'.
        """
        self.corpus_records = pairs
        customer_texts = [p["customer_text"] for p in pairs]

        if not customer_texts:
            self.embeddings = np.empty((0, 384), dtype=np.float32)
            self.is_built = True
            return self

        self.embeddings = self.encode(customer_texts)
        self.is_built = True
        return self

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K most semantically similar historical customer interactions.
        Returns evidence records with similarity scores and complete provenance.
        """
        if not self.is_built or self.embeddings is None or len(self.corpus_records) == 0:
            return []

        query_vec = self.encode([query])  # shape: (1, dim), L2-normalized

        # Cosine similarity via dot product
        similarities = np.dot(self.embeddings, query_vec[0])  # shape: (N,)

        # Top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if similarity_threshold is not None and score < similarity_threshold:
                continue

            record = self.corpus_records[idx]
            results.append({
                "evidence_id": f"ev_{record.get('brand_tweet_id', idx)}",
                "similarity": round(score, 4),
                "intent": record.get("intent", record.get("metadata", {}).get("intent", "other_unsupported")),
                "historical_customer": record["customer_text"],
                "historical_brand_reply": record["brand_reply"],
                "conversation_id": record.get("conversation_id", ""),
                "customer_tweet_id": record.get("customer_tweet_id", ""),
                "brand_tweet_id": record.get("brand_tweet_id", ""),
                "metadata": record.get("metadata", {})
            })

        return results

    def save(self, path: Path) -> None:
        """Serializes vector store and corpus to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "embedding_model_name": self.embedding_model_name,
            "corpus_records": self.corpus_records,
            "embeddings": self.embeddings,
            "is_built": self.is_built
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Deserializes vector store and corpus from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(embedding_model_name=state["embedding_model_name"])
        obj.corpus_records = state["corpus_records"]
        obj.embeddings = state["embeddings"]
        obj.is_built = state["is_built"]
        return obj
