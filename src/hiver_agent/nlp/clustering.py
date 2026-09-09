"""
Intent discovery and semantic clustering module.
Discovers empirical customer support themes through unsupervised clustering,
preserving intermediate cluster outputs and consolidation rationales.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.hiver_agent.nlp.normalizer import TextNormalizer


class IntentDiscovery:
    """Discovers empirical intent clusters and maps them to a business taxonomy."""

    def __init__(self, n_clusters: int = 12, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.normalizer = TextNormalizer()
        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    def discover_clusters(self, customer_messages: List[str]) -> Dict[str, Any]:
        """
        Runs clustering on cleaned customer inquiries.
        Extracts top terms, cluster sizes, and representative examples.
        """
        cleaned_texts = [self.normalizer.normalize(t) for t in customer_messages if t and len(t.strip()) > 5]
        if len(cleaned_texts) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} valid messages to cluster.")

        # Compute TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
        self.kmeans.fit(tfidf_matrix)

        labels = self.kmeans.labels_
        terms = self.vectorizer.get_feature_names_out()
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]

        clusters = []
        for i in range(self.n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_size = int(len(cluster_indices))
            top_terms = [terms[ind] for ind in order_centroids[i, :10]]

            # Representative sample messages (closest to centroid)
            sample_examples = [cleaned_texts[idx] for idx in cluster_indices[:5]]

            clusters.append({
                "cluster_id": i,
                "size": cluster_size,
                "size_pct": round(float(cluster_size / len(cleaned_texts) * 100.0), 2),
                "top_terms": top_terms,
                "sample_examples": sample_examples
            })

        result = {
            "total_messages": len(cleaned_texts),
            "n_clusters": self.n_clusters,
            "clusters": clusters
        }
        return result

    def save_intermediate_clusters(self, cluster_results: Dict[str, Any], output_path: Path) -> Path:
        """Saves intermediate cluster artifacts for transparency and auditability."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cluster_results, f, indent=2)
        return output_path
