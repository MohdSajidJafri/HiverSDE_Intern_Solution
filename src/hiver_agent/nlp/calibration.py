"""
Multiclass Temperature Scaling calibration module.
Calibrates multinomial logit outputs via temperature scaling T > 0,
and computes diagnostic calibration metrics (ECE, Brier score, reliability diagrams).
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


class MulticlassTemperatureScaler:
    """
    Calibrates multiclass logits by optimizing a single temperature parameter T > 0.
    Directly aligns output confidence with empirical probability while strictly
    preserving argmax classification predictions.
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted: bool = False

    def _cross_entropy(self, t: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> float:
        """Computes multiclass cross-entropy (NLL) for a given temperature t."""
        scaled_logits = logits / t[0]
        # Log-sum-exp trick for numerical stability
        max_logits = np.max(scaled_logits, axis=1, keepdims=True)
        log_sum_exp = max_logits + np.log(np.sum(np.exp(scaled_logits - max_logits), axis=1, keepdims=True))
        log_probs = scaled_logits - log_sum_exp

        # NLL over true class labels
        n = len(labels)
        nll = -np.sum(log_probs[np.arange(n), labels]) / n
        return float(nll)

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> "MulticlassTemperatureScaler":
        """
        Optimizes scalar temperature T > 0 via L-BFGS on validation logits.
        """
        val_logits = np.asarray(val_logits, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.int64)

        init_t = np.array([1.0])
        bounds = [(0.05, 10.0)]

        res = minimize(
            self._cross_entropy,
            init_t,
            args=(val_logits, val_labels),
            method="L-BFGS-B",
            bounds=bounds
        )

        self.temperature = float(res.x[0]) if res.success else 1.0
        self.is_fitted = True
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Returns calibrated probabilities for input logits."""
        t = self.temperature if self.is_fitted else 1.0
        scaled = np.asarray(logits, dtype=np.float64) / t
        return softmax(scaled, axis=1)

    @staticmethod
    def compute_ece(
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Computes Expected Calibration Error (ECE) and reliability diagram data.
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(labels)
        bins_data = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
            bin_size = int(np.sum(in_bin))

            if bin_size > 0:
                bin_acc = float(np.mean(accuracies[in_bin]))
                bin_conf = float(np.mean(confidences[in_bin]))
                bin_error = abs(bin_acc - bin_conf)
                ece += (bin_size / n_total) * bin_error

                bins_data.append({
                    "bin_index": i,
                    "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                    "bin_size": bin_size,
                    "bin_accuracy": round(bin_acc, 4),
                    "bin_confidence": round(bin_conf, 4),
                    "calibration_gap": round(bin_error, 4)
                })
            else:
                bins_data.append({
                    "bin_index": i,
                    "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                    "bin_size": 0,
                    "bin_accuracy": 0.0,
                    "bin_confidence": 0.0,
                    "calibration_gap": 0.0
                })

        return round(float(ece), 4), bins_data

    @staticmethod
    def compute_brier_score(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
        """
        Computes multiclass Brier score:
        Brier = (1/N) * sum_n sum_k (p_nk - y_nk)^2
        """
        n = len(labels)
        one_hot = np.zeros((n, n_classes))
        one_hot[np.arange(n), labels] = 1.0
        brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
        return round(float(brier), 4)
