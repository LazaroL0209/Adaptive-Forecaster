import numpy as np
from scipy.stats import entropy
from scipy.special import kl_div

class DriftDetector:
    def __init__(self, threshold=0.3, window=30):
        self.threshold  = threshold
        self.window     = window
        self.reference  = None
        self.residuals  = []

    def set_reference(self, residuals: list):
        self.reference = np.array(residuals)

    def update(self, residual: float) -> bool:
        self.residuals.append(residual)
        if len(self.residuals) < self.window:
            return False
        recent = np.array(self.residuals[-self.window:])
        score  = self._kl_divergence(self.reference, recent)
        print(f"Drift score: {score:.4f} (threshold: {self.threshold})")
        return score > self.threshold

    def _kl_divergence(self, ref, curr):
        bins = np.linspace(
            min(ref.min(), curr.min()),
            max(ref.max(), curr.max()), 30)
        p, _ = np.histogram(ref,  bins=bins, density=True)
        q, _ = np.histogram(curr, bins=bins, density=True)
        p = p + 1e-9
        q = q + 1e-9
        p /= p.sum()
        q /= q.sum()
        return float(np.sum(kl_div(p, q)))