"""
Expected Calibration Error (ECE) with bootstrap CI
"""
import numpy as np
from sklearn.utils import resample

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(y_true)
    return ece

def bootstrap_ece(y_true, y_prob, n_bins=10, B=1000, random_state=42):
    np.random.seed(random_state)
    eces = []
    for _ in range(B):
        idx = resample(np.arange(len(y_true)), replace=True)
        eces.append(compute_ece(y_true[idx], y_prob[idx], n_bins))
    return np.mean(eces), np.percentile(eces, [2.5, 97.5])
