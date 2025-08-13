"""
Decision curve analysis (net benefit per Vickers & Elkin)
"""
import numpy as np

def net_benefit(y_true, y_prob, threshold):
    tp = np.sum((y_prob >= threshold) & (y_true == 1))
    fp = np.sum((y_prob >= threshold) & (y_true == 0))
    n = len(y_true)
    nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return nb

def decision_curve(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    return [(t, net_benefit(y_true, y_prob, t)) for t in thresholds]
