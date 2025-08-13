"""
Calibration utilities: Platt (sigmoid) and isotonic regression
Reliability curve plotting
"""
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
import numpy as np

# Platt scaling (sigmoid)
def platt_calibration(y_true, y_prob):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(y_prob.reshape(-1, 1), y_true)
    return lr

# Isotonic regression
def isotonic_calibration(y_true, y_prob):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(y_prob, y_true)
    return ir

# Reliability curve
def reliability_curve(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return prob_true, prob_pred
