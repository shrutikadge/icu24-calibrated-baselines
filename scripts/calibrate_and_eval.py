"""
Calibrate OOF predictions, select best calibration, compute ECE (with CI), subgroup metrics,
and write per-plot CSVs for figures.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from src.metrics.calibration import platt_calibration, isotonic_calibration, reliability_curve
from src.metrics.ece import compute_ece, bootstrap_ece

OOF_PATH = 'reports/oof_predictions.csv'
REPORTS_PATH = 'reports/'
N_BINS = 10
B = 1000

os.makedirs(REPORTS_PATH, exist_ok=True)

# Load OOF preds
df = pd.read_csv(OOF_PATH)
summary = {}

for model in df['model'].unique():
    mdf = df[df['model'] == model].copy()
    y_true = mdf['y_true'].values
    y_prob = mdf['y_prob'].values

    # Calibrators
    platt = platt_calibration(y_true, y_prob)
    y_prob_platt = platt.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    iso = isotonic_calibration(y_true, y_prob)
    y_prob_iso = iso.predict(y_prob)

    # Metrics pre/post
    brier_pre = brier_score_loss(y_true, y_prob)
    nll_pre = log_loss(y_true, y_prob)
    brier_platt = brier_score_loss(y_true, y_prob_platt)
    brier_iso = brier_score_loss(y_true, y_prob_iso)
    nll_platt = log_loss(y_true, y_prob_platt)
    nll_iso = log_loss(y_true, y_prob_iso)

    # Pick best by Brier; report NLL too
    if brier_platt <= brier_iso:
        best_name = 'platt'
        y_prob_best = y_prob_platt
        brier_best, nll_best = brier_platt, nll_platt
    else:
        best_name = 'isotonic'
        y_prob_best = y_prob_iso
        brier_best, nll_best = brier_iso, nll_iso

    # ECE with CI pre and post
    ece_pre, ece_pre_ci = bootstrap_ece(y_true, y_prob, n_bins=N_BINS, B=B)
    ece_post, ece_post_ci = bootstrap_ece(y_true, y_prob_best, n_bins=N_BINS, B=B)

    # Reliability curve CSVs
    prob_true_pre, prob_pred_pre = reliability_curve(y_true, y_prob, n_bins=N_BINS)
    pd.DataFrame({'prob_pred': prob_pred_pre, 'prob_true': prob_true_pre}).to_csv(
        os.path.join(REPORTS_PATH, f'{model}_reliability_pre.csv'), index=False)

    prob_true_post, prob_pred_post = reliability_curve(y_true, y_prob_best, n_bins=N_BINS)
    pd.DataFrame({'prob_pred': prob_pred_post, 'prob_true': prob_true_post}).to_csv(
        os.path.join(REPORTS_PATH, f'{model}_reliability_post.csv'), index=False)

    # Subgroup metrics on best-calibrated probabilities
    sub_rows = []
    # Sex
    for sex in ['F', 'M']:
        mask = (mdf['sex'] == sex).values
        if np.any(mask):
            auc = roc_auc_score(y_true[mask], y_prob_best[mask])
            ece_s, ece_ci = bootstrap_ece(y_true[mask], y_prob_best[mask], n_bins=N_BINS, B=200)
            sub_rows.append({'group': f'sex_{sex}', 'AUROC': auc, 'ECE': ece_s, 'ECE_L': ece_ci[0], 'ECE_U': ece_ci[1]})
    # Age bands
    ages = mdf['age'].values
    bands = {
        'age_lt50': ages < 50,
        'age_50_70': (ages >= 50) & (ages <= 70),
        'age_gt70': ages > 70,
    }
    for name, mask in bands.items():
        if np.any(mask):
            auc = roc_auc_score(y_true[mask], y_prob_best[mask])
            ece_s, ece_ci = bootstrap_ece(y_true[mask], y_prob_best[mask], n_bins=N_BINS, B=200)
            sub_rows.append({'group': name, 'AUROC': auc, 'ECE': ece_s, 'ECE_L': ece_ci[0], 'ECE_U': ece_ci[1]})

    pd.DataFrame(sub_rows).to_csv(os.path.join(REPORTS_PATH, f'subgroup_ece_{model}.csv'), index=False)

    # Summary aggregation
    summary[model] = {
        'brier_pre': brier_pre,
        'nll_pre': nll_pre,
        'brier_platt': brier_platt,
        'brier_iso': brier_iso,
        'nll_platt': nll_platt,
        'nll_iso': nll_iso,
        'best': best_name,
        'brier_best': brier_best,
        'nll_best': nll_best,
        'ece_pre': ece_pre,
        'ece_pre_ci': [float(ece_pre_ci[0]), float(ece_pre_ci[1])],
        'ece_post': ece_post,
        'ece_post_ci': [float(ece_post_ci[0]), float(ece_post_ci[1])],
    }

with open(os.path.join(REPORTS_PATH, 'calibration_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
