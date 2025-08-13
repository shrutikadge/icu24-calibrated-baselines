"""
Make figures: AUROC/AUPRC, calibration curve (pre/post), subgroup ECE bars, decision curve
"""
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.metrics.calibration import reliability_curve
from src.metrics.decision_curve import decision_curve

REPORTS_PATH = 'reports/'
FIGURES_PATH = REPORTS_PATH + 'figures/'
DOCS_FIGURES_PATH = 'docs/figures/'

os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(DOCS_FIGURES_PATH, exist_ok=True)

df = pd.read_csv(REPORTS_PATH + 'oof_predictions.csv')
with open(os.path.join(REPORTS_PATH, 'calibration_summary.json')) as f:
    cal_summ = json.load(f)

# AUROC/AUPRC by model
for model in df['model'].unique():
    y_true = df[df['model'] == model]['y_true'].values
    y_prob = df[df['model'] == model]['y_prob'].values
    # AUROC/AUPRC
    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.bar(['AUROC', 'AUPRC'], [auroc, auprc])
    plt.title(f'{model} AUROC/AUPRC')
    plt.savefig(FIGURES_PATH + f'{model}_auroc_auprc.png')
    plt.savefig(DOCS_FIGURES_PATH + f'{model}_auroc_auprc.png')
    plt.close()

    # Calibration curve pre/post
    pre = pd.read_csv(os.path.join(REPORTS_PATH, f'{model}_reliability_pre.csv'))
    post = pd.read_csv(os.path.join(REPORTS_PATH, f'{model}_reliability_post.csv'))
    plt.figure()
    plt.plot(pre['prob_pred'], pre['prob_true'], marker='o', label='Pre')
    plt.plot(post['prob_pred'], post['prob_true'], marker='s', label='Post (best)')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title(f'{model} Reliability Curve (Pre/Post)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.legend()
    plt.savefig(FIGURES_PATH + f'{model}_reliability.png')
    plt.savefig(DOCS_FIGURES_PATH + f'{model}_reliability.png')
    plt.close()

    # Subgroup ECE bars (sex, age bands)
    sub = pd.read_csv(os.path.join(REPORTS_PATH, f'subgroup_ece_{model}.csv'))
    plt.figure()
    plt.bar(sub['group'], sub['ECE'])
    plt.title(f'{model} Subgroup ECE (post)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH + f'{model}_subgroup_ece.png')
    plt.savefig(DOCS_FIGURES_PATH + f'{model}_subgroup_ece.png')
    plt.close()

# Decision curve
    curve = decision_curve(y_true, y_prob)
    thresholds, net_benefits = zip(*curve)
    plt.figure()
    plt.plot(thresholds, net_benefits)
    plt.title(f'{model} Decision Curve')
    plt.xlabel('Threshold')
    plt.ylabel('Net Benefit')
    plt.savefig(FIGURES_PATH + f'{model}_decision_curve.png')
    plt.savefig(DOCS_FIGURES_PATH + f'{model}_decision_curve.png')
    plt.close()
