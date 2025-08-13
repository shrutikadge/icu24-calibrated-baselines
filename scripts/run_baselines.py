"""
Run baselines: age-only LR, EBM, GBDT; 5-fold CV; save OOF preds and metrics
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import json
from src.models.ebm import EBMWrapper
from src.models.gbdt import GBDTWrapper

DATA_PATH = 'data/synthetic/first24h_minimal.csv'
REPORTS_PATH = 'reports/'
N_SPLITS = 5
RANDOM_STATE = 42

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['label'])
y = df['label']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# One-hot encoder for categoricals (dense for compatibility)
# scikit-learn compatibility: 1.2+ uses sparse_output, older uses sparse
try:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", ohe, categorical_cols),
    ],
    remainder="passthrough",
)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

models = {
    'age_lr': LogisticRegression(),
    'ebm': EBMWrapper(),
    'gbdt': GBDTWrapper()
}

oof_preds = []
metrics = {m: [] for m in models}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit encoder on train, transform both
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Age-only LR (no encoding needed)
    models['age_lr'].fit(X_train_raw[['age']], y_train)
    age_probs = models['age_lr'].predict_proba(X_test_raw[['age']])[:, 1]

    # EBM
    models['ebm'].fit(X_train, y_train)
    ebm_probs = models['ebm'].predict_proba(X_test)

    # GBDT
    models['gbdt'].fit(X_train, y_train)
    gbdt_probs = models['gbdt'].predict_proba(X_test)

    # Save OOF preds with subgroup columns for downstream fairness eval
    for model, probs in zip(['age_lr', 'ebm', 'gbdt'], [age_probs, ebm_probs, gbdt_probs]):
        for i, (yt, yp) in enumerate(zip(y_test.values, probs)):
            rec = {
                'y_true': yt,
                'y_prob': float(yp),
                'model': model,
                'fold': fold,
                'age': int(X_test_raw.iloc[i]['age']),
                'sex': str(X_test_raw.iloc[i]['sex'])
            }
            oof_preds.append(rec)
        metrics[model].append({
            'AUROC': float(roc_auc_score(y_test, probs)),
            'AUPRC': float(average_precision_score(y_test, probs)),
            'Brier': float(brier_score_loss(y_test, probs))
        })

pd.DataFrame(oof_preds).to_csv(REPORTS_PATH + 'oof_predictions.csv', index=False)
with open(REPORTS_PATH + 'fold_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
