"""
GBDT wrapper: prefer LightGBM, then XGBoost; fallback to RandomForestClassifier.
"""

class GBDTWrapper:
    def __init__(self, backend='auto', **kwargs):
        self.backend = None
        self.model = None
        self._init_model(backend, **kwargs)

    def _init_model(self, backend, **kwargs):
        # Try in order: explicit backend, else auto-detect
        candidates = []
        if backend in ('lightgbm', 'xgboost'):
            candidates = [backend]
        else:
            candidates = ['lightgbm', 'xgboost']

        for cand in candidates:
            try:
                if cand == 'lightgbm':
                    import lightgbm as lgb
                    self.model = lgb.LGBMClassifier(**kwargs)
                    self.backend = 'lightgbm'
                    return
                if cand == 'xgboost':
                    import xgboost as xgb
                    self.model = xgb.XGBClassifier(eval_metric='logloss', **kwargs)
                    self.backend = 'xgboost'
                    return
            except Exception:
                continue
        # Fallback to sklearn RandomForest if no GBDT backend available
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**kwargs)
        self.backend = 'random_forest'

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict_proba(self, X):
        import numpy as np
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        # Fallback: decision_function -> sigmoid
        if hasattr(self.model, 'decision_function'):
            s = self.model.decision_function(X)
            return 1 / (1 + np.exp(-s))
        # Last resort: predictions as probabilities
        return self.model.predict(X)

    def predict(self, X):
        return self.model.predict(X)
