"""
EBM (Explainable Boosting Machine) wrapper.
Tries to use interpret's ExplainableBoostingClassifier, and falls back to
scikit-learn LogisticRegression if interpret is unavailable.
"""

class EBMWrapper:
    def __init__(self, **kwargs):
        self.is_fallback = False
        self._init_model(**kwargs)

    def _init_model(self, **kwargs):
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
            self.model = ExplainableBoostingClassifier(**kwargs)
        except Exception:
            # Fallback to LogisticRegression (transparent baseline)
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000)
            self.is_fallback = True

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict_proba(self, X):
        import numpy as np
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        # Some sklearn models may not implement predict_proba
        preds = self.model.decision_function(X)
        # Sigmoid approximation
        return 1 / (1 + np.exp(-preds))

    def predict(self, X):
        return self.model.predict(X)
