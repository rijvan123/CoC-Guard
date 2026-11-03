"""Logistic regression and LightGBM baselines on structured static attributes (Table A4)."""
from __future__ import annotations

import numpy as np


class LogisticRegressionBaseline:
    """L2-regularised logistic regression, C = 1.0 (Table A4)."""

    def __init__(self, C: float = 1.0, max_iter: int = 300, seed: int = 11):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)

    def fit(self, X, y, X_dev=None, y_dev=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LightGBMBaseline:
    """LightGBM with 1,000 trees, feature fraction 0.80 and learning rate 0.05 (Table A4)."""

    def __init__(self, n_estimators: int = 1000, learning_rate: float = 0.05, feature_fraction: float = 0.8,
                 seed: int = 11, early_stopping_rounds: int | None = 50, n_jobs: int = -1, num_leaves: int = 63):
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, colsample_bytree=feature_fraction,
                           random_state=seed, n_jobs=n_jobs, num_leaves=num_leaves, verbose=-1)
        self.early = early_stopping_rounds
        self.model = None

    def fit(self, X, y, X_dev=None, y_dev=None):
        import warnings
        import lightgbm as lgb
        warnings.filterwarnings("ignore", message=".*eval_set.*", category=Warning)
        self.model = lgb.LGBMClassifier(**self.params)
        kwargs = {}
        if X_dev is not None and self.early:
            kwargs = {"eval_set": [(X_dev, y_dev)], "callbacks": [lgb.early_stopping(self.early, verbose=False)]}
        self.model.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_
