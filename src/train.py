"""
train.py — Model zoo, deliberately small: baseline -> logistic regression
-> one tree model (HistGradientBoostingClassifier, chosen because it
ships with scikit-learn already, so there's no install risk under time
pressure). Whichever wins on the temporal validation set, on the metrics
that matter (PR-AUC / precision@k / net rupee value — not accuracy
alone), is the one that gets used for the real test predictions.

Every fit() below is called ONLY on the "fit" half of the temporal split.
Nothing here ever calls .fit() or .fit_transform() on the validation or
test set — that would leak the validation/test distribution into the
model or its preprocessing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


class DoNothingBaseline:
    """Predicts the observed fit-period fraud rate for every claim — i.e.
    it cannot rank claims at all. This exists to prove accuracy alone is
    a meaningless target here (it will score ~98.7% while catching 0
    fraud at any real operating threshold), not as a candidate we'd ship."""

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba_positive(self, X):
        return np.full(len(X), self.rate_)


class LogisticRegressionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, random_state=42)

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict_proba_positive(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)[:, 1]


class HGBModel:
    def __init__(self, class_weight=None):
        self.class_weight = class_weight
        self.model = HistGradientBoostingClassifier(random_state=42, class_weight=class_weight)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba_positive(self, X):
        return self.model.predict_proba(X)[:, 1]


class HGBModelBalanced(HGBModel):
    """Same model, class_weight='balanced'. Kept as a separate registry
    entry (not assumed better or worse) so the validation step can
    measure, on this data, whether reweighting helps ranking quality or
    just distorts the predicted probabilities — rather than assuming
    either outcome ahead of time."""
    def __init__(self):
        super().__init__(class_weight="balanced")


MODEL_REGISTRY = {
    "baseline_do_nothing": DoNothingBaseline,
    "logistic_regression": LogisticRegressionModel,
    "hist_gradient_boosting": HGBModel,
    "hist_gradient_boosting_balanced": HGBModelBalanced,
}


def fit_all(fit_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    X = fit_df[feature_cols]
    y = fit_df["is_fraud"].values
    fitted = {}
    for name, cls in MODEL_REGISTRY.items():
        fitted[name] = cls().fit(X, y)
    return fitted
