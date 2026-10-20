"""Baselines. B0 never alarms. B1 is a logistic regression on summaries of the cached ACF."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def never_alarm(X):
    return np.zeros((len(X), 4))


def _features(X):
    """Mean and spread over the 64 bins for every (lag, link): 2 x 50 x 4 = 400 numbers."""
    return np.concatenate([X.mean(axis=3).reshape(len(X), -1), X.std(axis=3).reshape(len(X), -1)], axis=1)


def fit_b1(X, y):
    """Fit on training windows only (scaler included). Returns a function windows -> probabilities (n, 4)."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000))
    model.fit(_features(X), y)
    if model[-1].classes_.tolist() != [0, 1, 2, 3]:
        raise ValueError("training data must contain all four classes")
    return lambda Z: model.predict_proba(_features(Z))
