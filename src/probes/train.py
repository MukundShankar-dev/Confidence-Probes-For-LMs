import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from joblib import dump
from typing import Dict


class Probe:
    def __init__(self, kind: str = "logreg"):
        self.kind = kind
        if kind == "logreg":
            self.clf = LogisticRegression(max_iter=200)
        else:
            self.clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=30)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)[:, 1]
        # fallback via decision function
        from sklearn.preprocessing import MinMaxScaler
        dec = self.clf.decision_function(X)
        return MinMaxScaler().fit_transform(dec.reshape(-1, 1))[:, 0]

    def save(self, path: str):
        dump(self.clf, path)
