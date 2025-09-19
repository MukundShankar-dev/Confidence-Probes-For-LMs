import os
import numpy as np
from typing import Optional

from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression

class Probe:
    """
    Lightweight classifier over features with optional isotonic calibration.
    - predict_score(X): raw model probability (or scaled decision function)
    - predict_proba(X, calibrated=True): calibrated if a calibrator is present
    """
    def __init__(self, kind: str = "logreg"):
        self.kind = kind
        if kind == "logreg":
            self.clf = LogisticRegression(max_iter=200)
        else:
            self.clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=30)
        self.calibrator: Optional[IsotonicRegression] = None

    # ---------------- core model ----------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return raw positive-class probability if available, else a scaled score."""
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)[:, 1]
        # fallback via decision function scaled to [0,1]
        from sklearn.preprocessing import MinMaxScaler
        dec = self.clf.decision_function(X)
        return MinMaxScaler().fit_transform(dec.reshape(-1, 1))[:, 0]

    # ---------------- calibration ----------------
    def fit_calibrator(self, p_raw: np.ndarray, y: np.ndarray):
        """Fit an isotonic regression calibrator on validation raw probs."""
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(p_raw, y)

    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        p = self.predict_score(X)
        if calibrated and self.calibrator is not None:
            # sklearn's IsotonicRegression expects 1D
            return self.calibrator.predict(p)
        return p

    # ---------------- persistence ----------------
    def save(self, clf_path: str, calib_path: Optional[str] = None):
        dump(self.clf, clf_path)
        if calib_path is not None and self.calibrator is not None:
            dump(self.calibrator, calib_path)

    @classmethod
    def load(cls, clf_path: str, kind: str = "logreg",
             calib_path: Optional[str] = None) -> "Probe":
        obj = cls(kind)
        obj.clf = load(clf_path)
        if calib_path is not None and os.path.exists(calib_path):
            try:
                obj.calibrator = load(calib_path)
            except Exception:
                obj.calibrator = None
        return obj
