from typing import Dict, List
from ..probes.train import Probe


class ConfidenceGate:
    def __init__(self, probe: Probe, threshold: float):
        self.probe = probe
        self.tau = threshold

    def decide(self, feat_vec) -> bool:
        # True = answer directly; False = RAG/refuse
        score = self.probe.predict_proba(feat_vec)[0]
        return bool(score >= self.tau), float(score)
