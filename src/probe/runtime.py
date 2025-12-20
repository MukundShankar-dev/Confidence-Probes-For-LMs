import json
import numpy as np
import torch
import torch.nn as nn

"""
This module is no longer in use... all probe usage is handled by files in scripts/
"""

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, hidden2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden2), nn.GELU(),
            nn.Linear(hidden2, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class ProbeRuntime:
    """
    Load a probe.pt (saved by probe_trainer.py) and provide a predict(features)->p_true function.
    Expect the same feature_spec as in training (vector_feats 256-d each + scalar_feats).
    """
    def __init__(self, probe_path: str, device: str = None):
        state = torch.load(probe_path, map_location="cpu")
        self.mu = state["mu"]; self.sd = state["sd"]
        self.feature_spec = state["feature_spec"]
        self.in_dim = state["in_dim"]
        self.model = MLP(self.in_dim, hidden=state["hidden"], hidden2=state["hidden2"])
        self.model.load_state_dict(state["model_state"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        self.vec_keys = self.feature_spec["vector_feats"]
        self.sca_keys = self.feature_spec["scalar_feats"]
        self.packed_dim = int(self.feature_spec.get("packed_dim_each", 256))

    def _stack_one(self, row: dict):
        feats = []
        # vectors
        for k in self.vec_keys:
            v = row.get(k)
            if isinstance(v, list):
                vv = (v[:self.packed_dim] + [0.0]*max(0, self.packed_dim - len(v)))
            else:
                vv = [0.0]*self.packed_dim
            feats.extend(float(x) for x in vv)
        # scalars
        for k in self.sca_keys:
            val = row.get(k, 0.0)
            feats.append(float(val if val is not None else 0.0))
        return np.asarray(feats, dtype=np.float32)

    def predict(self, feature_row: dict) -> float:
        x = self._stack_one(feature_row)[None, :]
        # standardize with training stats
        x = (x - self.mu) / (self.sd + 1e-8)
        with torch.no_grad():
            xt = torch.tensor(x, device=self.device)
            p = self.model(xt).clamp(0.0, 1.0).item()
        return float(p)
