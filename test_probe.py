#!/usr/bin/env python3
"""Quick test to see why probe confidences are all 0.0"""

import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Load probe
probe_path = Path("backend_llama31/probe_mlp/probe_model.joblib")
probe = joblib.load(probe_path)

print("Probe loaded successfully")
print(f"Probe type: {type(probe)}")

# Create a test feature vector with all NaNs
test_features_nan = {
    "model_confidence": np.nan,
    "lp_mean": np.nan,
    "seq_conf": np.nan,
    "entropy_mean": np.nan,
    "entropy_std": np.nan,
    "margin_mean": np.nan,
    "margin_min": np.nan,
    "rescore_logp": np.nan,
    "answer_len": np.nan,
    "parsed_json_ok": np.nan,
    "parsed_p_true_ok": np.nan,
    "is_unknown": np.nan,
}

# Add hidden state features (all NaN)
for name in ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]:
    for j in range(256):
        test_features_nan[f"{name}_{j}"] = np.nan

# Create DataFrame
df_nan = pd.DataFrame([test_features_nan])
print(f"\nTest DataFrame shape: {df_nan.shape}")
print(f"Test DataFrame (first 5 cols): {df_nan.iloc[:, :5]}")

# Try prediction with all NaNs
try:
    prob_nan = probe.predict_proba(df_nan.values)[:, 1][0]
    print(f"\nProbe output with all NaNs: {prob_nan}")
except Exception as e:
    print(f"\nERROR with all NaNs: {e}")

# Now try with some reasonable values
test_features_good = {
    "model_confidence": 0.8,
    "lp_mean": -0.5,
    "seq_conf": 0.6,
    "entropy_mean": 1.2,
    "entropy_std": 0.3,
    "margin_mean": 2.5,
    "margin_min": 1.0,
    "rescore_logp": -1.5,
    "answer_len": 10,
    "parsed_json_ok": 1,
    "parsed_p_true_ok": 1,
    "is_unknown": 0,
}

# Add hidden state features (random values)
for name in ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]:
    for j in range(256):
        test_features_good[f"{name}_{j}"] = np.random.randn()

df_good = pd.DataFrame([test_features_good])
print(f"\nTest DataFrame with good values shape: {df_good.shape}")

try:
    prob_good = probe.predict_proba(df_good.values)[:, 1][0]
    print(f"Probe output with good values: {prob_good}")
except Exception as e:
    print(f"\nERROR with good values: {e}")