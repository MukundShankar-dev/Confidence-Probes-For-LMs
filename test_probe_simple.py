#!/usr/bin/env python3
"""
Test probe with manually constructed features that should give high confidence
"""

import joblib
import pandas as pd
import numpy as np

print("="*80)
print("PROBE TEST WITH KNOWN GOOD FEATURES")
print("="*80)

# Load probe
probe = joblib.load('backend_llama31/probe_mlp/probe_model.joblib')
print("\n✓ Probe loaded")

# These are features from a correct answer with high model confidence
# (from your chat debug output)
features = {
    'model_confidence': 1.0,
    'lp_mean': -0.038756537337879614,
    'seq_conf': 0.9619848880400633,
    'entropy_mean': 0.11453838956519273,
    'entropy_std': 0.17644415272598407,
    'margin_mean': 7.504092852274577,
    'margin_min': 1.34375,
    'rescore_logp': -3.4242673539556563,
    'answer_len': 13,
    'parsed_json_ok': 1.0,
    'parsed_p_true_ok': 1.0,
    'is_unknown': 0.0,
}

# Add hidden states (from your debug - actual values from the model)
# Using actual values from your output
h_last_sample = [0.311279296875, 0.845703125, 0.35986328125]
h_pool_sample = [0.5849609375, -0.9931640625, 1.001953125]
h_last_mid_sample = [-0.08538818359375, -0.0467529296875, 0.0625]
h_pool_mid_sample = [0.005657196044921875, -0.10833740234375, -0.075439453125]

# Extend to 256 with random values (placeholder)
np.random.seed(42)
for i in range(256):
    features[f'h_last_256_{i}'] = h_last_sample[i % 3] if i < 3 else np.random.randn() * 0.1
    features[f'h_pool_256_{i}'] = h_pool_sample[i % 3] if i < 3 else np.random.randn() * 0.1
    features[f'h_last_mid_256_{i}'] = h_last_mid_sample[i % 3] if i < 3 else np.random.randn() * 0.1
    features[f'h_pool_mid_256_{i}'] = h_pool_mid_sample[i % 3] if i < 3 else np.random.randn() * 0.1

df = pd.DataFrame([features])

print(f"\nDataFrame shape: {df.shape}")
print(f"Column order (first 20): {list(df.columns[:20])}")
print(f"\nScalar values:")
for k in ['model_confidence', 'lp_mean', 'seq_conf', 'entropy_mean', 'entropy_std', 
          'margin_mean', 'margin_min', 'rescore_logp', 'answer_len', 
          'parsed_json_ok', 'parsed_p_true_ok', 'is_unknown']:
    print(f"  {k}: {df[k].values[0]}")

# Test prediction
prob = probe.predict_proba(df.values)[:, 1][0]

print(f"\n{'='*80}")
print(f"PREDICTION")
print(f"{'='*80}")
print(f"\nProbability: {prob:.6f}")
print(f"\nExpected: High confidence (>0.5) since:")
print(f"  - model_confidence = 1.0 (model very confident)")
print(f"  - Answer is correct (Paris)")
print(f"  - All features look good")

if prob < 0.01:
    print(f"\n❌ PROBE IS BROKEN!")
    print(f"\nThe probe outputs near-zero even with:")
    print(f"  - High model confidence (1.0)")
    print(f"  - Good generation stats (low entropy, high margin)")
    print(f"  - All metadata flags correct")
    print(f"  - Real hidden states from model")
    print(f"\nThis means either:")
    print(f"  1. Probe was trained incorrectly")
    print(f"  2. Probe file is corrupted")
    print(f"  3. StandardScaler normalization is broken")
    print(f"  4. Feature order STILL doesn't match training")
    
    # Check scaler
    if hasattr(probe, 'steps'):
        print(f"\n{'='*80}")
        print(f"CHECKING SCALER")
        print(f"{'='*80}")
        scaler = probe.steps[1][1]
        scaled = scaler.transform(df.values)
        print(f"Scaled values (first 12): {scaled[0, :12]}")
        print(f"Scaled min/max/mean: {scaled.min():.3f} / {scaled.max():.3f} / {scaled.mean():.3f}")
        
        # Check if mean/std are reasonable
        print(f"\nScaler mean (first 12): {scaler.mean_[:12]}")
        print(f"Scaler std (first 12): {scaler.scale_[:12]}")
        
        # Manually check one feature
        raw_val = df['model_confidence'].values[0]
        scaler_mean = scaler.mean_[list(df.columns).index('model_confidence')]
        scaler_std = scaler.scale_[list(df.columns).index('model_confidence')]
        expected_scaled = (raw_val - scaler_mean) / scaler_std
        actual_scaled = scaled[0, list(df.columns).index('model_confidence')]
        print(f"\nManual check for model_confidence:")
        print(f"  Raw: {raw_val}")
        print(f"  Scaler mean: {scaler_mean}")
        print(f"  Scaler std: {scaler_std}")
        print(f"  Expected scaled: {expected_scaled}")
        print(f"  Actual scaled: {actual_scaled}")
        print(f"  Match: {np.isclose(expected_scaled, actual_scaled)}")

else:
    print(f"\n✓ PROBE WORKS! (prob = {prob:.3f})")
    print(f"\nThis means chat_inference is extracting features differently than expected.")