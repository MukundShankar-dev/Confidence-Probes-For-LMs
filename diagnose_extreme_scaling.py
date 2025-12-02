#!/usr/bin/env python3
"""
Find which features are causing extreme scaled values
"""

import joblib
import pandas as pd
import numpy as np

probe = joblib.load('backend_llama31/probe_mlp/probe_model.joblib')
scaler = probe.steps[1][1]

# Same features as before
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

# Add hidden states
np.random.seed(42)
for i in range(256):
    features[f'h_last_256_{i}'] = np.random.randn() * 0.1  # Small random values
    features[f'h_pool_256_{i}'] = np.random.randn() * 0.1
    features[f'h_last_mid_256_{i}'] = np.random.randn() * 0.1
    features[f'h_pool_mid_256_{i}'] = np.random.randn() * 0.1

df = pd.DataFrame([features])
scaled = scaler.transform(df.values)

print("="*80)
print("FINDING EXTREME SCALED VALUES")
print("="*80)

# Find features with extreme scaled values
extreme_threshold = 10.0
extreme_indices = np.where(np.abs(scaled[0]) > extreme_threshold)[0]

print(f"\nFound {len(extreme_indices)} features with |scaled_value| > {extreme_threshold}")
print(f"\nTop 20 most extreme:")

# Sort by absolute value
sorted_indices = sorted(extreme_indices, key=lambda i: abs(scaled[0, i]), reverse=True)[:20]

for idx in sorted_indices:
    col_name = df.columns[idx]
    raw_val = df.values[0, idx]
    scaled_val = scaled[0, idx]
    scaler_mean = scaler.mean_[idx]
    scaler_std = scaler.scale_[idx]
    
    print(f"\n{col_name}:")
    print(f"  Raw value: {raw_val:.6f}")
    print(f"  Scaled value: {scaled_val:.3f}")
    print(f"  Scaler mean: {scaler_mean:.6f}")
    print(f"  Scaler std: {scaler_std:.6f}")
    print(f"  Z-score: {(raw_val - scaler_mean) / scaler_std:.3f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Check if hidden states are the problem
hidden_indices = [i for i in extreme_indices if any(x in df.columns[i] for x in ['h_last', 'h_pool'])]
scalar_indices = [i for i in extreme_indices if i not in hidden_indices]

print(f"\nExtreme values in:")
print(f"  Hidden state features: {len(hidden_indices)}")
print(f"  Scalar features: {len(scalar_indices)}")

if len(hidden_indices) > len(scalar_indices):
    print("\n✓ Hidden states are the problem!")
    print("\nPossible causes:")
    print("  1. Chat extracts hidden states differently than training")
    print("  2. L2 normalization is missing in chat but was in training")
    print("  3. Hidden states from different layer/position")
    
    # Check if pack_256 function normalizes
    print("\n" + "="*80)
    print("CHECKING EVAL pack_256 FUNCTION")
    print("="*80)
    
    eval_code = '''
def pack_256(vec: torch.Tensor):
    """L2-normalize then take first 256 dims."""
    if vec is None:
        return None
    v = torch.nn.functional.normalize(vec.float(), dim=-1)  # <-- L2 NORMALIZE!
    v = v[:256] if v.shape[-1] >= 256 else torch.nn.functional.pad(v, (0, 256 - v.shape[-1]))
    return [float(x) for x in v.cpu()]
'''
    print(eval_code)
    print("\n⚠️  EVAL NORMALIZES HIDDEN STATES WITH L2 NORM!")
    print("Chat probably doesn't do this!")