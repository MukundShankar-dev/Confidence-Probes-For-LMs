#!/usr/bin/env python3
"""
Inspect probe and diagnose why it outputs 0.0
"""
import joblib
import numpy as np
import pandas as pd
import sys

probe_path = sys.argv[1] if len(sys.argv) > 1 else 'backend_llama31/probe_mlp/probe_model.joblib'

print("="*80)
print(f"INSPECTING PROBE: {probe_path}")
print("="*80)

# Load the probe
probe = joblib.load(probe_path)

print("\nProbe type:", type(probe).__name__)

# Check if it's a pipeline
if hasattr(probe, 'steps'):
    print("\nPipeline steps:")
    for i, (name, step) in enumerate(probe.steps):
        print(f"  {i}. {name}: {type(step).__name__}")
    
    # Check for feature names
    if hasattr(probe, 'feature_names_in_'):
        print(f"\n✓ Feature names stored: {len(probe.feature_names_in_)}")
        print("\nFirst 20 features (alphabetically sorted by pandas):")
        for i, name in enumerate(probe.feature_names_in_[:20]):
            print(f"  {i:3d}. {name}")
        
        print("\nLast 5 features:")
        for i, name in enumerate(probe.feature_names_in_[-5:]):
            idx = len(probe.feature_names_in_) - 5 + i
            print(f"  {idx:3d}. {name}")
    else:
        print("\n❌ No feature names stored!")

# Test with the actual features from chat
print("\n" + "="*80)
print("TEST WITH CHAT FEATURES")
print("="*80)

test_features = {
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

# Add dummy hidden states (will be zeros for now)
for key in ['h_last_256', 'h_pool_256', 'h_last_mid_256', 'h_pool_mid_256']:
    for i in range(256):
        test_features[f'{key}_{i}'] = 0.0

df = pd.DataFrame([test_features])

print(f"\nDataFrame shape: {df.shape}")
print(f"Expected shape: (1, 1036) = 12 scalars + 4×256 hidden")

print("\nFirst 20 column names (after pandas sorts alphabetically):")
for i, col in enumerate(df.columns[:20]):
    print(f"  {i:3d}. {col}")

print("\nScalar features in order:")
scalar_cols = [c for c in df.columns if not any(x in c for x in ['h_last', 'h_pool'])]
for i, col in enumerate(scalar_cols):
    print(f"  {col}: {df[col].values[0]}")

# Predict
print("\n" + "="*80)
print("PREDICTION")
print("="*80)

try:
    prob = probe.predict_proba(df.values)[:, 1][0]
    print(f"\n✓ Predicted probability: {prob:.10f}")
    
    if prob < 0.001:
        print("\n❌ Probe outputs near-zero!")
        
        # Check intermediate steps
        if hasattr(probe, 'steps') and len(probe.steps) >= 2:
            print("\nChecking intermediate steps...")
            
            # After imputer
            imputer = probe.steps[0][1]
            imputed = imputer.transform(df.values)
            print(f"\n1. After Imputer:")
            print(f"   Shape: {imputed.shape}")
            print(f"   First 12 values: {imputed[0, :12]}")
            print(f"   Any NaN: {np.any(np.isnan(imputed))}")
            print(f"   Any Inf: {np.any(np.isinf(imputed))}")
            
            # After scaler
            scaler = probe.steps[1][1]
            scaled = scaler.transform(imputed)
            print(f"\n2. After StandardScaler:")
            print(f"   Shape: {scaled.shape}")
            print(f"   First 12 values: {scaled[0, :12]}")
            print(f"   Min: {scaled.min():.3f}, Max: {scaled.max():.3f}, Mean: {scaled.mean():.3f}, Std: {scaled.std():.3f}")
            print(f"   Any NaN: {np.any(np.isnan(scaled))}")
            print(f"   Any Inf: {np.any(np.isinf(scaled))}")
            
            # Classifier prediction
            classifier = probe.steps[-1][1]
            raw_scores = classifier.decision_function(scaled)
            print(f"\n3. Classifier:")
            print(f"   Decision function: {raw_scores[0]:.6f}")
            print(f"   Probability: {prob:.10f}")
            
            # Check if all hidden states being zero is the issue
            print("\n" + "="*80)
            print("TESTING WITH NON-ZERO HIDDEN STATES")
            print("="*80)
            
            # Create features with random hidden states
            test_features_nonzero = test_features.copy()
            for key in ['h_last_256', 'h_pool_256', 'h_last_mid_256', 'h_pool_mid_256']:
                for i in range(256):
                    test_features_nonzero[f'{key}_{i}'] = np.random.randn()
            
            df2 = pd.DataFrame([test_features_nonzero])
            prob2 = probe.predict_proba(df2.values)[:, 1][0]
            print(f"Probability with random hidden states: {prob2:.6f}")
            
            if prob2 > 0.01:
                print("\n✓ Non-zero hidden states fix it!")
                print("Issue: Hidden states are all zeros in chat inference")
            else:
                print("\n❌ Still near-zero with random hidden states")
                print("Issue: Something else is wrong")
    
    else:
        print("\n✓ Probe working correctly!")

except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if prob < 0.001:
    print("""
Probe outputs near-zero probability. Possible causes:

1. **Hidden states are all zeros** (most likely)
   - Check if extract_features_chat is populating hidden states
   - All zeros → probe sees missing data → predicts 0

2. **Feature order mismatch**
   - DataFrame sorts columns alphabetically
   - Training used different order
   - Check feature_names_in_ vs DataFrame.columns

3. **Extreme scaled values**
   - StandardScaler produces huge/tiny values
   - Classifier saturates to 0

4. **Model weights corrupted**
   - Probe file is broken
   - Re-train the probe

Run with actual hidden states to test!
    """)