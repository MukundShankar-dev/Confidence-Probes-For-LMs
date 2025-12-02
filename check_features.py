#!/usr/bin/env python3
"""
Comprehensive check of features across training, eval, and chat inference.
This will identify ANY mismatches.
"""

import json
import joblib
import pandas as pd
import numpy as np

print("="*80)
print("COMPREHENSIVE FEATURE CHECKING")
print("="*80)

# 1. Check training data
print("\n" + "="*80)
print("1. TRAINING DATA FEATURES")
print("="*80)

try:
    with open('data/probe_splits_80_10_10/llama_train_split.jsonl', 'r') as f:
        first_line = f.readline()
        train_example = json.loads(first_line)
    
    print("\n✓ Training data loaded")
    print(f"Total keys: {len(train_example)}")
    
    scalar_keys = [
        "model_confidence", "lp_mean", "seq_conf",
        "entropy_mean", "entropy_std",
        "margin_mean", "margin_min",
        "rescore_logp", "answer_len",
        "parsed_json_ok", "parsed_p_true_ok", "is_unknown",
    ]
    
    vector_keys = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]
    
    print("\nScalar features:")
    for key in scalar_keys:
        if key in train_example:
            val = train_example[key]
            print(f"  ✓ {key}: {val}")
        else:
            print(f"  ✗ {key}: MISSING")
    
    print("\nVector features:")
    for key in vector_keys:
        if key in train_example:
            val = train_example[key]
            if val is None:
                print(f"  ✗ {key}: None")
            elif isinstance(val, list):
                print(f"  ✓ {key}: List[{len(val)}]")
            else:
                print(f"  ? {key}: {type(val)}")
        else:
            print(f"  ✗ {key}: MISSING")
    
except FileNotFoundError:
    print("\n✗ Training data not found at data/probe_splits_80_10_10/llama_train_split.jsonl")
    train_example = None

# 2. Check probe expectations
print("\n" + "="*80)
print("2. PROBE MODEL EXPECTATIONS")
print("="*80)

try:
    probe = joblib.load('backend_llama31/probe_mlp/probe_model.joblib')
    print("\n✓ Probe loaded")
    
    if hasattr(probe, 'feature_names_in_'):
        print(f"\nProbe expects {len(probe.feature_names_in_)} features")
        print("\nFirst 20 features:")
        for i, name in enumerate(probe.feature_names_in_[:20]):
            print(f"  {i:3d}. {name}")
        
        print("\nLast 10 features:")
        for i in range(len(probe.feature_names_in_) - 10, len(probe.feature_names_in_)):
            name = probe.feature_names_in_[i]
            print(f"  {i:3d}. {name}")
        
        # Check for hidden states
        hidden_features = [f for f in probe.feature_names_in_ if any(x in f for x in ['h_last_256', 'h_pool_256', 'h_last_mid_256', 'h_pool_mid_256'])]
        print(f"\nHidden state features: {len(hidden_features)} (should be 1024 = 4×256)")
        
        if len(hidden_features) == 0:
            print("  ❌ NO HIDDEN STATES IN PROBE!")
        elif len(hidden_features) != 1024:
            print(f"  ⚠ Expected 1024, got {len(hidden_features)}")
    else:
        print("\n✗ Probe has no feature_names_in_")
        
except FileNotFoundError:
    print("\n✗ Probe not found at backend_llama31/probe_mlp/probe_model.joblib")

# 3. Check what chat_inference sends
print("\n" + "="*80)
print("3. CHAT INFERENCE FEATURE CONSTRUCTION")
print("="*80)

# Simulate what chat sends
chat_features = {
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

# Add hidden states (currently zeros)
for key in ['h_last_256', 'h_pool_256', 'h_last_mid_256', 'h_pool_mid_256']:
    for i in range(256):
        chat_features[f'{key}_{i}'] = 0.0

df = pd.DataFrame([chat_features])

print(f"\nDataFrame shape: {df.shape}")
print(f"Expected: (1, 1036) = 12 scalars + 4×256 hidden")

print("\nFirst 20 columns (pandas sorts alphabetically!):")
for i, col in enumerate(df.columns[:20]):
    print(f"  {i:3d}. {col}")

# 4. Compare everything
print("\n" + "="*80)
print("4. COMPARISON & DIAGNOSIS")
print("="*80)

if train_example and hasattr(probe, 'feature_names_in_'):
    # Check if probe feature names match training data keys
    probe_scalars = [f for f in probe.feature_names_in_ if not any(x in f for x in ['_0', '_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9'])]
    
    print("\nScalar features comparison:")
    print(f"  Probe expects: {probe_scalars[:12]}")
    print(f"  Chat sends: {list(df.columns[:12])}")
    
    if probe_scalars[:12] == list(df.columns[:12]):
        print("  ✓ MATCH!")
    else:
        print("  ✗ MISMATCH!")
        
        for i, (p, c) in enumerate(zip(probe_scalars[:12], df.columns[:12])):
            if p != c:
                print(f"    Position {i}: probe wants '{p}', chat sends '{c}'")

# 5. Test prediction with actual probe
print("\n" + "="*80)
print("5. ACTUAL PREDICTION TEST")
print("="*80)

try:
    prob = probe.predict_proba(df.values)[:, 1][0]
    print(f"\n✓ Prediction successful: {prob:.10f}")
    
    if prob < 0.001:
        print("\n❌ PROBLEM: Probe outputs near-zero!")
        print("\nPossible causes:")
        print("  1. Hidden states are all zeros → probe sees missing data")
        print("  2. Feature order mismatch")
        print("  3. Scaler issues")
        
        # Check if zeros are the problem
        print("\n" + "="*80)
        print("6. TESTING WITH NON-ZERO HIDDEN STATES")
        print("="*80)
        
        chat_features_random = chat_features.copy()
        for key in ['h_last_256', 'h_pool_256', 'h_last_mid_256', 'h_pool_mid_256']:
            for i in range(256):
                chat_features_random[f'{key}_{i}'] = np.random.randn() * 0.1
        
        df_random = pd.DataFrame([chat_features_random])
        prob_random = probe.predict_proba(df_random.values)[:, 1][0]
        print(f"\nPrediction with random hidden states: {prob_random:.6f}")
        
        if prob_random > 0.01:
            print("\n✓✓✓ DIAGNOSIS: Hidden states being zeros is the problem!")
            print("\nSOLUTION: Fix chat_inference.py to actually populate hidden states")
            print("Currently they're probably returning None or empty lists")
        else:
            print("\n❌ Still near-zero even with random hidden states")
            print("The problem is something else (feature order? scaler?)")
    else:
        print("\n✓ Probe working correctly!")
        
except Exception as e:
    print(f"\n✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
Run this script to diagnose the issue:

    python check_features.py

If hidden states are the problem, check in chat_inference.py:
    1. Is use_hidden=True being passed correctly?
    2. Are hidden states being extracted from model?
    3. Are they being converted to lists properly?
    4. Are they being added to the features dict?

Debug by adding print statements in extract_features_chat():
    print(f"Hidden states extracted: {h_last_256 is not None}")
    print(f"h_last_256 sample: {h_last_256[:5] if h_last_256 else 'None'}")
""")