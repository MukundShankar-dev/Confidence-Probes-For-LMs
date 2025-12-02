#!/usr/bin/env python3
"""
Recalibrate probe using Platt scaling (logistic regression on validation set)

This learns a transformation: calibrated_prob = sigmoid(a * raw_prob + b)
to map raw probabilities to well-calibrated probabilities.

Usage:
    python calibrate_probe.py \
        --probe_dir backend_llama31/probe_mlp \
        --val_data data/probe_splits_80_10_10/llama_val_split.jsonl \
        --output backend_llama31/probe_mlp_calibrated
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt

def load_validation_predictions(probe_dir, val_data_file):
    """Load probe and get predictions on validation data"""
    
    print(f"Loading probe from {probe_dir}...")
    probe = joblib.load(Path(probe_dir) / "probe_model.joblib")
    
    print(f"Loading validation data from {val_data_file}...")
    with open(val_data_file) as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} validation examples")
    
    # Extract features
    SCALAR_KEYS = [
        "model_confidence", "lp_mean", "seq_conf",
        "entropy_mean", "entropy_std",
        "margin_mean", "margin_min",
        "rescore_logp", "answer_len",
        "parsed_json_ok", "parsed_p_true_ok", "is_unknown",
    ]
    VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]
    
    raw_probs = []
    y_true = []
    
    for ex in data:
        # Build feature dict
        feat_dict = {}
        for k in SCALAR_KEYS:
            feat_dict[k] = ex.get(k, np.nan)
        
        # Check if hidden states exist
        has_hidden = all(ex.get(k) is not None for k in VECTOR_KEYS)
        
        if has_hidden:
            for name in VECTOR_KEYS:
                vec = ex.get(name, [])
                if vec:
                    for j, v in enumerate(vec):
                        feat_dict[f"{name}_{j}"] = float(v)
        
        # Create DataFrame (alphabetical order)
        df = pd.DataFrame([feat_dict])
        
        # Predict
        prob = probe.predict_proba(df.values)[:, 1][0]
        raw_probs.append(prob)
        y_true.append(ex.get('em', 0))
    
    return np.array(raw_probs), np.array(y_true)

def fit_platt_scaling(raw_probs, y_true):
    """Fit Platt scaling (logistic regression)"""
    
    print("\n" + "="*80)
    print("FITTING PLATT SCALING")
    print("="*80)
    
    # Reshape for sklearn
    X = raw_probs.reshape(-1, 1)
    
    # Fit logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y_true)
    
    # Get calibrated probabilities
    cal_probs = lr.predict_proba(X)[:, 1]
    
    # Compute metrics
    raw_brier = brier_score_loss(y_true, raw_probs)
    cal_brier = brier_score_loss(y_true, cal_probs)
    
    raw_logloss = log_loss(y_true, raw_probs)
    cal_logloss = log_loss(y_true, cal_probs)
    
    print(f"\nBrier Score:")
    print(f"  Raw:        {raw_brier:.4f}")
    print(f"  Calibrated: {cal_brier:.4f}")
    print(f"  Improvement: {(raw_brier - cal_brier)/raw_brier*100:.1f}%")
    
    print(f"\nLog Loss:")
    print(f"  Raw:        {raw_logloss:.4f}")
    print(f"  Calibrated: {cal_logloss:.4f}")
    print(f"  Improvement: {(raw_logloss - cal_logloss)/raw_logloss*100:.1f}%")
    
    print(f"\nPlatt Scaling Parameters:")
    print(f"  Coefficient (a): {lr.coef_[0][0]:.4f}")
    print(f"  Intercept (b): {lr.intercept_[0]:.4f}")
    print(f"  Formula: sigmoid({lr.coef_[0][0]:.4f} * raw_prob + {lr.intercept_[0]:.4f})")
    
    # Example transformations
    print(f"\nExample Transformations:")
    test_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for p in test_probs:
        cal_p = lr.predict_proba([[p]])[0, 1]
        print(f"  {p:.1f} → {cal_p:.3f}")
    
    return lr

def fit_isotonic_regression(raw_probs, y_true):
    """Fit isotonic regression (non-parametric)"""
    
    print("\n" + "="*80)
    print("FITTING ISOTONIC REGRESSION")
    print("="*80)
    
    # Fit isotonic regression
    iso = IsotonicRegression(out_of_bounds='clip')
    cal_probs = iso.fit_transform(raw_probs, y_true)
    
    # Compute metrics
    raw_brier = brier_score_loss(y_true, raw_probs)
    cal_brier = brier_score_loss(y_true, cal_probs)
    
    print(f"\nBrier Score:")
    print(f"  Raw:        {raw_brier:.4f}")
    print(f"  Calibrated: {cal_brier:.4f}")
    print(f"  Improvement: {(raw_brier - cal_brier)/raw_brier*100:.1f}%")
    
    # Example transformations
    print(f"\nExample Transformations:")
    test_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for p in test_probs:
        cal_p = iso.predict([p])[0]
        print(f"  {p:.1f} → {cal_p:.3f}")
    
    return iso

def plot_calibration(raw_probs, y_true, platt_cal, iso_cal, output_file):
    """Plot calibration curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibration curve
    ax = axes[0]
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Raw probabilities
    bin_counts = np.histogram(raw_probs, bins=bins)[0]
    bin_sums = np.histogram(raw_probs, bins=bins, weights=y_true)[0]
    bin_means = np.zeros_like(bin_sums, dtype=float)
    np.divide(bin_sums, bin_counts, where=bin_counts > 0, out=bin_means)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(bin_centers, bin_means, 'o-', label='Raw probabilities')
    
    # Platt-calibrated
    platt_probs = platt_cal.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    bin_counts_p = np.histogram(platt_probs, bins=bins)[0]
    bin_sums_p = np.histogram(platt_probs, bins=bins, weights=y_true)[0]
    bin_means_p = np.zeros_like(bin_sums_p, dtype=float)
    np.divide(bin_sums_p, bin_counts_p, where=bin_counts_p > 0, out=bin_means_p)
    ax.plot(bin_centers, bin_means_p, 's-', label='Platt scaling')
    
    # Isotonic-calibrated
    iso_probs = iso_cal.predict(raw_probs)
    bin_counts_i = np.histogram(iso_probs, bins=bins)[0]
    bin_sums_i = np.histogram(iso_probs, bins=bins, weights=y_true)[0]
    bin_means_i = np.zeros_like(bin_sums_i, dtype=float)
    np.divide(bin_sums_i, bin_counts_i, where=bin_counts_i > 0, out=bin_means_i)
    ax.plot(bin_centers, bin_means_i, '^-', label='Isotonic regression')
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Empirical Accuracy')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram
    ax = axes[1]
    ax.hist(raw_probs, bins=20, alpha=0.5, label='Raw')
    ax.hist(platt_probs, bins=20, alpha=0.5, label='Platt')
    ax.hist(iso_probs, bins=20, alpha=0.5, label='Isotonic')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Count')
    ax.set_title('Probability Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved calibration plot to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe_dir', required=True, help='Directory with probe_model.joblib')
    parser.add_argument('--val_data', required=True, help='Validation JSONL file')
    parser.add_argument('--output', required=True, help='Output directory for calibrated probe')
    parser.add_argument('--method', choices=['platt', 'isotonic', 'both'], default='both')
    args = parser.parse_args()
    
    # Load predictions
    raw_probs, y_true = load_validation_predictions(args.probe_dir, args.val_data)
    
    print(f"\nValidation set statistics:")
    print(f"  Size: {len(y_true)}")
    print(f"  Positive rate: {y_true.mean():.3f}")
    print(f"  Raw prob mean: {raw_probs.mean():.3f}")
    print(f"  Raw prob std: {raw_probs.std():.3f}")
    
    # Fit calibration
    platt_cal = None
    iso_cal = None
    
    if args.method in ['platt', 'both']:
        platt_cal = fit_platt_scaling(raw_probs, y_true)
    
    if args.method in ['isotonic', 'both']:
        iso_cal = fit_isotonic_regression(raw_probs, y_true)
    
    # Save calibrated models
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy original probe
    import shutil
    original_probe_file = Path(args.probe_dir) / "probe_model.joblib"
    shutil.copy(original_probe_file, output_dir / "probe_model.joblib")
    
    # Save calibration models
    if platt_cal:
        joblib.dump(platt_cal, output_dir / "platt_calibration.joblib")
        print(f"\n✓ Saved Platt calibration to {output_dir / 'platt_calibration.joblib'}")
    
    if iso_cal:
        joblib.dump(iso_cal, output_dir / "isotonic_calibration.joblib")
        print(f"✓ Saved Isotonic calibration to {output_dir / 'isotonic_calibration.joblib'}")
    
    # Plot
    if platt_cal and iso_cal:
        plot_calibration(raw_probs, y_true, platt_cal, iso_cal, 
                        output_dir / "calibration_curves.png")
    
    print("\n" + "="*80)
    print("CALIBRATION COMPLETE")
    print("="*80)
    print(f"\nTo use calibrated probe in chat_inference:")
    print(f"  1. Add --calibration platt or --calibration isotonic flag")
    print(f"  2. Load calibration model after probe")
    print(f"  3. Apply: calibrated_prob = calibration_model.predict_proba([[raw_prob]])[0, 1]")

if __name__ == "__main__":
    main()
    