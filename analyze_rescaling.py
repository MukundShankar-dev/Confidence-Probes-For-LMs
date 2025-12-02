#!/usr/bin/env python3
"""
Quick fix: Rescale probabilities using observed statistics

If probe outputs are systematically too low, we can apply:
  rescaled_prob = (raw_prob - mean) / std * target_std + target_mean

Or simpler: shift to center around 0.5
  rescaled_prob = clip((raw_prob - threshold) * scale + 0.5, 0, 1)
"""

import numpy as np

def analyze_probe_distribution():
    """
    From your eval results:
    - Threshold: 0.327
    - Precision: 0.592 (when prob > 0.327, 59% are correct)
    - Recall: 0.806 (catches 80% of correct answers)
    - F1: 0.683
    
    This suggests:
    - Probabilities are too uniformly distributed
    - Need to stretch the range to make discrimination clearer
    """
    
    print("="*80)
    print("PROBE PROBABILITY RESCALING")
    print("="*80)
    
    # Current behavior
    threshold = 0.327
    precision = 0.592
    recall = 0.806
    
    print(f"\nCurrent probe behavior:")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Precision: {precision:.3f} (when predicting 'correct')")
    print(f"  Recall: {recall:.3f} (catches 80% of correct answers)")
    
    # What we want
    print(f"\nDesired behavior:")
    print(f"  When model is clearly correct (Paris): prob > 0.7")
    print(f"  When model is uncertain: prob ≈ 0.5")
    print(f"  When model is clearly wrong: prob < 0.3")
    
    # Rescaling options
    print(f"\n" + "="*80)
    print("RESCALING OPTIONS")
    print("="*80)
    
    # Option 1: Linear stretch around threshold
    print(f"\nOption 1: Stretch around threshold")
    print(f"  Formula: (prob - {threshold}) * 2.0 + 0.5")
    print(f"  This maps:")
    print(f"    {threshold} → 0.5 (at threshold)")
    print(f"    0.0 → {(0 - threshold) * 2.0 + 0.5:.2f}")
    print(f"    1.0 → {(1 - threshold) * 2.0 + 0.5:.2f}")
    print(f"  Clip to [0, 1]")
    
    # Test on your example
    your_prob = 0.169
    rescaled = np.clip((your_prob - threshold) * 2.0 + 0.5, 0, 1)
    print(f"\n  Your example (Paris): {your_prob:.3f} → {rescaled:.3f}")
    
    # Option 2: Sigmoid-like stretch
    print(f"\nOption 2: Sigmoid stretch")
    print(f"  Formula: sigmoid((prob - {threshold}) * 5.0)")
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    rescaled_sig = sigmoid((your_prob - threshold) * 5.0)
    print(f"  Your example (Paris): {your_prob:.3f} → {rescaled_sig:.3f}")
    
    # Option 3: Empirical quantile mapping
    print(f"\nOption 3: Assume probe outputs are Beta-distributed")
    print(f"  Fit Beta(α, β) to observed probabilities")
    print(f"  Then map through inverse CDF")
    print(f"  (Requires validation set statistics)")
    
    # Recommendation
    print(f"\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print(f"""
For quick testing without retraining:

1. Try linear stretch (Option 1):
   ```python
   def rescale_prob(raw_prob, threshold=0.327, stretch=2.0):
       return np.clip((raw_prob - threshold) * stretch + 0.5, 0, 1)
   ```

2. If that doesn't work well, use Platt scaling:
   ```bash
   python calibrate_probe.py \\
       --probe_dir backend_llama31/probe_mlp \\
       --val_data data/probe_splits_80_10_10/llama_val_split.jsonl \\
       --output backend_llama31/probe_mlp_calibrated
   ```

3. For production, retrain with better calibration:
   - Use temperature scaling during training
   - Add focal loss to handle class imbalance
   - Ensure validation set is representative
""")

if __name__ == "__main__":
    analyze_probe_distribution()