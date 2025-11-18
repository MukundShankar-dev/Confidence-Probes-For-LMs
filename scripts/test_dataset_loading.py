#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test loading all datasets with train split

Usage:
    python -m scripts.test_dataset_loading
"""

from src.data.datasets import load_qa

# Datasets to test
DATASETS = [
    "triviaqa",
    "squad_v2", 
    "hotpotqa",
    "mmlu",
    "gsm8k",
]

def test_dataset(name, split, limit=10):
    """Test loading a dataset and show sample"""
    print(f"\n{'='*80}")
    print(f"Testing: {name} (split={split})")
    print('='*80)
    
    try:
        ds = load_qa(name, split, limit=limit)
        print(f"✅ Successfully loaded {len(ds)} examples")
        
        # Show first example
        if len(ds) > 0:
            ex = ds[0]
            print(f"\nFirst example:")
            print(f"  Question: {ex['question'][:100]}...")
            print(f"  Answers: {ex['answers']}")
            print(f"  Context: {ex['context'][:50] if ex['context'] else '(empty)'}...")
            
            # Check for <unk> or empty answers
            if any(ans in ['<unk>', 'Unknown', ''] for ans in ex['answers']):
                print(f"  ⚠️  Warning: Found placeholder answer: {ex['answers']}")
        
        # Try loading more to ensure it's not just the first few
        if limit < 100:
            print(f"\nTrying to load 100 examples...")
            ds_full = load_qa(name, split, limit=100)
            print(f"✅ Successfully loaded {len(ds_full)} examples")
            
            # Check a few random examples for valid answers
            import random
            sample_indices = random.sample(range(len(ds_full)), min(3, len(ds_full)))
            print(f"\nSample answers from random examples:")
            for idx in sample_indices:
                ex = ds_full[idx]
                print(f"  Example {idx}: {ex['answers'][:2]}...")
                
        return True
        
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("DATASET LOADING TEST")
    print("="*80)
    print("Testing all datasets with TRAIN split")
    print("This simulates what happens in collect_internals.py")
    
    results = {}
    
    # Test train split for all datasets
    for dataset_name in DATASETS:
        success = test_dataset(dataset_name, split="train", limit=10)
        results[dataset_name] = success
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_good = True
    for dataset_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{dataset_name:15s}: {status}")
        if not success:
            all_good = False
    
    print("\n" + "="*80)
    if all_good:
        print("✅ All datasets loaded successfully!")
        print("You can proceed with probe training using train splits.")
    else:
        print("❌ Some datasets failed to load.")
        print("Fix the issues above before proceeding.")
    print("="*80)
    
    # Now test validation/test splits for eval
    print("\n\n" + "="*80)
    print("BONUS: Testing evaluation splits")
    print("="*80)
    
    eval_splits = {
        "triviaqa": "validation",
        "squad_v2": "validation",
        "hotpotqa": "validation",
        "mmlu": "validation",
        "gsm8k": "test",
    }
    
    eval_results = {}
    for dataset_name, eval_split in eval_splits.items():
        print(f"\nTesting {dataset_name} with {eval_split} split...")
        success = test_dataset(dataset_name, split=eval_split, limit=10)
        eval_results[dataset_name] = success
    
    print("\n" + "="*80)
    print("EVALUATION SPLITS SUMMARY")
    print("="*80)
    
    all_eval_good = True
    for dataset_name, success in eval_results.items():
        eval_split = eval_splits[dataset_name]
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{dataset_name:15s} ({eval_split:10s}): {status}")
        if not success:
            all_eval_good = False
    
    print("\n" + "="*80)
    if all_eval_good:
        print("✅ All evaluation splits loaded successfully!")
        print("You can proceed with probe evaluation.")
    else:
        print("❌ Some evaluation splits failed to load.")
        print("You may need to adjust your evaluation strategy.")
    print("="*80)

if __name__ == "__main__":
    main()