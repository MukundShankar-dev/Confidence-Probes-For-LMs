import json
from collections import Counter

# Check train split for llama
with open('data/probe_splits_80_10_10/backend_llama31/train.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f]
    
em_counts = Counter([row['em'] for row in train_data])
total = len(train_data)

print(f"Total examples: {total}")
print(f"Class 0 (wrong): {em_counts[0]} ({em_counts[0]/total*100:.1f}%)")
print(f"Class 1 (correct): {em_counts[1]} ({em_counts[1]/total*100:.1f}%)")
print(f"Imbalance ratio: {em_counts[0]/em_counts[1]:.2f}:1")