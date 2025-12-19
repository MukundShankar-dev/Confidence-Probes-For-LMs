import json

with open("results/transfer/qwen2.5_7b_super_gen/qwen2.5_1.5b/metrics.json") as f:
    results = json.load(f)

total_n = sum(r["n"] for r in results)

metrics = [
    "model_accuracy",
    "probe_accuracy",
    "probe_precision",
    "probe_recall",
    "probe_f1",
    "auc_roc",
    "auc_pr",
    "brier",
]

overall = {}
for m in metrics:
    overall[m] = sum(r[m] * r["n"] for r in results) / total_n

print(f"Total N = {total_n:,}")
for k, v in overall.items():
    print(f"{k:>16}: {v:.4f}")
