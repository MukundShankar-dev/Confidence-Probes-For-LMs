import json
import argparse as ap

def main():
    args = ap.ArgumentParser()
    args.add_argument("--input_file", type=str, default="results/transfer/qwen2.5_7b_generalizable/qwen2.5_1.5b/metrics.json",
                     help="Path to the input JSON file containing metrics.")
    args = args.parse_args()

    print(f"Loading results from {args.input_file}...")
    with open(args.input_file) as f:
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

if __name__ == "__main__":
    main()
