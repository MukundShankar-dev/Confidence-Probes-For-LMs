#!/usr/bin/env python3
# scripts/merge_results.py
import argparse, glob, json, os, re, sys
from src.eval.metrics import evaluate_batch

def natural_shard_key(path: str):
    """Sort by numeric shard id if present, else by name."""
    m = re.search(r"shard(\d+)", os.path.basename(path))
    return (0, int(m.group(1))) if m else (1, path)

def main():
    ap = argparse.ArgumentParser(description="Merge TriviaQA shard files and evaluate.")
    ap.add_argument("model_dir", help="Directory like results/triviaqa/{model}")
    ap.add_argument("--out", help="Override output path (.jsonl). Default: {model_dir}/{model}_final_only_all.jsonl")
    ap.add_argument("--pattern", help="Custom glob inside model_dir (overrides inference), e.g. '*_final_only_shard*.json'")
    ap.add_argument("--keep-local-idx", action="store_true",
                    help="Do NOT overwrite 'idx'; keep per-file idx values.")
    ap.add_argument("--no-eval", action="store_true", help="Skip evaluation after merge.")
    args = ap.parse_args()

    model_dir = args.model_dir.rstrip("/")

    if not os.path.isdir(model_dir):
        print(f"error: '{model_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    model = os.path.basename(model_dir)

    # Discover shard files
    if args.pattern:
        shard_glob = os.path.join(model_dir, args.pattern)
        files = sorted(glob.glob(shard_glob), key=natural_shard_key)
    else:
        primary = os.path.join(model_dir, f"{model}_final_only_shard*.json")
        files = sorted(glob.glob(primary), key=natural_shard_key)
        shard_glob = primary
        if not files:
            # Fallback to any prefix (helps if files are named like '8b_final_only_shard*.json')
            fallback = os.path.join(model_dir, "*_final_only_shard*.json")
            files = sorted(glob.glob(fallback), key=natural_shard_key)
            shard_glob = fallback

    if not files:
        print(f"error: no files matched pattern: {shard_glob}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or os.path.join(model_dir, f"{model}_final_only_all.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    global_idx = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for fp in files:
            m = re.search(r"shard(\d+)", os.path.basename(fp))
            shard_id = int(m.group(1)) if m else None
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("overall"):
                        continue
                    row["shard_id"] = shard_id
                    row["local_idx"] = row.get("idx")
                    row["global_idx"] = global_idx
                    if not args.keep_local_idx:
                        row["idx"] = global_idx
                    w.write(json.dumps(row, ensure_ascii=False) + "\n")
                    global_idx += 1

    print(f"[merge] wrote {out_path} with {global_idx} rows")

    if args.no_eval:
        return

    # Evaluate (if fields present)
    preds, refs = [], []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("overall"):
                continue
            if "prediction" in row and "references" in row:
                preds.append(row["prediction"])
                refs.append(row["references"])

    if preds and refs and len(preds) == len(refs):
        print(evaluate_batch(preds, refs))
    else:
        print("[eval] skipped: missing 'prediction'/'references' fields in merged rows or empty set.")

if __name__ == "__main__":
    main()
