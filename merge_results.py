#!/usr/bin/env python3
# scripts/merge_results.py
import argparse, glob, json, os, re, sys
from src.eval.metrics import evaluate_batch

def natural_shard_key(path: str):
    """Sort by numeric shard id if present, else by name."""
    m = re.search(r"shard(\d+)", os.path.basename(path))
    return (0, int(m.group(1))) if m else (1, path)

def _to_float01(val):
    """Parse floats or percent-like strings to [0,1], else return None."""
    if val is None:
        return None
    try:
        s = str(val).strip().replace("％", "%")
        if s.endswith("%"):
            v = float(s[:-1]) / 100.0
        else:
            v = float(s)
        if v > 1.0:  # tolerate 0..100
            v = v / 100.0
        return max(0.0, min(1.0, v))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Merge shard files and evaluate (EM/F1 + Brier).")
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
            # Fallback to any prefix (e.g., '8b_final_only_shard*.json')
            fallback = os.path.join(model_dir, "*_final_only_shard*.json")
            files = sorted(glob.glob(fallback), key=natural_shard_key)
            shard_glob = fallback

    if not files:
        print(f"error: no files matched pattern: {shard_glob}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or os.path.join(model_dir, f"{model}_final_only_all.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ---------- Merge ----------
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
                        continue  # skip per-shard summaries
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

    # ---------- Evaluate from merged ----------
    preds, refs = [], []
    brier_sum = 0.0
    brier_n = 0

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("overall"):
                continue
            # EM/F1 batch eval (prediction + references)
            if "prediction" in row and "references" in row:
                preds.append(row["prediction"])
                refs.append(row["references"])
            # Brier (needs model_confidence and ground-truth correctness)
            # Prefer explicit per-row EM if present; else infer from refs/pred?
            # (We rely on 'em' which run_baseline writes.)
            if "model_confidence" in row and "em" in row:
                p = _to_float01(row["model_confidence"])
                y = int(row["em"]) if isinstance(row["em"], (int, bool)) else None
                if p is not None and y in (0, 1):
                    brier_sum += (p - y) ** 2
                    brier_n += 1

    # Print metrics
    if preds and refs and len(preds) == len(refs):
        metrics = evaluate_batch(preds, refs)
    else:
        metrics = {}
        print("[eval] EM/F1 skipped: missing 'prediction'/'references' in merged rows or empty set.")

    if brier_n > 0:
        metrics["brier"] = brier_sum / brier_n
        metrics["brier_n"] = brier_n
    else:
        print("[eval] Brier skipped: no usable 'model_confidence'+'em' pairs in merged rows.")

    if metrics:
        print(metrics)

if __name__ == "__main__":
    main()
