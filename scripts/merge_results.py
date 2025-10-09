#!/usr/bin/env python3
import argparse, glob, json, os, re, sys
from src.eval.metrics import evaluate_batch

def natural_key(s):
    # sort shard files numerically if possible
    m = re.search(r"shard(\d+)", s)
    return int(m.group(1)) if m else s

def main():
    ap = argparse.ArgumentParser(description="Merge TriviaQA shard files and evaluate.")
    ap.add_argument("model_dir", help="Directory like results/triviaqa/{model}")
    ap.add_argument("--overwrite-idx", action="store_true",
                    help="If set, overwrite idx with global_idx (default: same behavior as original script).")
    args = ap.parse_args()

    model_dir = args.model_dir.rstrip("/")

    if not os.path.isdir(model_dir):
        print(f"error: '{model_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Infer model name from the directory basename
    model = os.path.basename(model_dir)
    # Expect files like results/triviaqa/{model}/{model}_final_only_shard*.json
    shard_glob = os.path.join(model_dir, f"{model}_final_only_shard*.json")
    files = sorted(glob.glob(shard_glob), key=natural_key)

    if not files:
        print(f"error: no files matched pattern: {shard_glob}", file=sys.stderr)
        sys.exit(1)

    out_path = os.path.join(model_dir, f"{model}_final_only_all.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    global_idx = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for fp in files:
            # try to pick up shard id from filename for provenance
            m = re.search(r"shard(\d+)", os.path.basename(fp))
            shard_id = int(m.group(1)) if m else None
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("overall"):    # skip summary lines
                        continue
                    row["shard_id"] = shard_id
                    row["local_idx"] = row.get("idx")
                    row["global_idx"] = global_idx
                    if args.overwrite-idx or True:  # keep original behavior: overwrite idx
                        row["idx"] = global_idx
                    w.write(json.dumps(row, ensure_ascii=False) + "\n")
                    global_idx += 1

    print(f"[merge] wrote {out_path} with {global_idx} rows")

    # Evaluate
    preds, refs = [], []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("overall"):
                continue
            preds.append(row["prediction"])
            refs.append(row["references"])

    print(evaluate_batch(preds, refs))

if __name__ == "__main__":
    main()
