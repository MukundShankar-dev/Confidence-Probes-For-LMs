import glob, json, os, re
from src.eval.metrics import evaluate_batch

files = sorted(glob.glob("results/triviaqa/harmony_medium_shard*.json"))
out_path = "results/triviaqa/harmony_medium_all.jsonl"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

global_idx = 0
with open(out_path, "w", encoding="utf-8") as w:
    for fp in files:
        # try to pick up shard id from filename for provenance
        m = re.search(r"shard(\d+)", fp); shard_id = int(m.group(1)) if m else None
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("overall"):    # skip summary lines
                    continue
                row["shard_id"] = shard_id
                row["local_idx"] = row.get("idx")
                row["global_idx"] = global_idx
                row["idx"] = global_idx     # optional: overwrite idx
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                global_idx += 1

print(f"[merge] wrote {out_path} with {global_idx} rows")

in_path = "results/triviaqa/harmony_medium_all.jsonl"
preds, refs = [], []
with open(in_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row.get("overall"): continue
        preds.append(row["prediction"])
        refs.append(row["references"])
print(evaluate_batch(preds, refs))