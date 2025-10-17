# scripts/make_probe_split.py
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os
import random
import re
from typing import Dict, List, Tuple, Iterable

"""
Script to pool probe JSONL files from multiple datasets/backends,
and create stratified train/val/test splits both globally and per-backend.

Example usage:

python -m scripts.make_probe_split \
  --root data/probe \
  --out_dir data/probe_splits_80_10_10 \
  --train_frac 0.80 \
  --val_frac 0.10 \
  --test_frac 0.10 \
  --use_supplementary append

"""


def infer_dataset_backend(path: str) -> Tuple[str, str]:
    """
    Expect paths like: data/probe/<dataset>/<backend>/.../*.jsonl
    Return (dataset, backend). If we can't infer, return '_unknown'.
    """
    p = path.replace("\\", "/")
    m = re.search(r"/probe/([^/]+)/([^/]+)/", p)
    if m:
        return m.group(1), m.group(2)
    m2 = re.search(r"/probe/([^/]+)/", p)
    ds = m2.group(1) if m2 else "_unknown"
    m3 = re.search(r"/probe/[^/]+/([^/]+)/", p)
    be = m3.group(1) if m3 else "_unknown"
    return ds, be


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj


def pool_rows(files: List[str], tag_from_path: bool) -> List[dict]:
    rows = []
    for fp in files:
        ds_infer, be_infer = infer_dataset_backend(fp)
        for obj in iter_jsonl(fp):
            if obj.get("overall") is True:
                continue  # skip summary rows
            if tag_from_path:
                obj.setdefault("dataset", ds_infer)
                obj.setdefault("backend", be_infer)
            rows.append(obj)
    return rows


def stratified_split(
    rows: List[dict],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify_keys: Tuple[str, ...] = ("dataset",),
) -> Dict[str, List[int]]:
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "fractions must sum to 1.0"
    rnd = random.Random(seed)

    buckets: Dict[Tuple, List[int]] = {}
    for i, r in enumerate(rows):
        key = tuple(r.get(k, "_default") for k in stratify_keys)
        buckets.setdefault(key, []).append(i)

    splits = {"train": [], "val": [], "test": []}
    for key, idxs in buckets.items():
        rnd.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val
        splits["train"].extend(idxs[:n_train])
        splits["val"].extend(idxs[n_train:n_train + n_val])
        splits["test"].extend(idxs[n_train + n_val:])

    rnd.shuffle(splits["train"])
    rnd.shuffle(splits["val"])
    rnd.shuffle(splits["test"])
    return splits


def write_jsonl(path: str, rows: List[dict], indices: List[int]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        for i in indices:
            w.write(json.dumps(rows[i], ensure_ascii=False) + "\n")


def make_one_split(out_dir: str, rows: List[dict], train_frac: float, val_frac: float,
                   test_frac: float, seed: int, stratify_by_dataset: bool = True):
    os.makedirs(out_dir, exist_ok=True)

    # Assign stable pooled_id per output pool
    for i, r in enumerate(rows):
        r.setdefault("_pooled_id", i)

    pooled_path = os.path.join(out_dir, "pooled.jsonl")
    with open(pooled_path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    splits = stratified_split(
        rows,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        stratify_keys=("dataset",) if stratify_by_dataset else ("_default",),
    )

    split_map = {
        "meta": {
            "seed": seed,
            "counts": {k: len(v) for k, v in splits.items()},
            "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        },
        "splits": splits,  # indices into pooled.jsonl
    }
    with open(os.path.join(out_dir, "split_map.json"), "w", encoding="utf-8") as w:
        json.dump(split_map, w, ensure_ascii=False, indent=2)

    write_jsonl(os.path.join(out_dir, "train.jsonl"), rows, splits["train"])
    write_jsonl(os.path.join(out_dir, "val.jsonl"),   rows, splits["val"])
    write_jsonl(os.path.join(out_dir, "test.jsonl"),  rows, splits["test"])

    print(f"[ok] {out_dir}: train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")


def main():
    ap = argparse.ArgumentParser(description="Pool probe JSONLs and create global + per-backend splits.")
    ap.add_argument("--root", default="data/probe", help="Root directory (expects data/probe/<dataset>/<backend>/*.jsonl)")
    ap.add_argument("--glob", default="**/*.jsonl", help="Glob under root to pick up JSONLs")
    ap.add_argument("--out_dir", default="data/probe_splits", help="Where to write outputs")
    ap.add_argument("--train_frac", type=float, default=0.80)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no_tag_from_path", action="store_true",
                    help="Do NOT fill dataset/backend from directory names if missing")
    ap.add_argument("--skip_global", action="store_true", help="Do not write the pooled (all backends) split")

    # NEW: supplementary control
    ap.add_argument("--use_supplementary", choices=["append", "off", "only"], default="append",
                    help="How to handle rows from the supplementary dataset folder")
    ap.add_argument("--supplementary_dataset_name", type=str, default="supplementary",
                    help="Folder name used for supplementary under data/probe/<name>/...")

    # Optional: include/exclude datasets (by folder name) for finer control
    ap.add_argument("--datasets", type=str, default=None,
                    help="Comma-separated allowlist of dataset names to include (e.g., 'triviaqa,hotpot_qa,squad_v2,supplementary'). "
                         "If unset, include all discovered (subject to --use_supplementary).")
    ap.add_argument("--exclude_datasets", type=str, default=None,
                    help="Comma-separated blocklist of dataset names to exclude.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Discover files and pool everything
    files = sorted(glob.glob(os.path.join(args.root, args.glob), recursive=True))
    if not files:
        raise SystemExit(f"No JSONL files found under {args.root} with pattern {args.glob}")

    all_rows = pool_rows(files, tag_from_path=not args.no_tag_from_path)
    if not all_rows:
        raise SystemExit("No rows found after filtering (only 'overall' records?)")

    # 1b) Filter by supplementary policy
    supp_name = args.supplementary_dataset_name
    def _is_supp(r: dict) -> bool:
        return str(r.get("dataset", "")).lower() == supp_name.lower()

    if args.use_supplementary == "off":
        all_rows = [r for r in all_rows if not _is_supp(r)]
    elif args.use_supplementary == "only":
        all_rows = [r for r in all_rows if _is_supp(r)]
    # else "append": keep everything

    if not all_rows:
        raise SystemExit("No rows remain after applying --use_supplementary filter.")

    # 1c) Dataset allow/block lists
    if args.datasets:
        allow = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}
        all_rows = [r for r in all_rows if str(r.get("dataset", "")).lower() in allow]
    if args.exclude_datasets:
        block = {x.strip().lower() for x in args.exclude_datasets.split(",") if x.strip()}
        all_rows = [r for r in all_rows if str(r.get("dataset", "")).lower() not in block]

    if not all_rows:
        raise SystemExit("No rows remain after applying dataset include/exclude filters.")

    # 2) Collect per-backend subsets
    by_backend: Dict[str, List[dict]] = {}
    for r in all_rows:
        be = r.get("backend", "_unknown")
        by_backend.setdefault(be, []).append(r)

    # 3) Global pooled split (optional)
    if not args.skip_global:
        make_one_split(
            out_dir=os.path.join(args.out_dir, "pooled_all_backends"),
            rows=all_rows,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            stratify_by_dataset=True,
        )

    # 4) Per-backend splits (stratified by dataset within each backend)
    for be, rows in sorted(by_backend.items()):
        out_be = os.path.join(args.out_dir, f"backend_{be}")
        make_one_split(
            out_dir=out_be,
            rows=rows,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            stratify_by_dataset=True,
        )


if __name__ == "__main__":
    main()
