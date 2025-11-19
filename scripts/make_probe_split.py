# scripts/make_probe_split.py
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os
import random
import re
from typing import Dict, List, Tuple, Iterable
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

"""
Script to pool probe JSONL files from multiple datasets/backends,
and create stratified train/val/test splits both globally and per-backend.

Example usage:

python -m scripts.make_probe_split \
  --root data/probe \
  --out_dir data/probe_splits_80_10_10 \
  --train_frac 0.80 \
  --val_frac 0.10 \
  --test_frac 0.10

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
    console.print(f"\n[cyan]📂 Reading {len(files)} JSONL files...[/cyan]")
    
    rows = []
    total_files = len(files)
    report_every = max(1, total_files // 10)  # Report every 10%
    
    for idx, fp in enumerate(files, 1):
        ds_infer, be_infer = infer_dataset_backend(fp)
        file_rows = 0
        
        for obj in iter_jsonl(fp):
            if obj.get("overall") is True:
                continue  # skip summary rows
            if tag_from_path:
                obj.setdefault("dataset", ds_infer)
                obj.setdefault("backend", be_infer)
            rows.append(obj)
            file_rows += 1
        
        # Report progress every 10%
        if idx % report_every == 0 or idx == total_files:
            pct = 100 * idx / total_files
            console.print(f"  [green]✓[/green] Progress: {idx}/{total_files} files ({pct:.0f}%) | Total rows: {len(rows):,}")
    
    console.print(f"[green]✓ Finished reading all files. Total rows collected: {len(rows):,}[/green]\n")
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

    backend_name = os.path.basename(out_dir)
    console.print(f"\n[yellow]📊 Creating split for: {backend_name}[/yellow]")
    console.print(f"  Input rows: {len(rows):,}")
    
    # Assign stable pooled_id per output pool
    for i, r in enumerate(rows):
        r.setdefault("_pooled_id", i)

    console.print(f"  [dim]Writing pooled.jsonl...[/dim]")
    pooled_path = os.path.join(out_dir, "pooled.jsonl")
    with open(pooled_path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    console.print(f"  [dim]Performing stratified split (seed={seed})...[/dim]")
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
    
    console.print(f"  [dim]Writing split files...[/dim]")
    with open(os.path.join(out_dir, "split_map.json"), "w", encoding="utf-8") as w:
        json.dump(split_map, w, ensure_ascii=False, indent=2)

    write_jsonl(os.path.join(out_dir, "train.jsonl"), rows, splits["train"])
    write_jsonl(os.path.join(out_dir, "val.jsonl"),   rows, splits["val"])
    write_jsonl(os.path.join(out_dir, "test.jsonl"),  rows, splits["test"])

    # Create a nice summary table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Split", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")
    
    for split_name in ["train", "val", "test"]:
        count = len(splits[split_name])
        pct = 100 * count / len(rows)
        table.add_row(split_name, f"{count:,}", f"{pct:.1f}%")
    
    console.print(table)
    console.print(f"[green]✓ Completed {backend_name}[/green]\n")


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

    args = ap.parse_args()

    # Print startup banner
    console.print(Panel.fit(
        "[bold cyan]Probe Data Split Creator[/bold cyan]\n"
        f"Root: {args.root}\n"
        f"Output: {args.out_dir}\n"
        f"Split: {args.train_frac:.0%} train / {args.val_frac:.0%} val / {args.test_frac:.0%} test\n"
        f"Seed: {args.seed}",
        title="🔬 Make Probe Split",
        border_style="cyan"
    ))

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Discover files and pool everything
    console.print(f"\n[cyan]🔍 Discovering JSONL files in {args.root}...[/cyan]")
    files = sorted(glob.glob(os.path.join(args.root, args.glob), recursive=True))
    
    if not files:
        console.print(f"[red]❌ No JSONL files found under {args.root} with pattern {args.glob}[/red]")
        raise SystemExit(1)
    
    console.print(f"[green]✓ Found {len(files)} JSONL files[/green]")

    all_rows = pool_rows(files, tag_from_path=not args.no_tag_from_path)
    
    if not all_rows:
        console.print("[red]❌ No rows found after filtering (only 'overall' records?)[/red]")
        raise SystemExit(1)

    # 2) Collect per-backend subsets
    console.print(f"[cyan]📋 Organizing rows by backend...[/cyan]")
    by_backend: Dict[str, List[dict]] = {}
    for r in all_rows:
        be = r.get("backend", "_unknown")
        by_backend.setdefault(be, []).append(r)
    
    # Show backend summary
    backend_table = Table(show_header=True, header_style="bold magenta")
    backend_table.add_column("Backend", style="cyan")
    backend_table.add_column("Row Count", justify="right", style="green")
    backend_table.add_column("Percentage", justify="right", style="yellow")
    
    for be, rows in sorted(by_backend.items()):
        pct = 100 * len(rows) / len(all_rows)
        backend_table.add_row(be, f"{len(rows):,}", f"{pct:.1f}%")
    
    console.print(backend_table)

    # 3) Global pooled split (optional)
    if not args.skip_global:
        console.print(f"\n[bold yellow]═══ Creating Global Split (All Backends) ═══[/bold yellow]")
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
    console.print(f"[bold yellow]═══ Creating Per-Backend Splits ═══[/bold yellow]")
    for i, (be, rows) in enumerate(sorted(by_backend.items()), 1):
        console.print(f"\n[dim]Backend {i}/{len(by_backend)}[/dim]")
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
    
    # Final summary
    console.print(Panel.fit(
        f"[bold green]✓ All splits created successfully![/bold green]\n\n"
        f"Total rows processed: {len(all_rows):,}\n"
        f"Backends: {len(by_backend)}\n"
        f"Output directory: {args.out_dir}",
        title="🎉 Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()