"""
Efficient probe data split creator

Example usage:

# For standard prompts, per-model training
python -m scripts.make_probe_split_efficient \
  --root data/probe_standard/qwen2.5_7b \
  --out_dir data/probe_splits_standard/qwen2.5_7b \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15

# For transfer testing (test only)
python -m scripts.make_probe_split_efficient \
  --root data/probe_transfer_standard/qwen2.5_1.5b \
  --out_dir data/probe_splits_standard/qwen2.5_1.5b_transfer \
  --train_frac 0.0 --val_frac 0.0 --test_frac 1.0 \
  --skip_global
"""

import argparse
import glob
import json
import os
import random
import re
from typing import Dict, List, Tuple
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

console = Console()


def infer_dataset_backend(path: str) -> Tuple[str, str]:
    """Infer dataset and backend from path structure"""
    p = path.replace("\\", "/")
    # Match patterns like: /probe_standard/qwen2.5_7b/triviaqa/...
    m = re.search(r"/probe[^/]*/([^/]+)/([^/]+)/", p)
    if m:
        backend_or_model = m.group(1)
        dataset = m.group(2)
        # Swap if dataset is actually first
        if "qwen" in dataset.lower() or "llama" in dataset.lower():
            return backend_or_model, dataset
        return dataset, backend_or_model
    
    # Fallback patterns
    m2 = re.search(r"/probe[^/]*/([^/]+)/", p)
    dataset = m2.group(1) if m2 else "_unknown"
    backend = "_unknown"
    
    # Try to infer backend from filename or parent dir
    if "qwen" in p.lower():
        if "qwen2.5_7b" in p.lower() or "qwen2.5-7b" in p.lower():
            backend = "qwen2.5_7b"
        elif "qwen2.5_1.5b" in p.lower() or "qwen2.5-1.5b" in p.lower():
            backend = "qwen2.5_1.5b"
        elif "qwen2.5_14b" in p.lower() or "qwen2.5-14b" in p.lower():
            backend = "qwen2.5_14b"
        elif "qwen3_4b" in p.lower() or "qwen3-4b" in p.lower():
            backend = "qwen3_4b"
        else:
            backend = "qwen"
    elif "llama" in p.lower():
        backend = "llama31_8b" if "31" in p.lower() or "3.1" in p.lower() else "llama"
    
    return dataset, backend


def count_jsonl_rows(path: str) -> int:
    """Fast row counting, skipping summary rows"""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("overall") is True:
                    continue
                count += 1
            except Exception:
                continue
    return count


def build_file_index(files: List[str], tag_from_path: bool) -> Tuple[List[dict], int]:
    """
    Build index of all files with metadata
    Returns: (file_metadata, total_rows)
    """
    console.print(f"\n[cyan]Building file index...[/cyan]")
    
    file_metadata = []
    global_row_idx = 0
    
    for filepath in track(files, description="  Indexing files"):
        dataset, backend = infer_dataset_backend(filepath) if tag_from_path else ("_unknown", "_unknown")
        row_count = count_jsonl_rows(filepath)
        
        file_metadata.append({
            "path": filepath,
            "dataset": dataset,
            "backend": backend,
            "row_count": row_count,
            "start_idx": global_row_idx,
            "end_idx": global_row_idx + row_count
        })
        
        global_row_idx += row_count
    
    console.print(f"[green]✓ Indexed {global_row_idx:,} rows across {len(files)} files[/green]\n")
    return file_metadata, global_row_idx


def stratified_split_indices(
    file_metadata: List[dict],
    total_rows: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify_keys: Tuple[str, ...] = ("dataset",),
) -> Dict[str, List[int]]:
    """Create stratified splits using global row indices"""
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "fractions must sum to 1.0"
    rnd = random.Random(seed)
    
    # Group row indices by stratification key
    buckets: Dict[Tuple, List[int]] = {}
    
    for file_meta in file_metadata:
        key = tuple(file_meta.get(k, "_default") for k in stratify_keys)
        indices = list(range(file_meta["start_idx"], file_meta["end_idx"]))
        buckets.setdefault(key, []).extend(indices)
    
    # Split each bucket
    splits = {"train": [], "val": [], "test": []}
    for key, idxs in buckets.items():
        rnd.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        
        splits["train"].extend(idxs[:n_train])
        splits["val"].extend(idxs[n_train:n_train + n_val])
        splits["test"].extend(idxs[n_train + n_val:])
    
    # Shuffle final splits
    for split_name in splits:
        rnd.shuffle(splits[split_name])
    
    return splits


def save_split_metadata(
    out_dir: str,
    file_metadata: List[dict],
    splits: Dict[str, List[int]],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    total_rows: int
):
    """Save split metadata (compatible with original format)"""
    os.makedirs(out_dir, exist_ok=True)
    
    split_map = {
        "meta": {
            "seed": seed,
            "counts": {k: len(v) for k, v in splits.items()},
            "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
            "total_rows": total_rows,
            "mode": "indexed",  # Marker
        },
        "splits": splits,
    }
    
    with open(os.path.join(out_dir, "split_map.json"), "w", encoding="utf-8") as f:
        json.dump(split_map, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(out_dir, "file_index.json"), "w", encoding="utf-8") as f:
        json.dump({"files": file_metadata, "total_rows": total_rows}, f, ensure_ascii=False, indent=2)
    
    for split_name, indices in splits.items():
        with open(os.path.join(out_dir, f"{split_name}_indices.txt"), "w") as f:
            for idx in indices:
                f.write(f"{idx}\n")


def make_one_split(
    out_dir: str,
    file_metadata: List[dict],
    total_rows: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify_by_dataset: bool = True
):
    """Create indexed split"""
    os.makedirs(out_dir, exist_ok=True)
    
    name = os.path.basename(out_dir)
    console.print(f"\n[yellow]Creating split: {name}[/yellow]")
    console.print(f"  Rows: {total_rows:,} | Files: {len(file_metadata)}")
    
    splits = stratified_split_indices(
        file_metadata, total_rows,
        train_frac, val_frac, test_frac, seed,
        stratify_keys=("dataset",) if stratify_by_dataset else ("_default",),
    )
    
    save_split_metadata(out_dir, file_metadata, splits, train_frac, val_frac, test_frac, seed, total_rows)
    
    # Summary table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Split", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")
    
    for split_name in ["train", "val", "test"]:
        count = len(splits[split_name])
        pct = 100 * count / total_rows if total_rows > 0 else 0
        table.add_row(split_name, f"{count:,}", f"{pct:.1f}%")
    
    console.print(table)
    console.print(f"[green]✓ Completed {name}[/green]")


def main():
    ap = argparse.ArgumentParser(description="Efficient probe split creator")
    ap.add_argument("--root", default="data/probe")
    ap.add_argument("--glob", default="**/*.jsonl")
    ap.add_argument("--out_dir", default="data/probe_splits")
    ap.add_argument("--train_frac", type=float, default=0.80)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no_tag_from_path", action="store_true")
    ap.add_argument("--skip_global", action="store_true")
    
    args = ap.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]Efficient Probe Split Creator[/bold cyan]\n"
        f"Root: {args.root}\n"
        f"Output: {args.out_dir}\n"
        f"Split: {args.train_frac:.0%} / {args.val_frac:.0%} / {args.test_frac:.0%}\n\n"
        "[yellow]Uses indices - saves 100s of GB![/yellow]",
        border_style="cyan"
    ))
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Discover files
    console.print(f"\n[cyan]Discovering files...[/cyan]")
    files = sorted(glob.glob(os.path.join(args.root, args.glob), recursive=True))
    if not files:
        console.print(f"[red]No files found[/red]")
        raise SystemExit(1)
    console.print(f"[green]✓ Found {len(files)} files[/green]")
    
    # Build index
    file_metadata, total_rows = build_file_index(files, not args.no_tag_from_path)
    if total_rows == 0:
        console.print("[red]No rows found[/red]")
        raise SystemExit(1)
    
    # Organize by backend
    console.print(f"[cyan]Organizing by backend...[/cyan]")
    by_backend: Dict[str, List[dict]] = {}
    backend_rows = {}
    
    for meta in file_metadata:
        be = meta["backend"]
        by_backend.setdefault(be, []).append(meta)
        backend_rows[be] = backend_rows.get(be, 0) + meta["row_count"]
    
    # Summary table
    table = Table(title="Backend Summary", show_header=True, header_style="bold magenta")
    table.add_column("Backend", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("%", justify="right", style="yellow")
    
    for be in sorted(by_backend.keys()):
        rows = backend_rows[be]
        pct = 100 * rows / total_rows
        table.add_row(be, str(len(by_backend[be])), f"{rows:,}", f"{pct:.1f}%")
    
    console.print(table)
    
    # Global split
    if not args.skip_global:
        console.print(f"\n[bold yellow]═══ Global Split ═══[/bold yellow]")
        make_one_split(
            os.path.join(args.out_dir, "pooled_all_backends"),
            file_metadata, total_rows,
            args.train_frac, args.val_frac, args.test_frac, args.seed, True
        )
    
    # Per-backend splits
    console.print(f"\n[bold yellow]═══ Per-Backend Splits ═══[/bold yellow]")
    for i, (be, be_files) in enumerate(sorted(by_backend.items()), 1):
        console.print(f"\n[dim]Backend {i}/{len(by_backend)}[/dim]")
        make_one_split(
            os.path.join(args.out_dir, f"backend_{be}"),
            be_files, backend_rows[be],
            args.train_frac, args.val_frac, args.test_frac, args.seed, True
        )
    
    # Final summary
    console.print(Panel.fit(
        f"[bold green]Success[/bold green]\n\n"
        f"Rows: {total_rows:,}\n"
        f"Backends: {len(by_backend)}\n"
        f"Files: {len(files)}\n\n"
        title="Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
