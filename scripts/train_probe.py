# scripts/train_probe.py
# Multi-probe trainer with plots and artifacts.
# Updated with super-generalizable features for cross-model transfer (Qwen family)

"""
Usage:
    # Standard training (with hidden states)
    python -m scripts.train_probe \
        --data_train data/probe_splits_standard/qwen2.5_7b/train.jsonl \
        --data_val data/probe_splits_standard/qwen2.5_7b/val.jsonl \
        --output_dir backend_qwen_standard/probe_mlp \
        --use_hidden

    # Generalizable features only (excludes answer_len, format-specific)
    python -m scripts.train_probe \
        --data_train data/probe_splits_standard/qwen2.5_7b/train.jsonl \
        --data_val data/probe_splits_standard/qwen2.5_7b/val.jsonl \
        --output_dir backend_qwen_standard/probe_mlp_generalizable \
        --use_hidden \
        --only_generalizable

    # Super-generalizable (NO hidden states, only probability features)
    # For testing transfer across Qwen family (1.5B, 7B, 14B, Qwen3-4B)
    python -m scripts.train_probe \
        --data_train data/probe_splits_standard/qwen2.5_7b/train.jsonl \
        --data_val data/probe_splits_standard/qwen2.5_7b/val.jsonl \
        --output_dir backend_qwen_standard/probe_mlp_super_generalizable \
        --super_generalizable

    # Train all probe types for ablation
    python -m scripts.train_probe \
        --data_train ... \
        --data_val ... \
        --output_dir ... \
        --probe_type all
"""

import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    precision_recall_curve, roc_curve
)
from sklearn.isotonic import IsotonicRegression
import joblib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# ------------------------ Feature Definitions ------------------------

# All scalar features available in standard prompts data
SCALAR_KEYS = [
    # Probability-based features (super-generalizable)
    "entropy_mean", "entropy_std",
    "margin_mean", "margin_min",
    "lp_mean", "seq_conf",
    # Answer metadata
    "answer_len", "is_unknown",
]

# Hidden state features (4 vectors × 256 dims each = 1024 features)
VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]

# Feature generalizability tiers
# Tier 1: SUPER-GENERALIZABLE (probability-based, format-agnostic, architecture-agnostic)
# These transfer across model families, sizes, and architectures
SUPER_GENERALIZABLE_FEATURES = {
    "entropy_mean",    # Token-level uncertainty
    "entropy_std",     # Uncertainty variance
    "margin_mean",     # Confidence margin
    "margin_min",      # Minimum confidence
    "lp_mean",         # Log probability
    "seq_conf",        # Sequence confidence (mean top-1 prob)
}

# Tier 2: GENERALIZABLE (within same model family, excludes task-specific)
# Same as super-generalizable for standard prompts (no model_confidence anymore)
GENERALIZABLE_FEATURES = SUPER_GENERALIZABLE_FEATURES

# Tier 3: NON-GENERALIZABLE (task-specific)
# These may help within-dataset but hurt transfer
NON_GENERALIZABLE_FEATURES = {
    "answer_len",      # Dataset-specific (varies by task type)
    "is_unknown",      # Task-specific (SQuAD v2, indicates "unanswerable")
}

# ------------------------ Data Loading ------------------------

def load_rows(jsonl_path: Path, show_progress=True, limit=None):
    """
    Load JSONL data with optional progress and limit
    
    Supports two modes:
    1. Direct JSONL file: loads from the file
    2. Indexed split directory: loads using split indices
    """
    if not jsonl_path or not jsonl_path.exists():
        if show_progress:
            console.print(f"[yellow]Warning: {jsonl_path} not found[/yellow]")
        return []
    
    # Check if this is a split directory with indices
    if jsonl_path.is_dir():
        split_map_path = jsonl_path / "split_map.json"
        if split_map_path.exists():
            # This is an indexed split directory
            if show_progress:
                console.print(f"  [dim]Loading from indexed split: {jsonl_path.name}...[/dim]")
            return load_from_indexed_split(jsonl_path, show_progress, limit)
    
    # Otherwise, load as direct JSONL file
    rows = []
    if show_progress:
        console.print(f"  [dim]Loading {jsonl_path.name}...[/dim]")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    
    if show_progress:
        console.print(f"  [green]✓[/green] Loaded {len(rows):,} rows")
    
    return rows


def load_from_indexed_split(split_dir: Path, show_progress=True, limit=None):
    """
    Load data from indexed split directory
    
    Expected structure:
    split_dir/
      ├── split_map.json  (contains indices for train/val/test)
      ├── file_index.json (contains file paths and metadata)
      └── (optional) train_indices.txt, val_indices.txt, test_indices.txt
    """
    # Determine which split to load based on directory name
    split_name = None
    if "train" in split_dir.name.lower():
        split_name = "train"
    elif "val" in split_dir.name.lower():
        split_name = "val"
    elif "test" in split_dir.name.lower():
        split_name = "test"
    else:
        # If directory doesn't indicate split, default to train
        split_name = "train"
    
    # Load split metadata
    with open(split_dir / "split_map.json") as f:
        split_map = json.load(f)
    
    with open(split_dir / "file_index.json") as f:
        file_index = json.load(f)
    
    # Get indices for this split
    split_indices = split_map["splits"][split_name]
    
    if limit:
        split_indices = split_indices[:limit]
    
    if show_progress:
        console.print(f"  [dim]Loading {len(split_indices):,} rows from {split_name} split...[/dim]")
    
    # Build lookup for fast access
    file_metadata = file_index["files"]
    
    def get_file_and_line(global_idx):
        """Map global row index to (file_idx, line_in_file)"""
        for i, meta in enumerate(file_metadata):
            if meta["start_idx"] <= global_idx < meta["end_idx"]:
                line_in_file = global_idx - meta["start_idx"]
                return i, line_in_file
        return None, None
    
    # Group indices by file for efficient reading
    indices_by_file = {}
    for global_idx in split_indices:
        file_idx, line_num = get_file_and_line(global_idx)
        if file_idx is not None:
            indices_by_file.setdefault(file_idx, []).append((global_idx, line_num))
    
    # Read files and collect rows
    rows_dict = {}
    for file_idx, indices_in_file in indices_by_file.items():
        filepath = file_metadata[file_idx]["path"]
        line_nums_needed = {line_num for _, line_num in indices_in_file}
        
        with open(filepath, encoding="utf-8") as f:
            current_line = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    if obj.get("overall") is True:
                        continue
                    
                    if current_line in line_nums_needed:
                        # Find all global indices that map to this line
                        for global_idx, ln in indices_in_file:
                            if ln == current_line:
                                rows_dict[global_idx] = obj
                    
                    current_line += 1
                except Exception:
                    continue
    
    # Return rows in split order
    rows = []
    for global_idx in split_indices:
        if global_idx in rows_dict:
            rows.append(rows_dict[global_idx])
    
    if show_progress:
        console.print(f"  [green]✓[/green] Loaded {len(rows):,} rows from indexed split")
    
    return rows


def to_1d(arr):
    """Convert array/list to fixed 256-dim numpy array"""
    if arr is None:
        return None
    a = np.array(arr, dtype=float).ravel()
    if a.size < 256:
        a = np.pad(a, (0, 256 - a.size))
    elif a.size > 256:
        a = a[:256]
    return a


def filter_scalar_keys(exclude_features=None, only_generalizable=False, super_generalizable=False):
    """
    Filter SCALAR_KEYS based on generalizability requirements
    
    NOTE: With standard prompts, only_generalizable and super_generalizable 
    are identical (both exclude answer_len and is_unknown)
    
    Args:
        exclude_features: Explicit set of features to exclude
        only_generalizable: Exclude task-specific features (answer_len, is_unknown)
        super_generalizable: Same as only_generalizable for standard prompts
    """
    if super_generalizable or only_generalizable:
        # Both exclude task-specific features
        return [k for k in SCALAR_KEYS if k in SUPER_GENERALIZABLE_FEATURES]
    elif exclude_features:
        # Custom exclusion
        return [k for k in SCALAR_KEYS if k not in exclude_features]
    else:
        # All features (8 total)
        return SCALAR_KEYS


def build_df(rows, use_hidden=True, label_key="correct", show_progress=True, 
             exclude_features=None, only_generalizable=False, super_generalizable=False):
    """
    Build feature dataframe
    
    Args:
        rows: List of data rows
        use_hidden: Include hidden state features
        label_key: Key for label (usually "correct")
        show_progress: Show progress messages
        exclude_features: Explicit features to exclude
        only_generalizable: Use only generalizable features
        super_generalizable: Use only super-generalizable features (no hidden, no format-specific)
    """
    n_rows = len(rows)
    
    if show_progress:
        console.print(f"  [dim]Building feature matrix for {n_rows:,} examples...[/dim]")
    
    # Super-generalizable mode overrides use_hidden
    if super_generalizable:
        use_hidden = False
        if show_progress:
            console.print(f"  [cyan]Super-generalizable mode: using only probability features[/cyan]")
    
    # Filter scalar keys based on generalizability
    scalar_keys = filter_scalar_keys(exclude_features, only_generalizable, super_generalizable)
    
    if show_progress:
        if super_generalizable:
            console.print(f"  [yellow]Super-generalizable: {len(scalar_keys)} features (no hidden states)[/yellow]")
        elif only_generalizable:
            console.print(f"  [yellow]Generalizable: {len(scalar_keys)} features + {'hidden states' if use_hidden else 'no hidden'}[/yellow]")
        elif exclude_features:
            excluded = [k for k in SCALAR_KEYS if k not in scalar_keys]
            console.print(f"  [yellow]Excluding: {', '.join(excluded)}[/yellow]")
    
    # Pre-compute feature dimensions
    n_scalar = len(scalar_keys)
    n_vector = len(VECTOR_KEYS) * 256 if use_hidden else 0
    n_features = n_scalar + n_vector
    
    # Pre-allocate arrays for efficiency
    X = np.full((n_rows, n_features), np.nan, dtype=np.float32)
    y = np.zeros(n_rows, dtype=np.int32)
    ids = np.arange(n_rows)
    
    # Build column names
    col_names = list(scalar_keys)
    if use_hidden:
        for vec_name in VECTOR_KEYS:
            col_names.extend([f"{vec_name}_{j}" for j in range(256)])
    
    # Process rows in chunks
    chunk_size = 10000
    for chunk_start in range(0, n_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_rows)
        
        for i in range(chunk_start, chunk_end):
            r = rows[i]
            
            # Fill scalar features
            for j, k in enumerate(scalar_keys):
                val = r.get(k, np.nan)
                X[i, j] = val if val is not None else np.nan
            
            # Fill vector features
            if use_hidden:
                col_idx = n_scalar
                for vec_name in VECTOR_KEYS:
                    vec = to_1d(r.get(vec_name))
                    if vec is not None:
                        X[i, col_idx:col_idx+256] = vec
                    col_idx += 256
            
            # Label (support both "correct" and "em")
            y[i] = int(r.get(label_key, r.get("correct", r.get("em", 0))))
            ids[i] = r.get("idx", i)
        
        if show_progress and (chunk_end % 50000 == 0 or chunk_end == n_rows):
            pct = 100 * chunk_end / n_rows
            console.print(f"    [green]→[/green] Processed {chunk_end:,}/{n_rows:,} ({pct:.0f}%)")
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=col_names)
    
    if show_progress:
        console.print(f"  [green]✓[/green] Feature matrix: {df.shape}")
    
    return df, y, ids

# ------------------------ Probe Models ------------------------

def make_pipeline_mlp(use_hidden: bool, super_generalizable: bool, random_state: int):
    """MLP probe with memory-efficient SGD solver"""
    # Adjust architecture based on feature set
    if super_generalizable:
        # Smaller network for fewer features (~6 features)
        hidden_sizes = (64, 32)
    elif use_hidden:
        # Larger network for many features (~1031 features)
        hidden_sizes = (256, 128)
    else:
        # Medium network for scalar features only (~12 features)
        hidden_sizes = (128, 64)
    
    base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation="relu",
            solver="sgd",  # Memory-efficient mini-batch solver
            alpha=1e-4,
            batch_size=1024,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            momentum=0.9,
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
            verbose=False,
        )),
    ])
    return base


def make_calibrated(estimator, y_train_len: int):
    """Add isotonic or sigmoid calibration"""
    method = "isotonic" if y_train_len >= 2000 else "sigmoid"
    return CalibratedClassifierCV(estimator, method=method, cv=3)


def make_pipeline_logreg():
    """Logistic regression probe"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
        ("clf",    LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
        ))
    ])


def make_pipeline_logreg_cal(y_train_len: int):
    """Calibrated logistic regression"""
    base = make_pipeline_logreg()
    return make_calibrated(base, y_train_len)


def make_pipeline_tree(max_depth=10, min_samples_leaf=20):
    """Decision tree probe"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        ))
    ])

# ------------------------ Evaluation ------------------------

def evaluate_from_proba(proba, y_true, split_name, threshold=0.5):
    """Compute metrics from predicted probabilities"""
    metrics = {"split": split_name, "n": len(y_true)}
    
    # Basic accuracy with provided threshold
    y_pred = (proba >= threshold).astype(int)
    metrics["accuracy"] = float(np.mean(y_pred == y_true))
    metrics["threshold"] = float(threshold)  # Store threshold used
    
    # Probabilistic metrics
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, proba))
    except:
        metrics["auc_roc"] = None
    
    try:
        metrics["auc_pr"] = float(average_precision_score(y_true, proba))
    except:
        metrics["auc_pr"] = None
    
    try:
        metrics["brier"] = float(brier_score_loss(y_true, proba))
    except:
        metrics["brier"] = None
    
    try:
        # Use epsilon to avoid log(0)
        proba_safe = np.clip(proba, 1e-10, 1 - 1e-10)
        metrics["log_loss"] = float(log_loss(y_true, proba_safe))
    except:
        metrics["log_loss"] = None
    
    # Class balance
    pos_rate = float(np.mean(y_true))
    metrics["pos_rate"] = pos_rate
    
    return metrics


def plot_curves(probe_dir: Path, split_name: str, y_true, proba):
    """Plot ROC and PR curves"""
    try:
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_roc = roc_auc_score(y_true, proba)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.3f}")
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({split_name})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(probe_dir / f"roc_{split_name}.png", dpi=100)
        plt.close()
        
        # PR curve
        prec, rec, _ = precision_recall_curve(y_true, proba)
        auc_pr = average_precision_score(y_true, proba)
        
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, label=f"AP = {auc_pr:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve ({split_name})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(probe_dir / f"pr_{split_name}.png", dpi=100)
        plt.close()
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not plot curves for {split_name}: {e}[/yellow]")

# ------------------------ Training ------------------------

def train_and_eval_probe(
    probe_type: str,
    X_tr, y_tr, ids_tr,
    X_va, y_va, ids_va,
    X_te, y_te, ids_te,
    use_hidden: bool,
    super_generalizable: bool,
    probe_dir: Path,
    no_calibration: bool = False,
    feature_names=None
):
    """Train probe and evaluate on all splits"""
    
    probe_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold cyan]Training {probe_type} probe...[/bold cyan]")
    console.print(f"  Training samples: {len(y_tr):,}")
    console.print(f"  Features: {X_tr.shape[1]}")
    
    # Build model
    if probe_type == "mlp":
        model = make_pipeline_mlp(use_hidden, super_generalizable, random_state=42)
        if not no_calibration:
            model = make_calibrated(model, len(y_tr))
    elif probe_type == "logreg":
        model = make_pipeline_logreg()
    elif probe_type == "logreg_cal":
        model = make_pipeline_logreg_cal(len(y_tr))
    elif probe_type == "tree":
        model = make_pipeline_tree()
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    
    # Train
    console.print(f"  [dim]Fitting model...[/dim]")
    model.fit(X_tr, y_tr)
    console.print(f"  [green]✓[/green] Training complete")
    
    # Save model
    model_path = probe_dir / "probe_model.joblib"
    joblib.dump(model, model_path)
    console.print(f"  [green]✓[/green] Saved model to {model_path.name}")
    
    # Save feature names
    if feature_names is not None:
        with open(probe_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)
    
    # Evaluate on all splits
    def predict_proba(X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)
    
    split_evals = []
    
    # Find optimal threshold on validation set
    console.print("\n[bold]Finding Optimal Threshold[/bold]")
    if X_va is not None and len(X_va) > 0:
        from sklearn.metrics import f1_score
        proba_val = predict_proba(X_va)
        
        best_f1 = 0
        optimal_threshold = 0.5
        for thresh in np.arange(0.1, 0.9, 0.02):
            y_pred_thresh = (proba_val >= thresh).astype(int)
            f1 = f1_score(y_va, y_pred_thresh, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = thresh
        
        console.print(f"  [cyan]→ Optimal threshold: {optimal_threshold:.3f} (val F1={best_f1:.3f})[/cyan]")
    else:
        optimal_threshold = 0.5
        console.print(f"  [yellow]→ No validation set, using default threshold: 0.5[/yellow]")
    
    def do_eval(X, y, ids, name):
        if X is None or len(X) == 0:
            return
        console.print(f"  [dim]Evaluating on {name}...[/dim]")
        proba = predict_proba(X)
        
        # Use optimal threshold instead of 0.5
        metrics = evaluate_from_proba(proba, y, name, threshold=optimal_threshold)
        split_evals.append(metrics)
        
        # Save predictions
        pd.DataFrame({
            "id": ids,
            "y_true": y,
            "p_correct": proba,
            "y_pred": (proba >= optimal_threshold).astype(int)
        }).to_csv(probe_dir / f"pred_{name}.csv", index=False)
        
        # Plot curves
        plot_curves(probe_dir, name, y, proba)
        
        # Print metrics
        console.print(f"    [green]{name}:[/green] "
                     f"Acc={metrics['accuracy']:.3f} "
                     f"AUC={metrics.get('auc_roc', 0):.3f} "
                     f"AP={metrics.get('auc_pr', 0):.3f}")
    
    do_eval(X_tr, y_tr, ids_tr, "train")
    if X_va is not None and len(X_va) > 0:
        do_eval(X_va, y_va, ids_va, "val")
    if X_te is not None and len(X_te) > 0:
        do_eval(X_te, y_te, ids_te, "test")
    
    # Save report
    with open(probe_dir / "report.json", "w") as f:
        json.dump(split_evals, f, indent=2)
    
    # Save metadata
    meta = {
        "probe_type": probe_type,
        "use_hidden": use_hidden,
        "super_generalizable": super_generalizable,
        "n_features": X_tr.shape[1],
        "feature_names": feature_names,
        "model_path": str(model_path),
        "optimal_threshold": float(optimal_threshold),  # NEW: save threshold
    }
    with open(probe_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return {"probe_dir": str(probe_dir), "metrics": split_evals}

# ------------------------ Main ------------------------

def main():
    parser = argparse.ArgumentParser(description="Train confidence probes")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to indexed split directory (contains split_map.json)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit training examples (for testing)")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for probe models")
    
    # Features
    parser.add_argument("--use_hidden", action="store_true",
                       help="Include 4x256 hidden state features")
    parser.add_argument("--only_generalizable", action="store_true",
                       help="Use only generalizable features (excludes answer_len, format-specific)")
    parser.add_argument("--super_generalizable", action="store_true",
                       help="Use only super-generalizable features (probability-based, no hidden states)")
    parser.add_argument("--exclude_features", type=str, default=None,
                       help="Comma-separated features to exclude")
    
    # Probe type
    parser.add_argument("--probe_type", type=str, default="mlp",
                       choices=["mlp", "logreg", "logreg_cal", "tree", "all"])
    
    # Training options
    parser.add_argument("--label_key", type=str, default="correct",
                       help="Label key in data (default: 'correct')")
    parser.add_argument("--no_calibration", action="store_true",
                       help="Skip calibration (saves memory)")
    
    args = parser.parse_args()
    
    # Print configuration
    console.print(Panel.fit(
        "[bold cyan]Confidence Probe Training[/bold cyan]\n"
        f"Data: {args.data_dir}\n"
        f"Output: {args.output_dir}\n"
        f"Probe type: {args.probe_type}\n"
        f"Features: {'Super-generalizable' if args.super_generalizable else 'Generalizable' if args.only_generalizable else 'All'}\n"
        f"Hidden states: {args.use_hidden and not args.super_generalizable}",
        border_style="cyan"
    ))
    
    # Parse feature exclusion
    exclude_features = None
    if args.exclude_features:
        exclude_features = set(f.strip() for f in args.exclude_features.split(','))
        console.print(f"[yellow]Excluding: {', '.join(sorted(exclude_features))}[/yellow]")
    
    # Load data from indexed split directory
    console.print("\n[bold]Loading Data from Indexed Splits[/bold]")
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        console.print(f"[red]Error: Data directory not found: {data_dir}[/red]")
        return
    
    # The directory contains split_map.json with train/val/test indices
    # We'll load each split separately by looking at the split_map
    with open(data_dir / "split_map.json") as f:
        split_map = json.load(f)
    
    with open(data_dir / "file_index.json") as f:
        file_index = json.load(f)
    
    # Helper function to load a specific split
    def load_split_data(split_name):
        split_indices = split_map["splits"].get(split_name, [])
        if not split_indices:
            return []
        
        if args.limit:
            split_indices = split_indices[:args.limit]
        
        console.print(f"  [dim]Loading {split_name}: {len(split_indices):,} examples...[/dim]")
        
        file_metadata = file_index["files"]
        
        def get_file_and_line(global_idx):
            for i, meta in enumerate(file_metadata):
                if meta["start_idx"] <= global_idx < meta["end_idx"]:
                    return i, global_idx - meta["start_idx"]
            return None, None
        
        # Group by file
        indices_by_file = {}
        for global_idx in split_indices:
            file_idx, line_num = get_file_and_line(global_idx)
            if file_idx is not None:
                indices_by_file.setdefault(file_idx, []).append((global_idx, line_num))
        
        # Read files
        rows_dict = {}
        for file_idx, indices_in_file in indices_by_file.items():
            filepath = file_metadata[file_idx]["path"]
            line_nums_needed = {line_num for _, line_num in indices_in_file}
            
            with open(filepath, encoding="utf-8") as f:
                current_line = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("overall") is True:
                            continue
                        if current_line in line_nums_needed:
                            for global_idx, ln in indices_in_file:
                                if ln == current_line:
                                    rows_dict[global_idx] = obj
                        current_line += 1
                    except Exception:
                        continue
        
        # Return in order
        rows = [rows_dict[i] for i in split_indices if i in rows_dict]
        console.print(f"  [green]✓[/green] Loaded {len(rows):,} rows from {split_name}")
        return rows
    
    # Load all splits
    rows_tr = load_split_data("train")
    rows_va = load_split_data("val")
    rows_te = load_split_data("test")
    
    if not rows_tr:
        console.print("[red]Error: No training data loaded[/red]")
        return
    
    # Build features
    console.print("\n[bold]Building Features[/bold]")
    X_tr, y_tr, ids_tr = build_df(
        rows_tr,
        use_hidden=args.use_hidden,
        label_key=args.label_key,
        exclude_features=exclude_features,
        only_generalizable=args.only_generalizable,
        super_generalizable=args.super_generalizable
    )
    
    X_va, y_va, ids_va = None, None, None
    if rows_va:
        X_va, y_va, ids_va = build_df(
            rows_va,
            use_hidden=args.use_hidden,
            label_key=args.label_key,
            exclude_features=exclude_features,
            only_generalizable=args.only_generalizable,
            super_generalizable=args.super_generalizable,
            show_progress=False
        )
    
    X_te, y_te, ids_te = None, None, None
    if rows_te:
        X_te, y_te, ids_te = build_df(
            rows_te,
            use_hidden=args.use_hidden,
            label_key=args.label_key,
            exclude_features=exclude_features,
            only_generalizable=args.only_generalizable,
            super_generalizable=args.super_generalizable,
            show_progress=False
        )
    
    feature_names = list(X_tr.columns)
    
    # Train probes
    output_dir = Path(args.output_dir)
    
    probe_types = ["mlp", "logreg", "logreg_cal", "tree"] if args.probe_type == "all" else [args.probe_type]
    
    results = []
    for pt in probe_types:
        probe_dir = output_dir / f"probe_{pt}"
        result = train_and_eval_probe(
            pt, X_tr, y_tr, ids_tr,
            X_va, y_va, ids_va,
            X_te, y_te, ids_te,
            use_hidden=args.use_hidden,
            super_generalizable=args.super_generalizable,
            probe_dir=probe_dir,
            no_calibration=args.no_calibration,
            feature_names=feature_names
        )
        results.append(result)
    
    # Summary table
    console.print("\n[bold cyan]Training Complete![/bold cyan]")
    table = Table(title="Results Summary", show_header=True, header_style="bold magenta")
    table.add_column("Probe", style="cyan")
    table.add_column("Split", style="yellow")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("AUC-ROC", justify="right", style="green")
    table.add_column("AUC-PR", justify="right", style="green")
    
    for res in results:
        for metrics in res["metrics"]:
            table.add_row(
                Path(res["probe_dir"]).name,
                metrics["split"],
                f"{metrics['accuracy']:.3f}",
                f"{metrics.get('auc_roc', 0):.3f}",
                f"{metrics.get('auc_pr', 0):.3f}"
            )
    
    console.print(table)
    console.print(f"\n[green]✓ All probes saved to {output_dir}[/green]")


if __name__ == "__main__":
    main()