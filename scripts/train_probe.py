# train_probe.py
# Trains a per-backend (llama, qwen) probability probe on your JSONL splits
# under data/probe_splits_80_10_10. Saves calibrated models + reports.

import argparse, json, os, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    precision_recall_curve
)
import joblib


def find_split_files(data_dir: Path, model_key: str):
    """
    Locate train/val/test jsonl files for a given backend model key (e.g., 'llama', 'qwen').
    We search recursively for files whose path contains 'backend_<model_key>' and
    end with one of ['train.jsonl', 'valid.jsonl', 'val.jsonl', 'test.jsonl'].
    Returns a dict with possible keys 'train', 'val', 'test' -> Path
    """
    patterns = ["**/*.jsonl"]
    candidates = []
    for pat in patterns:
        for p in data_dir.glob(pat):
            if f"backend_{model_key}" in str(p):
                name = p.name.lower()
                if name.endswith("train.jsonl"):
                    candidates.append(("train", p))
                elif name.endswith("valid.jsonl") or name.endswith("val.jsonl"):
                    candidates.append(("val", p))
                elif name.endswith("test.jsonl"):
                    candidates.append(("test", p))
    chosen = {}
    for split in ["train", "val", "test"]:
        split_paths = [pp for s, pp in candidates if s == split]
        if split_paths:
            chosen[split] = sorted(split_paths, key=lambda x: len(str(x)))[-1]
    return chosen


def load_rows(jsonl_path: Path):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    # tolerate occasional parse issues
                    pass
    return rows


def to_1d(arr):
    if arr is None:
        return None
    a = np.array(arr, dtype=float).ravel()
    if a.size < 256:
        a = np.pad(a, (0, 256 - a.size))
    elif a.size > 256:
        a = a[:256]
    return a


SCALAR_KEYS = [
    "model_confidence", "lp_mean", "seq_conf",
    "entropy_mean", "entropy_std",
    "margin_mean", "margin_min",
    "rescore_logp", "answer_len",
    "parsed_json_ok", "parsed_p_true_ok", "is_unknown",
]
VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]


def build_df(rows, use_hidden=True, label_key="em"):
    feats, labels, ids = [], [], []
    for i, r in enumerate(rows):
        x = {}
        for k in SCALAR_KEYS:
            x[k] = r.get(k, np.nan)
        if use_hidden:
            for name in VECTOR_KEYS:
                vec = to_1d(r.get(name))
                if vec is not None:
                    for j, v in enumerate(vec):
                        x[f"{name}_{j}"] = float(v)
                else:
                    for j in range(256):
                        x[f"{name}_{j}"] = np.nan
        feats.append(x)
        labels.append(int(r.get(label_key, 0)))
        ids.append(r.get("idx", i))
    df = pd.DataFrame(feats)
    y = np.array(labels, dtype=int)
    ids = np.array(ids)
    return df, y, ids


def train_probe(train_df, y_train, use_hidden=True, random_state=7):
    """
    Returns a calibrated sklearn pipeline that outputs probabilities.
    """
    base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 64) if use_hidden else (64, 32),
            activation="relu",
            alpha=1e-4,              # L2
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=60,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=random_state,
            verbose=False,
        )),
    ])
    # Keep probabilities calibrated: isotonic if enough data; otherwise Platt
    method = "isotonic" if len(y_train) >= 2000 else "sigmoid"
    calibrated = CalibratedClassifierCV(base, method=method, cv=3)
    calibrated.fit(train_df.values, y_train)
    return calibrated


def evaluate(model, X, y, split_name):
    proba = model.predict_proba(X)[:, 1]
    out = {}
    out["AUROC"] = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
    out["AUPRC"] = float(average_precision_score(y, proba))
    out["Brier"] = float(brier_score_loss(y, proba))
    try:
        out["NLL"] = float(log_loss(y, proba, eps=1e-7))
    except Exception:
        out["NLL"] = float("nan")
    precision, recall, thr = precision_recall_curve(y, proba)
    f1 = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    out["best_threshold"] = float(thr[best_idx-1]) if best_idx > 0 and best_idx-1 < len(thr) else 0.5
    out["best_F1"] = float(np.nanmax(f1))
    out["split"] = split_name
    return out, proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/probe_splits_80_10_10")
    parser.add_argument("--models", type=str, default="llama,qwen", help="comma-separated: llama,qwen")
    parser.add_argument("--use_hidden", action="store_true", help="include 4x256 hidden vectors")
    parser.add_argument("--label_key", type=str, default="em")
    parser.add_argument("--out_dir", type=str, default=None, help="where to save probes (default: <data_dir>/probe_models)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "probe_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    for mk in model_keys:
        split_files = find_split_files(data_dir, mk)
        if not split_files:
            print(f"[WARN] No files found for backend_{mk} under {data_dir}")
            continue
        print(f"== backend_{mk} ==")
        for k, v in split_files.items():
            print(f"  {k}: {v}")

        rows_train = load_rows(split_files.get("train")) if split_files.get("train") else []
        rows_val   = load_rows(split_files.get("val"))   if split_files.get("val") else []
        rows_test  = load_rows(split_files.get("test"))  if split_files.get("test") else []

        if not rows_train:
            print(f"[WARN] backend_{mk}: missing or empty train split; skipping")
            continue

        df_tr, y_tr, id_tr = build_df(rows_train, use_hidden=args.use_hidden, label_key=args.label_key)
        df_va, y_va, id_va = (None, None, None)
        df_te, y_te, id_te = (None, None, None)

        if rows_val:
            df_va, y_va, id_va = build_df(rows_val, use_hidden=args.use_hidden, label_key=args.label_key)
        if rows_test:
            df_te, y_te, id_te = build_df(rows_test, use_hidden=args.use_hidden, label_key=args.label_key)

        model = train_probe(df_tr, y_tr, use_hidden=args.use_hidden)

        model_dir = out_dir / f"backend_{mk}"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "probe_model.joblib")

        feat_names = list(df_tr.columns)
        with open(model_dir / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(feat_names, f, ensure_ascii=False, indent=2)

        split_evals = []

        def do_eval(df, y, ids, split_name):
            if df is None:
                return
            metrics, proba = evaluate(model, df.values, y, split_name)
            split_evals.append(metrics)
            out = pd.DataFrame({
                "id": ids,
                "y_true": y,
                "p_right": proba
            })
            out.to_csv(model_dir / f"pred_{split_name}.csv", index=False)

        do_eval(df_tr, y_tr, id_tr, "train")
        if df_va is not None:
            do_eval(df_va, y_va, id_va, "val")
        if df_te is not None:
            do_eval(df_te, y_te, id_te, "test")

        report_path = model_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(split_evals, f, indent=2)

        print(f"[OK] Saved probe for backend_{mk} to {model_dir}")
        for r in split_evals:
            print(f"  [{r['split']}] AUROC={r['AUROC']:.3f} AUPRC={r['AUPRC']:.3f} Brier={r['Brier']:.3f} bestF1={r['best_F1']:.3f} thr={r['best_threshold']:.3f}")


if __name__ == "__main__":
    main()
