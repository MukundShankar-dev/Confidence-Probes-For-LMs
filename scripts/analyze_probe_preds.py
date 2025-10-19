#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze probe predictions per dataset with class balance and optional fixed-threshold eval.

Usage:
  python -m scripts.analyze_probe_preds \
    --probe_dir data/probe_splits_80_10_10/probe_models/backend_llama/probe_logreg_cal \
    --splits_root data/probe_splits_80_10_10/pooled_all_backends \
    --by dataset \
    --base_only           # optional: exclude dataset=="supplementary"
    --fixed_thr 0.40      # optional: also report Acc/F1 at a fixed threshold

Assumptions:
- Predictions: <probe_dir>/pred_{train,val,test}.csv with columns: id,y_true,p_right
- GT rows: <splits_root>/{train,val,test}.jsonl with "idx","dataset","backend"
"""

import argparse, json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, precision_recall_curve, f1_score, accuracy_score
)

SPLITS = ("train", "val", "test")

def load_jsonl_with_ids(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("overall") is True:
                continue
            rows.append({
                "id": obj.get("idx", i),
                "dataset": obj.get("dataset", ""),
                "backend": obj.get("backend", "")
            })
    return pd.DataFrame(rows)

def load_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize columns just in case
    r = {c.lower(): c for c in df.columns}
    need = {"id","y_true","p_right"}
    if not need.issubset({c.lower() for c in df.columns}):
        raise SystemExit(f"{path} must contain columns id,y_true,p_right")
    return df.rename(columns={r["id"]:"id", r["y_true"]:"y_true", r["p_right"]:"p_right"})

def metrics_from_proba(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    out: Dict[str,float] = {}
    out["n"] = int(len(y))
    out["pos_rate"] = float(np.mean(y)) if len(y) else float("nan")
    out["AUPRC"] = float(average_precision_score(y, p)) if len(y) else float("nan")
    try:
        out["AUROC"] = float(roc_auc_score(y, p)) if len(set(y))>1 else float("nan")
    except Exception:
        out["AUROC"] = float("nan")
    out["Brier"] = float(brier_score_loss(y, p)) if len(y) else float("nan")
    try:
        out["NLL"] = float(log_loss(y, np.clip(p, 1e-7, 1-1e-7)))
    except Exception:
        out["NLL"] = float("nan")
    # best F1 (informative but don't tune on test in practice)
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = (2*prec*rec)/np.clip(prec+rec, 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    out["bestF1"] = float(np.nanmax(f1))
    out["best_thr"] = float(thr[best_idx-1]) if best_idx>0 and best_idx-1<len(thr) else 0.5
    return out

def at_fixed_threshold(y: np.ndarray, p: np.ndarray, thr: float):
    yhat = (p >= thr).astype(int)
    return {
        "thr": float(thr),
        "Acc": float(accuracy_score(y, yhat)),
        "F1":  float(f1_score(y, yhat)) if len(set(y))>1 else float("nan")
    }

def summarize(split: str, merged: pd.DataFrame, group_keys: List[str], base_only: bool, fixed_thr: float=None):
    df = merged.copy()
    if base_only:
        df = df[df["dataset"].str.lower() != "supplementary"]

    print(f"\n=== {split.upper()} (base_only={base_only}) ===")
    # overall
    m = metrics_from_proba(df["y_true"].to_numpy(), df["p_right"].to_numpy())
    print(f"[overall] n={m['n']:5d}  pos={m['pos_rate']:.3f}  AUROC={m['AUROC']:.3f}  "
          f"AUPRC={m['AUPRC']:.3f}  Brier={m['Brier']:.3f}  NLL={m['NLL']:.3f}  "
          f"bestF1={m['bestF1']:.3f}  thr={m['best_thr']:.3f}")
    if fixed_thr is not None:
        fx = at_fixed_threshold(df["y_true"].to_numpy(), df["p_right"].to_numpy(), fixed_thr)
        print(f"  @thr={fx['thr']:.3f}: Acc={fx['Acc']:.3f}  F1={fx['F1']:.3f}")

    # per group
    by = group_keys if group_keys else ["dataset"]
    for key, g in df.groupby(by):
        key_str = key if isinstance(key, str) else "/".join(map(str, key))
        mg = metrics_from_proba(g["y_true"].to_numpy(), g["p_right"].to_numpy())
        print(f"[{key_str}] n={mg['n']:5d}  pos={mg['pos_rate']:.3f}  AUROC={mg['AUROC']:.3f}  "
              f"AUPRC={mg['AUPRC']:.3f}  Brier={mg['Brier']:.3f}  NLL={mg['NLL']:.3f}  "
              f"bestF1={mg['bestF1']:.3f}  thr={mg['best_thr']:.3f}")
        if fixed_thr is not None:
            fxg = at_fixed_threshold(g["y_true"].to_numpy(), g["p_right"].to_numpy(), fixed_thr)
            print(f"  @thr={fxg['thr']:.3f}: Acc={fxg['Acc']:.3f}  F1={fxg['F1']:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--splits_root", required=True)
    ap.add_argument("--base_only", action="store_true")
    ap.add_argument("--by", type=str, default="dataset")
    ap.add_argument("--fixed_thr", type=float, default=None)
    args = ap.parse_args()

    probe_dir = Path(args.probe_dir)
    splits_root = Path(args.splits_root)
    group_keys = [x.strip() for x in args.by.split(",") if x.strip()]

    for split in SPLITS:
        pred_csv = probe_dir / f"pred_{split}.csv"
        gt_jsonl = splits_root / f"{split}.jsonl"
        if not pred_csv.exists() or not gt_jsonl.exists():
            continue
        preds = load_pred_csv(pred_csv)
        gt = load_jsonl_with_ids(gt_jsonl)
        merged = preds.merge(gt, on="id", how="left")
        summarize(split, merged, group_keys, base_only=args.base_only, fixed_thr=args.fixed_thr)

if __name__ == "__main__":
    main()
