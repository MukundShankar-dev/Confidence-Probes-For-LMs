#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a lightweight probe (small MLP) to predict P(True) from collected internals.

Inputs: one or more JSONL files produced by collect_internals.py
Output: a .pt with model + normalization stats + feature spec, and a metrics JSON.

USAGE:
  python -m scripts.probe_trainer \
    --data data/probe/triviaqa/*.jsonl \
    --val_frac 0.1 --test_frac 0.1 \
    --epochs 20 --hidden 512 --hidden2 256 \
    --out_dir outputs/probe_llama31_triviaqa

Notes:
- Loss = MSE (Brier-aligned); we predict p_true in [0,1]
- Evaluation: Brier (mean squared error to EM), ROC-AUC (optional)
"""
import argparse, glob, json, os, random, math
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --------------- Utilities ---------------

def read_jsonl_many(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows

def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))

def train_val_test_split(N, val_frac, test_frac, seed=0):
    idx = list(range(N))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(N * val_frac)
    n_test = int(N * test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val+n_test]
    train_idx = idx[n_val+n_test:]
    return train_idx, val_idx, test_idx

def stdz_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd

def stdz_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd

def stack_feature(row: Dict[str, Any], feats: List[str]) -> List[float]:
    """
    Flatten numeric feature fields into a single vector.
    Supports scalars and lists (like h_last_256).
    Missing values -> 0.0
    """
    out = []
    for k in feats:
        v = row.get(k, None)
        if isinstance(v, list):
            out.extend([float(x) if x is not None else 0.0 for x in v])
        elif v is None:
            out.append(0.0)
        else:
            out.append(float(v))
    return out

# --------------- Model ---------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=512, hidden2=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),  # predict p(True) in [0,1]
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# --------------- Main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="One or more JSONL paths (glob accepted).")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--hidden2", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = []
    for pattern in args.data:
        paths.extend(glob.glob(pattern))
    assert paths, "No input files matched."

    rows = read_jsonl_many(paths)

    # Select features (compact but strong). You can add/remove here.
    vector_feats = [
        "h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256",
    ]
    scalar_feats = [
        "lp_mean", "rescore_logp",
        "entropy_mean", "entropy_std", "entropy_last",
        "margin_mean", "margin_min", "margin_last",
        "seq_conf", "answer_len",
        "parsed_json_ok", "parsed_p_true_ok", "is_unknown",
    ]

    feat_keys = []
    for k in vector_feats + scalar_feats:
        feat_keys.append(k)

    # Build X, y
    X_list, y_list = [], []
    for r in rows:
        # label = EM
        if "em" not in r:
            continue
        y_list.append(1.0 if int(r["em"]) == 1 else 0.0)

        # concat feat vectors
        feats = []
        for k in vector_feats:
            v = r.get(k, None)
            if isinstance(v, list):
                feats.extend([float(x) if x is not None else 0.0 for x in v])
            else:
                # If missing, pad zeros (256)
                feats.extend([0.0] * 256)

        for k in scalar_feats:
            v = r.get(k, None)
            feats.append(0.0 if v is None else float(v))

        X_list.append(feats)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    N, D = X.shape
    print(f"[data] N={N}, D={D}")

    # Split
    tr_idx, va_idx, te_idx = train_val_test_split(N, args.val_frac, args.test_frac, seed=args.seed)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]
    Xte, yte = X[te_idx], y[te_idx]

    # Standardize
    mu, sd = stdz_fit(Xtr)
    Xtr_s = stdz_apply(Xtr, mu, sd)
    Xva_s = stdz_apply(Xva, mu, sd)
    Xte_s = stdz_apply(Xte, mu, sd)

    # Torch tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=D, hidden=args.hidden, hidden2=args.hidden2).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()  # Brier-aligned

    def run_epoch(Xb, yb, train=True):
        model.train(train)
        total_loss = 0.0
        bs = args.batch_size
        idx = np.arange(len(Xb))
        if train:
            np.random.shuffle(idx)
        for s in range(0, len(Xb), bs):
            j = idx[s:s+bs]
            xb = torch.tensor(Xb[j], device=device)
            ygt = torch.tensor(yb[j], device=device)
            with torch.set_grad_enabled(train):
                yhat = model(xb)
                loss = loss_fn(yhat, ygt)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += float(loss.item()) * len(j)
        return total_loss / max(1, len(Xb))

    def predict(Xb):
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(Xb, device=device)
            yhat = model(xb).clamp(0.0, 1.0).cpu().numpy()
        return yhat

    best_va = float("inf")
    best_state = None
    for ep in range(1, args.epochs + 1):
        tr_loss = run_epoch(Xtr_s, ytr, train=True)
        va_pred = predict(Xva_s)
        va_brier = brier_score(va_pred, yva)
        print(f"[ep {ep:02d}] train_brier={tr_loss:.4f} | val_brier={va_brier:.4f}")
        if va_brier < best_va:
            best_va = va_brier
            best_state = {
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "mu": mu,
                "sd": sd,
                "in_dim": D,
                "hidden": args.hidden,
                "hidden2": args.hidden2,
                "feature_spec": {
                    "vector_feats": vector_feats,
                    "scalar_feats": scalar_feats,
                    "packed_dim_each": 256,
                },
            }

    # Final eval on test
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    yte_hat = predict(Xte_s)
    te_brier = brier_score(yte_hat, yte)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(args.out_dir, "probe.pt"))
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "val_brier": float(best_va),
            "test_brier": float(te_brier),
            "N_train": int(len(Xtr)),
            "N_val": int(len(Xva)),
            "N_test": int(len(Xte)),
        }, f, indent=2)
    print(f"[probe] saved to {args.out_dir} | test_brier={te_brier:.4f}")

if __name__ == "__main__":
    main()
