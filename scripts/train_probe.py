# scripts/train_probe.py
# Multi-probe trainer with plots and artifacts.
# Transformer probe: per-token projections, token-type embeddings, CLS pooling,
# stochastic depth, attention dumps, and SAFE calibration with debug plots.

"""
    Usage:
    python -m scripts.train_probe \
        --data_dir data/probe_splits_80_10_10 \
        --models llama,qwen \
        --probe_type all \
        --use_hidden

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

# ------------------------ Data discovery ------------------------

def find_split_files(data_dir: Path, model_key: str):
    candidates = []
    for p in data_dir.glob("**/*.jsonl"):
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


def load_rows(jsonl_path: Path, show_progress=True):
    rows = []
    if not jsonl_path:
        return rows
    
    if show_progress:
        console.print(f"  [dim]Loading {jsonl_path.name}...[/dim]")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    
    if show_progress:
        console.print(f"  [green]✓[/green] Loaded {len(rows):,} rows from {jsonl_path.name}")
    
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

def build_df(rows, use_hidden=True, label_key="em", show_progress=True):
    """Build feature dataframe with progress indicator and memory efficiency"""
    n_rows = len(rows)
    
    if show_progress:
        console.print(f"  [dim]Building feature matrix for {n_rows:,} examples...[/dim]")
    
    # Pre-allocate for memory efficiency
    feats, labels, ids = [], [], []
    
    # Process in chunks to show progress
    chunk_size = 10000
    for chunk_start in range(0, n_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_rows)
        
        for i in range(chunk_start, chunk_end):
            r = rows[i]
            x = {}
            for k in SCALAR_KEYS:
                x[k] = r.get(k, np.nan)
            if use_hidden:
                for name in VECTOR_KEYS:
                    vec = to_1d(r.get(name))
                    if vec is not None:
                        for j, v in enumerate(vec):
                            x[f"{name}_{j}"] = float(v)
            feats.append(x)
            labels.append(int(r.get(label_key, 0)))
            ids.append(r.get("idx", i))
        
        if show_progress and (chunk_end % 50000 == 0 or chunk_end == n_rows):
            pct = 100 * chunk_end / n_rows
            console.print(f"    [green]→[/green] Processed {chunk_end:,}/{n_rows:,} ({pct:.0f}%)")
    
    if show_progress:
        console.print(f"  [dim]Converting to DataFrame...[/dim]")
    
    df = pd.DataFrame(feats)
    y = np.array(labels, dtype=int)
    ids = np.array(ids)
    
    if show_progress:
        console.print(f"  [green]✓[/green] Feature matrix ready: {df.shape}")
    
    return df, y, ids

# ------------------------ Classic probe builders ------------------------

def make_pipeline_mlp(use_hidden: bool, random_state: int):
    # With 332k examples and 64GB RAM limit, use SGD solver for memory efficiency
    # SGD uses mini-batches instead of loading all data at once
    base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 128) if use_hidden else (128, 64),  # 2 layers
            activation="relu",
            solver="sgd",  # SGD is memory-efficient (mini-batch)
            alpha=1e-4,
            batch_size=1024,  # Larger batches for efficiency
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            momentum=0.9,
            max_iter=50,  # Fewer epochs with SGD
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
            verbose=False,
        )),
    ])
    return base

def make_calibrated(estimator, y_train_len: int):
    method = "isotonic" if y_train_len >= 2000 else "sigmoid"
    return CalibratedClassifierCV(estimator, method=method, cv=3)

def make_pipeline_logreg():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
        ("clf",    LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            # class_weight="balanced"
        ))
    ])

def make_pipeline_logreg_cal(y_train_len: int):
    base = make_pipeline_logreg()
    return make_calibrated(base, y_train_len)

def make_pipeline_tree(max_depth=10, min_samples_leaf=20):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=7
        ))
    ])

# ------------------------ Transformer probe (PyTorch) ------------------------

import torch
import torch.nn as nn
import torch.utils.data as td

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.empty(x.shape[0], 1, 1, device=x.device).bernoulli_(keep)
        return x * mask / keep

class EncoderBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, d_ff=1024, p_drop=0.1, p_stoch=0.05):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=p_drop, batch_first=True)
        self.drop_path = DropPath(p_stoch)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p_drop),
        )

    def forward(self, x, need_attn=False):
        x2, attn = self.mha(x, x, x, need_weights=need_attn, average_attn_weights=False)
        x = self.ln1(x + self.drop_path(x2))
        x2 = self.ff(x)
        x = self.ln2(x + self.drop_path(x2))
        return x, attn  # attn: [B, H, T, T] if need_attn else None

class TinyTransformerProbe(nn.Module):
    def __init__(self,
                 n_tokens:int,   # dynamic!
                 d_model=256, nhead=8, num_layers=3, d_ff=1024,
                 p_drop=0.1, p_stoch=0.05):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens  # includes scalar token, excludes CLS

        # per-token projections
        self.hidden_projs = nn.ModuleList([nn.Sequential(nn.LayerNorm(256), nn.Linear(256, d_model))
                                           for _ in range(max(0, n_tokens-1))])
        self.scalar_proj = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, d_model))

        self.token_type = nn.Embedding(n_tokens, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_drop = nn.Dropout(p_drop)

        self.encoders = nn.ModuleList([
            EncoderBlock(d_model, nhead, d_ff, p_drop, p_stoch) for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens_256, token_mask, return_attn=False):
        """
        tokens_256: [B, T, D], where T = #present_hidden_tokens + 1 (scalar at end)
        token_mask: [T] bool tensor: True for hidden pack positions, False for scalar (last).
        """
        if token_mask.dim() != 1:
            token_mask = token_mask.view(-1)
        B, T, D = tokens_256.shape
        assert token_mask.numel() == T, "token_mask length must equal sequence length T"

        tok_list = []
        hid_idx = 0
        for t in range(T):
            if token_mask[t]:  # hidden pack
                tok_list.append(self.hidden_projs[hid_idx](tokens_256[:, t, :]))
                hid_idx += 1
            else:  # scalar (assumed last position)
                scal_raw = tokens_256[:, t, :13]
                tok_list.append(self.scalar_proj(scal_raw))
        x = torch.stack(tok_list, dim=1)  # [B,T,d_model]

        # add token-type embeddings (0..T-1)
        tt = torch.arange(T, device=x.device)
        x = x + self.token_type(tt)

        # prepend CLS
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)  # [B,T+1,D]
        x = self.pos_drop(x)

        attn_list = [] if return_attn else None
        for layer in self.encoders:
            x, attn = layer(x, need_attn=return_attn)
            if return_attn:
                attn_list.append(attn)
        pooled = x[:, 0, :]  # CLS
        p = self.head(pooled).squeeze(-1)
        return (p, attn_list) if return_attn else p

# Build transformer tokens from DataFrame (DYNAMIC token presence)
def build_transformer_tokens(X_df: pd.DataFrame, use_hidden: bool):
    """
    Returns: tokens [B,T,256], token_mask [T] (True if hidden pack, False if scalar),
             and token_labels list for debugging/attention.
    Include a token only if all its 256 cols exist; otherwise omit it.
    """
    B = len(X_df)
    tokens = []
    labels = []
    mask = []

    if use_hidden:
        for name in VECTOR_KEYS:
            cols = [f"{name}_{i}" for i in range(256)]
            if set(cols).issubset(X_df.columns):
                mat = X_df[cols].fillna(0.0).to_numpy(dtype=np.float32)
                tokens.append(mat)
                labels.append(name)
                mask.append(True)
            # else: skip silently

    # scalars as final token, pad to 256
    scalars = X_df[SCALAR_KEYS].fillna(0.0).to_numpy(dtype=np.float32)
    scal_tok = scalars
    if scal_tok.shape[1] < 256:
        scal_tok = np.pad(scal_tok, ((0,0),(0,256-scal_tok.shape[1])), 'constant')
    elif scal_tok.shape[1] > 256:
        scal_tok = scal_tok[:, :256]
    tokens.append(scal_tok.astype(np.float32))
    labels.append("scalars_256")
    mask.append(False)

    out = np.stack(tokens, axis=1)  # [B,T,256]
    return out, np.array(mask, dtype=bool), labels

def _reliability(ax, y_true, p, bins=10):
    edges = np.linspace(0, 1, bins+1)
    idx = np.digitize(p, edges) - 1
    idx = np.clip(idx, 0, bins-1)
    acc = []
    conf = []
    for b in range(bins):
        m = idx == b
        if m.any():
            acc.append(np.mean(y_true[m]))
            conf.append(np.mean(p[m]))
        else:
            acc.append(np.nan); conf.append((edges[b]+edges[b+1])/2)
    ax.plot(conf, acc, marker='o')
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel('Confidence'); ax.set_ylabel('Empirical accuracy'); ax.set_title('Reliability')

def train_transformer_probe(
    X_train_df, y_train,
    X_val_df=None, y_val=None,
    use_hidden=True, epochs=40, batch_size=256, lr=1e-3, weight_decay=1e-4,
    d_model=256, nhead=8, num_layers=3, d_ff=1024, p_drop=0.1, p_stoch=0.05,
    device=None, out_dir=None, calibrate=False, patience=5,
    min_iso_samples=2000, min_platt_samples=400, cal_debug=False
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tok, train_mask_np, train_labels = build_transformer_tokens(X_train_df, use_hidden)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    n_tokens = X_train_tok.shape[1]
    train_mask_t = torch.tensor(train_mask_np, dtype=torch.bool).to(device)

    if X_val_df is not None and y_val is not None:
        X_val_tok, val_mask_np, val_labels = build_transformer_tokens(X_val_df, use_hidden)
        val_mask_t = torch.tensor(val_mask_np, dtype=torch.bool).to(device)
    else:
        X_val_tok, val_mask_np, val_mask_t, val_labels = None, None, None, None

    # Sanity: masks between train/val must match if we calibrate
    if calibrate and (X_val_tok is not None):
        if not np.array_equal(train_mask_np, val_mask_np):
            print("[WARN] Token set differs between train and val; skipping calibration for safety.")
            calibrate = False

    model = TinyTransformerProbe(
        n_tokens=n_tokens, d_model=d_model, nhead=nhead, num_layers=num_layers,
        d_ff=d_ff, p_drop=p_drop, p_stoch=p_stoch
    ).to(device)

    # Datasets WITHOUT mask (mask is sequence-level, reused for all batches)
    train_ds = td.TensorDataset(torch.tensor(X_train_tok), y_train_t)
    train_ld = td.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    if X_val_tok is not None:
        val_ds = td.TensorDataset(torch.tensor(X_val_tok), torch.tensor(y_val, dtype=torch.float32))
        val_ld = td.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        val_ld = None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.MSELoss()

    best = {"brier": float("inf"), "state_dict": None, "patience": patience}
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            p = model(xb, train_mask_t)
            loss = loss_fn(p, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if val_ld is not None:
            model.eval()
            with torch.no_grad():
                preds, ys = [], []
                for xb, yb in val_ld:
                    xb = xb.to(device)
                    p = model(xb, val_mask_t).cpu().numpy()
                    preds.append(p); ys.append(yb.numpy())
                p = np.concatenate(preds); yv = np.concatenate(ys)
                brier = float(np.mean((p - yv) ** 2))
                if brier < best["brier"]:
                    best["brier"] = brier
                    best["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
                    best["patience"] = patience
                else:
                    best["patience"] -= 1
                    if best["patience"] <= 0:
                        break

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    model.eval()

    # ----- Calibration (safe) -----
    cal_blob = None
    if calibrate:
        # choose validation if available; else a slice of train
        if X_val_tok is not None:
            Xc, mc_t, yc = X_val_tok, val_mask_t, y_val
        else:
            n = min(4096, len(X_train_tok))
            Xc, mc_t, yc = X_train_tok[:n], train_mask_t, y_train[:n]

        with torch.no_grad():
            p_raw = model(torch.tensor(Xc).to(device), mc_t).cpu().numpy()

        n_c = len(p_raw)
        if n_c >= min_iso_samples:
            cal = IsotonicRegression(out_of_bounds="clip").fit(p_raw, yc)
            cal_type = "isotonic"
        elif n_c >= min_platt_samples:
            lr_cal = LogisticRegression().fit(p_raw.reshape(-1, 1), yc)
            cal = lr_cal
            cal_type = "platt"
        else:
            print(f"[WARN] Not enough samples for calibration (n={n_c}). Skipping.")
            cal = None
            cal_type = None

        if cal is not None:
            cal_blob = {"type": cal_type, "payload": pickle.dumps(cal)}

        # Optional debug plots
        if cal_debug:
            try:
                fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
                axs[0].hist(p_raw, bins=30); axs[0].set_title("Pre-calib prob hist"); axs[0].set_xlabel("p")
                if cal is not None:
                    if cal_type == "platt":
                        p_post = cal.predict_proba(p_raw.reshape(-1,1))[:,1]
                    else:
                        p_post = cal.transform(p_raw)
                else:
                    p_post = p_raw
                axs[1].hist(p_post, bins=30); axs[1].set_title("Post-calib prob hist"); axs[1].set_xlabel("p")
                _reliability(axs[2], yc, p_post, bins=10)
                fig.tight_layout()
                if out_dir is not None:
                    (out_dir / "calib_debug").mkdir(exist_ok=True)
                    fig.savefig(out_dir / "calib_debug" / "calibration_debug.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] Failed to save calibration debug plots: {e}")

    # Save artifact
    if out_dir is not None:
        save = {
            "type": "xform",
            "use_hidden": use_hidden,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "calibration": cal_blob,
            "arch": {
                "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
                "d_ff": d_ff, "p_drop": p_drop, "p_stoch": p_stoch
            },
            "token_labels": (train_labels if 'train_labels' in locals() else []),
            "packed_dim": 256,
            "train_mask": train_mask_np.tolist()
        }
        torch.save(save, out_dir / "probe_model.pt")
    return model, cal_blob, train_mask_np

def xform_predict_proba(model_obj, X_df, use_hidden=True, cal_blob=None, train_mask=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build tokens for this split
    X_tok, mask_np, _ = build_transformer_tokens(X_df, use_hidden)
    mask_t = torch.tensor(mask_np, dtype=torch.bool).to(device)

    # If a reference train_mask is provided, enforce same token order by intersection
    if train_mask is not None:
        train_mask_np = np.array(train_mask, dtype=bool)
        if not np.array_equal(train_mask_np, mask_np):
            tr_idx = np.where(train_mask_np)[0].tolist()
            te_idx = np.where(mask_np)[0].tolist()
            common = [i for i in tr_idx if i in te_idx]
            # include scalar token (last position) if present on both
            if (len(train_mask_np) > 0 and not train_mask_np[-1]) and (len(mask_np) > 0 and not mask_np[-1]):
                common += [X_tok.shape[1]-1]
            if len(common) < 1:
                raise RuntimeError("Token mismatch between train and inference masks; no common tokens.")
            X_tok = X_tok[:, common, :]
            mask_np = mask_np[common]
            mask_t = torch.tensor(mask_np, dtype=torch.bool).to(device)

    # Rebuild model from blob if dict provided
    if isinstance(model_obj, dict):
        arch = model_obj.get("arch", {})
        n_tokens = X_tok.shape[1]
        model = TinyTransformerProbe(
            n_tokens=n_tokens,
            d_model=arch.get("d_model", 256),
            nhead=arch.get("nhead", 8),
            num_layers=arch.get("num_layers", 3),
            d_ff=arch.get("d_ff", 1024),
            p_drop=arch.get("p_drop", 0.1),
            p_stoch=arch.get("p_stoch", 0.05),
        ).to(device)
        model.load_state_dict(model_obj["model_state_dict"])
        cal_blob = model_obj.get("calibration", cal_blob)
    else:
        model = model_obj.to(device)

    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X_tok).to(device), mask_t).cpu().numpy()

    # apply calibration if present
    if cal_blob is not None:
        cal = pickle.loads(cal_blob["payload"])
        if cal_blob["type"] == "platt":
            p = cal.predict_proba(p.reshape(-1, 1))[:, 1]
        else:
            p = cal.transform(p)
    return p

# ---- Attention dumping for Transformer probe ----

def dump_attention_maps(model_obj, X_df, use_hidden, out_dir: Path, max_batches=8, batch_size=256, train_mask=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tok, mask_np, labels = build_transformer_tokens(X_df, use_hidden)

    # Align to train_mask if provided
    if train_mask is not None:
        train_mask_np = np.array(train_mask, dtype=bool)
        if not np.array_equal(train_mask_np, mask_np):
            tr_idx = np.where(train_mask_np)[0].tolist()
            te_idx = np.where(mask_np)[0].tolist()
            common = [i for i in tr_idx if i in te_idx]
            if (len(train_mask_np) > 0 and not train_mask_np[-1]) and (len(mask_np) > 0 and not mask_np[-1]):
                common += [X_tok.shape[1]-1]
            if len(common) < 1:
                print("[WARN] Attention dump: token mismatch; skipping.")
                return
            X_tok = X_tok[:, common, :]
            mask_np = mask_np[common]
            labels = [labels[i] for i in common]

    mask_t = torch.tensor(mask_np, dtype=torch.bool).to(device)

    if isinstance(model_obj, dict):
        arch = model_obj.get("arch", {})
        model = TinyTransformerProbe(
            n_tokens=X_tok.shape[1],
            d_model=arch.get("d_model", 256),
            nhead=arch.get("nhead", 8),
            num_layers=arch.get("num_layers", 3),
            d_ff=arch.get("d_ff", 1024),
            p_drop=arch.get("p_drop", 0.1),
            p_stoch=arch.get("p_stoch", 0.05),
        ).to(device)
        model.load_state_dict(model_obj["model_state_dict"])
    else:
        model = model_obj.to(device)
    model.eval()

    ds = td.TensorDataset(torch.tensor(X_tok))
    ld = td.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    tok_labels = ["CLS"] + labels
    T = len(tok_labels)

    with torch.no_grad():
        layer_count = len(model.encoders)
        acc = [np.zeros((T, T), dtype=np.float64) for _ in range(layer_count)]
        batches_done = 0
        for (xb,) in ld:
            xb = xb.to(device)
            _, attn_list = model(xb, mask_t, return_attn=True)
            for li, attn in enumerate(attn_list):
                a = attn.mean(dim=1).mean(dim=0).detach().cpu().numpy()  # [T,T]
                acc[li] += a
            batches_done += 1
            if batches_done >= max_batches:
                break

        for li in range(layer_count):
            avg = acc[li] / max(1, batches_done)
            row_sums = avg.sum(axis=1, keepdims=True) + 1e-12
            avg_norm = avg / row_sums
            np.save(out_dir / f"attn_layer{li+1}.npy", avg_norm)
            pd.DataFrame(avg_norm, index=tok_labels, columns=tok_labels).to_csv(out_dir / f"attn_layer{li+1}.csv")

            plt.figure(figsize=(4.8, 3.8))
            plt.imshow(avg_norm, aspect="equal")
            plt.colorbar()
            plt.xticks(range(T), tok_labels, rotation=45, ha="right")
            plt.yticks(range(T), tok_labels)
            plt.title(f"Transformer Attention (Layer {li+1})")
            plt.tight_layout()
            plt.savefig(out_dir / f"attn_layer{li+1}.png", dpi=150)
            plt.close()

        with open(out_dir / "attn_summary.json", "w", encoding="utf-8") as f:
            json.dump({"tokens": tok_labels, "layers": [f"attn_layer{k+1}" for k in range(layer_count)]}, f, indent=2)

# ------------------------ Evaluation + plotting ------------------------

def evaluate_from_proba(proba, y, split_name):
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
    return out

def plot_curves(model_dir: Path, split_name: str, y, proba):
    # ROC
    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {split_name}")
    plt.tight_layout()
    plt.savefig(model_dir / f"roc_{split_name}.png", dpi=150)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y, proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {split_name}")
    plt.tight_layout()
    plt.savefig(model_dir / f"pr_{split_name}.png", dpi=150)
    plt.close()

# ------------------------ Train/eval orchestration ------------------------

def fit_and_eval_probe(probe_type: str, X_tr, y_tr, X_va, y_va, X_te, y_te,
                       use_hidden, out_root: Path,
                       random_state=7,
                       no_calibration=False,  # New parameter
                       # transformer-specific
                       xform_epochs=40, xform_dump_attn=False, xform_attn_split="val",
                       xform_attn_batches=8, xform_lr=1e-3, xform_wd=1e-4,
                       xform_heads=8, xform_layers=3, xform_ff=1024, xform_drop=0.1, xform_stoch=0.05,
                       xform_calibrate=False, xform_patience=5,
                       xform_min_iso=2000, xform_min_platt=400, xform_cal_debug=False):
    """
    Trains one probe type, writes artifacts under out_root / probe_<type>/
    """
    probe_dir = out_root / f"probe_{probe_type}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    if probe_type == "mlp":
        console.print(f"    [cyan]→ Training MLP (2-layer: 256→128, SGD solver)[/cyan]")
        base = make_pipeline_mlp(use_hidden, random_state)
        
        if no_calibration:
            console.print(f"    [yellow]⚠ Skipping calibration to save memory[/yellow]")
            model = base
        else:
            model = make_calibrated(base, len(y_tr))
        
        console.print(f"    [dim]Training on {len(y_tr):,} examples{'with isotonic calibration' if not no_calibration else ''}...[/dim]")
        model.fit(X_tr.values, y_tr)
        console.print(f"    [green]✓ MLP training completed[/green]")
        
        saver = lambda: joblib.dump(model, probe_dir / "probe_model.joblib")
        predict_proba = lambda X: model.predict_proba(X.values)[:, 1]

    elif probe_type == "logreg":
        console.print(f"    [cyan]→ Training Logistic Regression[/cyan]")
        model = make_pipeline_logreg()
        
        console.print(f"    [dim]Training on {len(y_tr):,} examples...[/dim]")
        model.fit(X_tr.values, y_tr)
        console.print(f"    [green]✓ LogReg training completed[/green]")
        
        saver = lambda: joblib.dump(model, probe_dir / "probe_model.joblib")
        predict_proba = lambda X: model.predict_proba(X.values)[:, 1]

    elif probe_type == "logreg_cal":
        model = make_pipeline_logreg_cal(len(y_tr))
        model.fit(X_tr.values, y_tr)
        saver = lambda: joblib.dump(model, probe_dir / "probe_model.joblib")
        predict_proba = lambda X: model.predict_proba(X.values)[:, 1]

    elif probe_type == "tree":
        model = make_pipeline_tree()
        model.fit(X_tr.values, y_tr)
        saver = lambda: joblib.dump(model, probe_dir / "probe_model.joblib")
        predict_proba = lambda X: model.predict_proba(X.values)[:, 1]

    elif probe_type == "xform":
        model, cal_blob, train_mask_np = train_transformer_probe(
            X_tr, y_tr, X_va, y_va,
            use_hidden=use_hidden,
            epochs=xform_epochs, lr=xform_lr, weight_decay=xform_wd,
            nhead=xform_heads, num_layers=xform_layers, d_ff=xform_ff,
            p_drop=xform_drop, p_stoch=xform_stoch,
            out_dir=probe_dir, calibrate=xform_calibrate, patience=xform_patience,
            min_iso_samples=xform_min_iso, min_platt_samples=xform_min_platt,
            cal_debug=xform_cal_debug
        )
        saver = lambda: torch.save({
            "type": "xform",
            "use_hidden": use_hidden,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "calibration": cal_blob,
            "arch": {"d_model": 256, "nhead": xform_heads, "num_layers": xform_layers,
                     "d_ff": xform_ff, "p_drop": xform_drop, "p_stoch": xform_stoch},
            "train_mask": train_mask_np.tolist()
        }, probe_dir / "probe_model.pt")
        predict_proba = lambda X: xform_predict_proba(
            model, X, use_hidden=use_hidden, cal_blob=cal_blob, train_mask=train_mask_np
        )

        # Optional: dump attention maps
        if xform_dump_attn:
            if (xform_attn_split == "val") and (X_va is not None):
                X_for_attn = X_va
            elif (xform_attn_split == "test") and (X_te is not None):
                X_for_attn = X_te
            else:
                X_for_attn = X_tr.head(min(4096, len(X_tr)))
            dump_attention_maps(model, X_for_attn, use_hidden, probe_dir,
                                max_batches=xform_attn_batches, train_mask=train_mask_np)

    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    # Save model + feature names
    saver()
    feat_names = list(X_tr.columns)
    with open(probe_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feat_names, f, ensure_ascii=False, indent=2)

    # Evaluate splits + write predictions, plots, report
    split_evals = []

    def do_eval(X, y, ids, name):
        if X is None:
            return
        proba = predict_proba(X)
        metrics = evaluate_from_proba(proba, y, name)
        split_evals.append(metrics)
        pd.DataFrame({"id": ids, "y_true": y, "p_right": proba}).to_csv(probe_dir / f"pred_{name}.csv", index=False)
        plot_curves(probe_dir, name, y, proba)

    do_eval(X_tr, y_tr, np.arange(len(y_tr)), "train")
    if X_va is not None:
        do_eval(X_va, y_va, np.arange(len(y_va)), "val")
    if X_te is not None:
        do_eval(X_te, y_te, np.arange(len(y_te)), "test")

    with open(probe_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(split_evals, f, indent=2)

    meta = {
        "probe_type": probe_type,
        "use_hidden": use_hidden,
        "feature_names_path": str(probe_dir / "feature_names.json"),
        "model_path": str(probe_dir / ("probe_model.pt" if probe_type == "xform" else "probe_model.joblib")),
        "pred_paths": {
            "train": str(probe_dir / "pred_train.csv"),
            "val":   str(probe_dir / "pred_val.csv") if (probe_dir / "pred_val.csv").exists() else None,
            "test":  str(probe_dir / "pred_test.csv") if (probe_dir / "pred_test.csv").exists() else None,
        },
        "plots": {
            "train": {"roc": str(probe_dir / "roc_train.png"), "pr": str(probe_dir / "pr_train.png")},
            "val":   {"roc": str(probe_dir / "roc_val.png") if (probe_dir / "roc_val.png").exists() else None,
                      "pr":  str(probe_dir / "pr_val.png") if (probe_dir / "pr_val.png").exists() else None},
            "test":  {"roc": str(probe_dir / "roc_test.png") if (probe_dir / "roc_test.png").exists() else None,
                      "pr":  str(probe_dir / "pr_test.png") if (probe_dir / "pr_test.png").exists() else None},
        }
    }
    with open(probe_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"probe_dir": str(probe_dir), "splits": split_evals, "meta": meta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/probe_splits_80_10_10")
    parser.add_argument("--models", type=str, default="llama,qwen", help="comma-separated: llama,qwen")
    parser.add_argument("--use_hidden", action="store_true", help="include 4x256 hidden vectors")
    parser.add_argument("--label_key", type=str, default="em")
    parser.add_argument("--out_dir", type=str, default=".", help="save dir (default: <data_dir>/probe_models)")
    parser.add_argument("--probe_type", type=str, default="mlp",
                        choices=["mlp", "logreg", "logreg_cal", "tree", "xform", "all"])
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit training examples (for testing with large datasets)")
    parser.add_argument("--no_calibration", action="store_true",
                        help="Skip isotonic calibration to save memory (for large datasets)")



    # Transformer hyperparams / behavior
    parser.add_argument("--xform_epochs", type=int, default=40)
    parser.add_argument("--xform_lr", type=float, default=1e-3)
    parser.add_argument("--xform_wd", type=float, default=1e-4)
    parser.add_argument("--xform_heads", type=int, default=8)
    parser.add_argument("--xform_layers", type=int, default=3)
    parser.add_argument("--xform_ff", type=int, default=1024)
    parser.add_argument("--xform_drop", type=float, default=0.1)
    parser.add_argument("--xform_stoch", type=float, default=0.05)
    parser.add_argument("--xform_calibrate", action="store_true")
    parser.add_argument("--xform_patience", type=int, default=5)
    parser.add_argument("--xform_min_iso", type=int, default=2000,
                        help="min samples to use isotonic calibration; else try Platt; else skip")
    parser.add_argument("--xform_min_platt", type=int, default=400,
                        help="min samples to use Platt calibration; else skip")
    parser.add_argument("--xform_cal_debug", action="store_true",
                        help="save calibration histograms + reliability diagram")

    # Attention export toggles
    parser.add_argument("--xform_dump_attn", action="store_true",
                        help="for Transformer probe: dump averaged attention maps")
    parser.add_argument("--xform_attn_split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="which split to use for averaging attention (default: val; falls back if missing)")
    parser.add_argument("--xform_attn_batches", type=int, default=8,
                        help="number of minibatches to average for attention maps")

    args = parser.parse_args()

    # Print startup banner
    console.print(Panel.fit(
        "[bold cyan]Probe Training Pipeline[/bold cyan]\n"
        f"Data: {args.data_dir}\n"
        f"Models: {args.models}\n"
        f"Probe types: {args.probe_type}\n"
        f"Use hidden states: {args.use_hidden}\n"
        f"Output: {args.out_dir or 'data/probe_models'}",
        title="🔬 Train Probes",
        border_style="cyan"
    ))

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "probe_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    for mk_idx, mk in enumerate(model_keys, 1):
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold yellow]Model {mk_idx}/{len(model_keys)}: backend_{mk}[/bold yellow]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")
        
        split_files = find_split_files(data_dir, mk)
        if not split_files:
            console.print(f"[red]⚠ No files found for backend_{mk} under {data_dir}[/red]")
            continue
        
        # Show split files
        file_table = Table(show_header=True, header_style="bold cyan")
        file_table.add_column("Split", style="cyan")
        file_table.add_column("File", style="dim")
        for k, v in split_files.items():
            file_table.add_row(k, str(v.relative_to(data_dir)))
        console.print(file_table)

        console.print(f"\n[cyan]📚 Loading data splits...[/cyan]")
        rows_train = load_rows(split_files.get("train"))
        rows_val   = load_rows(split_files.get("val"))
        rows_test  = load_rows(split_files.get("test"))
        
        # Apply limit if specified (for memory management with large datasets)
        if args.limit and rows_train:
            original_size = len(rows_train)
            rows_train = rows_train[:args.limit]
            console.print(f"  [yellow]⚠ Limited train set: {len(rows_train):,} / {original_size:,} examples[/yellow]")

        if not rows_train:
            console.print(f"[red]⚠ Empty train split for backend_{mk}; skipping[/red]")
            continue

        console.print(f"\n[cyan]🔧 Building feature matrices...[/cyan]")
        df_tr, y_tr, _ = build_df(rows_train, use_hidden=args.use_hidden, label_key=args.label_key)
        df_va, y_va, _ = (None, None, None)
        df_te, y_te, _ = (None, None, None)

        if rows_val:
            df_va, y_va, _ = build_df(rows_val, use_hidden=args.use_hidden, label_key=args.label_key)
        if rows_test:
            df_te, y_te, _ = build_df(rows_test, use_hidden=args.use_hidden, label_key=args.label_key)

        # Ensure consistent feature set across splits
        feat_cols = list(df_tr.columns)
        if df_va is not None:
            df_va = df_va.reindex(columns=feat_cols)
        if df_te is not None:
            df_te = df_te.reindex(columns=feat_cols)

        # Show dataset statistics
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Split", style="cyan")
        stats_table.add_column("Rows", justify="right", style="green")
        stats_table.add_column("Features", justify="right", style="yellow")
        stats_table.add_column("Positive %", justify="right", style="blue")
        
        stats_table.add_row("train", f"{len(y_tr):,}", str(len(feat_cols)), f"{100*y_tr.mean():.1f}%")
        if y_va is not None:
            stats_table.add_row("val", f"{len(y_va):,}", str(len(feat_cols)), f"{100*y_va.mean():.1f}%")
        if y_te is not None:
            stats_table.add_row("test", f"{len(y_te):,}", str(len(feat_cols)), f"{100*y_te.mean():.1f}%")
        
        console.print(stats_table)

        backend_dir = out_dir / f"backend_{mk}"
        backend_dir.mkdir(parents=True, exist_ok=True)
        with open(backend_dir / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(feat_cols, f, ensure_ascii=False, indent=2)

        probe_types = ["mlp", "logreg", "logreg_cal", "tree", "xform"] if args.probe_type == "all" else [args.probe_type]
        
        console.print(f"\n[bold cyan]🔬 Training {len(probe_types)} probe type(s)...[/bold cyan]")
        
        for pt_idx, pt in enumerate(probe_types, 1):
            console.print(f"\n[yellow]{'─'*60}[/yellow]")
            console.print(f"[yellow]Probe {pt_idx}/{len(probe_types)}: {pt}[/yellow]")
            console.print(f"[yellow]{'─'*60}[/yellow]")
            
            result = fit_and_eval_probe(
                pt, df_tr, y_tr, df_va, y_va, df_te, y_te,
                use_hidden=args.use_hidden,
                out_root=backend_dir,
                no_calibration=args.no_calibration,  # Pass memory-saving flag
                # xform knobs
                xform_epochs=args.xform_epochs,
                xform_dump_attn=args.xform_dump_attn,
                xform_attn_split=args.xform_attn_split,
                xform_attn_batches=args.xform_attn_batches,
                xform_lr=args.xform_lr,
                xform_wd=args.xform_wd,
                xform_heads=args.xform_heads,
                xform_layers=args.xform_layers,
                xform_ff=args.xform_ff,
                xform_drop=args.xform_drop,
                xform_stoch=args.xform_stoch,
                xform_calibrate=args.xform_calibrate,
                xform_patience=args.xform_patience,
                xform_min_iso=args.xform_min_iso,
                xform_min_platt=args.xform_min_platt,
                xform_cal_debug=args.xform_cal_debug
            )
            
            # Show results in a nice table
            results_table = Table(show_header=True, header_style="bold green")
            results_table.add_column("Split", style="cyan")
            results_table.add_column("AUROC", justify="right", style="green")
            results_table.add_column("AUPRC", justify="right", style="yellow")
            results_table.add_column("Brier", justify="right", style="blue")
            results_table.add_column("Best F1", justify="right", style="magenta")
            results_table.add_column("Threshold", justify="right", style="dim")
            
            for r in result["splits"]:
                results_table.add_row(
                    r['split'],
                    f"{r['AUROC']:.3f}",
                    f"{r['AUPRC']:.3f}",
                    f"{r['Brier']:.3f}",
                    f"{r['best_F1']:.3f}",
                    f"{r['best_threshold']:.3f}"
                )
            
            console.print(results_table)
            console.print(f"[green]✓ Saved to {result['probe_dir']}[/green]\n")
    
    # Final completion message
    console.print(Panel.fit(
        "[bold green]✓ All probes trained successfully![/bold green]\n\n"
        f"Output directory: {out_dir}\n"
        f"Models trained: {', '.join(model_keys)}\n"
        f"Probe types: {args.probe_type}",
        title="🎉 Training Complete",
        border_style="green"
    ))

if __name__ == "__main__":
    main()