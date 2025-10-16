# train_probe.py
# Extended: multiple probe types (+plots), unified artifacts for easy eval.
# Transformer probe upgraded with per-token projections, token-type embeddings, CLS pooling,
# stochastic depth, optional calibration, and attention-map dumping.

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

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


def load_rows(jsonl_path: Path):
    rows = []
    if not jsonl_path:
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
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

# ------------------------ Classic probe builders ------------------------

def make_pipeline_mlp(use_hidden: bool, random_state: int):
    base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 64) if use_hidden else (64, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=60,
            early_stopping=True,
            n_iter_no_change=5,
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
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000
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
                 use_hidden=True,
                 d_model=256, nhead=8, num_layers=3, d_ff=1024,
                 p_drop=0.1, p_stoch=0.05):
        super().__init__()
        self.use_hidden = use_hidden
        self.d_model = d_model

        # Per-token projections (put hidden/scalars on same footing)
        if use_hidden:
            self.proj_last     = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, d_model))
            self.proj_pool     = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, d_model))
            self.proj_last_mid = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, d_model))
            self.proj_pool_mid = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, d_model))
            n_tokens = 5
        else:
            n_tokens = 1

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

    def _project_tokens(self, tokens_256):
        """
        tokens_256 order (when use_hidden=True):
        [h_last_256, h_pool_256, h_last_mid_256, h_pool_mid_256, scalars_256(padded)]
        else: [scalars_256(padded)]
        """
        B, T, D = tokens_256.shape
        tok_list = []
        idx = 0
        if self.use_hidden:
            tok_list.append(self.proj_last(tokens_256[:, idx, :]));     idx += 1
            tok_list.append(self.proj_pool(tokens_256[:, idx, :]));     idx += 1
            tok_list.append(self.proj_last_mid(tokens_256[:, idx, :])); idx += 1
            tok_list.append(self.proj_pool_mid(tokens_256[:, idx, :])); idx += 1
        scal_raw = tokens_256[:, -1, :13]
        tok_list.append(self.scalar_proj(scal_raw))
        x = torch.stack(tok_list, dim=1)  # [B,T',d_model]
        return x

    def forward(self, tokens_256, return_attn=False):
        x = self._project_tokens(tokens_256)            # [B,T',D]
        B, T, D = x.shape
        tt = torch.arange(T, device=x.device)
        x = x + self.token_type(tt)

        # prepend CLS
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)                 # [B,T'+1,D]
        x = self.pos_drop(x)

        attn_list = [] if return_attn else None
        for layer in self.encoders:
            x, attn = layer(x, need_attn=return_attn)
            if return_attn:
                attn_list.append(attn)  # [B,H,T+1,T+1]
        pooled = x[:, 0, :]                             # CLS
        p = self.head(pooled).squeeze(-1)
        return (p, attn_list) if return_attn else p

# Build transformer tokens from DataFrame
def build_transformer_tokens(X_df: pd.DataFrame, use_hidden: bool):
    scalars = X_df[SCALAR_KEYS].fillna(0.0).to_numpy(dtype=np.float32)
    tokens = []

    if use_hidden:
        for name in VECTOR_KEYS:
            cols = [f"{name}_{i}" for i in range(256)]
            if not set(cols).issubset(set(X_df.columns)):
                mat = np.zeros((len(X_df), 256), dtype=np.float32)
            else:
                mat = X_df[cols].fillna(0.0).to_numpy(dtype=np.float32)
            tokens.append(mat.astype(np.float32))

    # scalar token: pad to 256 for uniform shape (projection happens in the model)
    pad_to = 256
    scal_tok = scalars
    if scal_tok.shape[1] < pad_to:
        scal_tok = np.pad(scal_tok, ((0, 0), (0, pad_to - scal_tok.shape[1])), 'constant')
    elif scal_tok.shape[1] > pad_to:
        scal_tok = scal_tok[:, :pad_to]
    tokens.append(scal_tok.astype(np.float32))

    out = np.stack(tokens, axis=1)  # [B,T,D]
    return out

def train_transformer_probe(
    X_train_df, y_train,
    X_val_df=None, y_val=None,
    use_hidden=True, epochs=40, batch_size=256, lr=1e-3, weight_decay=1e-4,
    d_model=256, nhead=8, num_layers=3, d_ff=1024, p_drop=0.1, p_stoch=0.05,
    device=None, out_dir=None, calibrate=True, patience=5
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tok = build_transformer_tokens(X_train_df, use_hidden)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    if X_val_df is not None and y_val is not None:
        X_val_tok = build_transformer_tokens(X_val_df, use_hidden)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
    else:
        X_val_tok, y_val_t = None, None

    model = TinyTransformerProbe(
        use_hidden=use_hidden, d_model=d_model, nhead=nhead, num_layers=num_layers,
        d_ff=d_ff, p_drop=p_drop, p_stoch=p_stoch
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.MSELoss()

    train_ds = td.TensorDataset(torch.tensor(X_train_tok), y_train_t)
    train_ld = td.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    if X_val_tok is not None:
        val_ds = td.TensorDataset(torch.tensor(X_val_tok), y_val_t)
        val_ld = td.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        val_ld = None

    best = {"brier": float("inf"), "state_dict": None, "patience": patience}
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            p = model(xb)
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
                    p = model(xb).cpu().numpy()
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

    # Optional calibration (isotonic if enough val; fallback to Platt)
    cal_blob = None
    if calibrate:
        # choose validation if available; else small slice of train
        if X_val_tok is not None:
            Xc, yc = X_val_tok, y_val
        else:
            n = min(4096, len(X_train_tok))
            Xc = X_train_tok[:n]
            yc = y_train[:n]
        with torch.no_grad():
            p_raw = model(torch.tensor(Xc).to(device)).cpu().numpy()
        if len(p_raw) >= 2000:
            cal = IsotonicRegression(out_of_bounds="clip").fit(p_raw, yc)
            cal_type = "isotonic"
            # store fitted y_ boundaries too
        else:
            lr_cal = LogisticRegression().fit(p_raw.reshape(-1, 1), yc)
            cal = lr_cal
            cal_type = "platt"
        cal_blob = {"type": cal_type, "payload": joblib.dumps(cal)}

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
            "scalar_keys": SCALAR_KEYS,
            "vector_keys": VECTOR_KEYS if use_hidden else [],
            "packed_dim": 256
        }
        torch.save(save, out_dir / "probe_model.pt")
    return model, cal_blob

def xform_predict_proba(model_obj, X_df, use_hidden=True, cal_blob=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(model_obj, dict):
        arch = model_obj.get("arch", {})
        model = TinyTransformerProbe(
            use_hidden=use_hidden,
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

    X_tok = build_transformer_tokens(X_df, use_hidden)
    with torch.no_grad():
        p = model(torch.tensor(X_tok).to(device)).cpu().numpy()

    # apply calibration if present
    if cal_blob is not None:
        cal = joblib.loads(cal_blob["payload"])
        if cal_blob["type"] == "platt":
            p = cal.predict_proba(p.reshape(-1, 1))[:, 1]
        else:
            p = cal.transform(p)
    return p

# ---- Attention dumping for Transformer probe ----

def dump_attention_maps(model_obj, X_df, use_hidden, out_dir: Path, max_batches=8, batch_size=256):
    """
    Computes averaged attention matrices per encoder layer across a subset
    of the provided split. Saves:
      - attn_layer{k}.npy  (T x T averaged)
      - attn_layer{k}.csv  (token x token)
      - attn_layer{k}.png  (heatmap)
      - attn_summary.json  (token labels + notes)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(model_obj, dict):
        arch = model_obj.get("arch", {})
        model = TinyTransformerProbe(
            use_hidden=use_hidden,
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

    X_tok = build_transformer_tokens(X_df, use_hidden)
    ds = td.TensorDataset(torch.tensor(X_tok))
    ld = td.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    tok_labels = (["CLS"] +
                  (VECTOR_KEYS if use_hidden else []) +
                  ["scalars_256"])
    T = len(tok_labels)

    with torch.no_grad():
        layer_count = len(model.encoders)
        acc = [np.zeros((T, T), dtype=np.float64) for _ in range(layer_count)]
        batches_done = 0
        for (xb,) in ld:
            xb = xb.to(device)
            # forward with attention
            # model.forward handles projections, token-type emb, and CLS
            _, attn_list = model(xb, return_attn=True)  # list of [B,H,T,T]
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

            plt.figure(figsize=(4.6, 3.6))
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
                       # transformer-specific
                       xform_epochs=40, xform_dump_attn=False, xform_attn_split="val",
                       xform_attn_batches=8, xform_lr=1e-3, xform_wd=1e-4,
                       xform_heads=8, xform_layers=3, xform_ff=1024, xform_drop=0.1, xform_stoch=0.05,
                       xform_calibrate=True, xform_patience=5):
    """
    Trains one probe type, writes artifacts under out_root / probe_<type>/
    """
    probe_dir = out_root / f"probe_{probe_type}"
    probe_dir.mkdir(parents=True, exist_ok=True)

    if probe_type == "mlp":
        base = make_pipeline_mlp(use_hidden, random_state)
        model = make_calibrated(base, len(y_tr))
        model.fit(X_tr.values, y_tr)
        saver = lambda: joblib.dump(model, probe_dir / "probe_model.joblib")
        predict_proba = lambda X: model.predict_proba(X.values)[:, 1]

    elif probe_type == "logreg":
        model = make_pipeline_logreg()
        model.fit(X_tr.values, y_tr)
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
        model, cal_blob = train_transformer_probe(
            X_tr, y_tr, X_va, y_va,
            use_hidden=use_hidden,
            epochs=xform_epochs, lr=xform_lr, weight_decay=xform_wd,
            nhead=xform_heads, num_layers=xform_layers, d_ff=xform_ff,
            p_drop=xform_drop, p_stoch=xform_stoch,
            out_dir=probe_dir, calibrate=xform_calibrate, patience=xform_patience
        )
        saver = lambda: torch.save({
            "type": "xform",
            "use_hidden": use_hidden,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "calibration": cal_blob,
            "arch": {"d_model": 256, "nhead": xform_heads, "num_layers": xform_layers,
                     "d_ff": xform_ff, "p_drop": xform_drop, "p_stoch": xform_stoch},
            "scalar_keys": SCALAR_KEYS,
            "vector_keys": VECTOR_KEYS if use_hidden else [],
            "packed_dim": 256
        }, probe_dir / "probe_model.pt")
        predict_proba = lambda X: xform_predict_proba(model, X, use_hidden=use_hidden, cal_blob=cal_blob)

        # Optional: dump attention maps
        if xform_dump_attn:
            if (xform_attn_split == "val") and (X_va is not None):
                X_for_attn = X_va
            elif (xform_attn_split == "test") and (X_te is not None):
                X_for_attn = X_te
            else:
                X_for_attn = X_tr.head(min(4096, len(X_tr)))
            dump_attention_maps(model, X_for_attn, use_hidden, probe_dir,
                                max_batches=xform_attn_batches)

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
    parser.add_argument("--out_dir", type=str, default=None, help="save dir (default: <data_dir>/probe_models)")
    parser.add_argument("--probe_type", type=str, default="mlp",
                        choices=["mlp", "logreg", "logreg_cal", "tree", "xform", "all"])

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

    # Attention export toggles
    parser.add_argument("--xform_dump_attn", action="store_true",
                        help="for Transformer probe: dump averaged attention maps")
    parser.add_argument("--xform_attn_split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="which split to use for averaging attention (default: val; falls back if missing)")
    parser.add_argument("--xform_attn_batches", type=int, default=8,
                        help="number of minibatches to average for attention maps")
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

        rows_train = load_rows(split_files.get("train"))
        rows_val   = load_rows(split_files.get("val"))
        rows_test  = load_rows(split_files.get("test"))

        if not rows_train:
            print(f"[WARN] backend_{mk}: missing or empty train split; skipping")
            continue

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

        backend_dir = out_dir / f"backend_{mk}"
        backend_dir.mkdir(parents=True, exist_ok=True)
        with open(backend_dir / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(feat_cols, f, ensure_ascii=False, indent=2)

        probe_types = ["mlp", "logreg", "logreg_cal", "tree", "xform"] if args.probe_type == "all" else [args.probe_type]
        for pt in probe_types:
            print(f"\n[train] backend_{mk} | probe_type={pt} | use_hidden={args.use_hidden}")
            result = fit_and_eval_probe(
                pt, df_tr, y_tr, df_va, y_va, df_te, y_te,
                use_hidden=args.use_hidden,
                out_root=backend_dir,
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
            )
            print(f"[OK] Saved {pt} to {result['probe_dir']}")
            for r in result["splits"]:
                print(f"  [{r['split']}] AUROC={r['AUROC']:.3f} AUPRC={r['AUPRC']:.3f} "
                      f"Brier={r['Brier']:.3f} bestF1={r['best_F1']:.3f} thr={r['best_threshold']:.3f}")

if __name__ == "__main__":
    main()
