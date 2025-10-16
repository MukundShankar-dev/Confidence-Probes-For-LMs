# scripts/demo.py
# Interactive demo: ask a trivia question, run an LM, parse self-reported confidence,
# compute probe features, and score P(correct) with a trained probe.
#
# Usage:
#   python -m scripts.demo --model llama31 --probe_dir probes/backend_llama/probe_mlp --use_hidden
#   python -m scripts.demo --model qwen    --probe_dir probes/backend_qwen/probe_xform --use_hidden
#
# Notes:
# - Pass the *exact* directory for a single trained probe (e.g., probes/backend_llama/probe_mlp).
#   The script will auto-detect whether it's a sklearn/joblib probe or a Transformer (.pt) probe.
# - It will also read feature_names.json in that directory to align feature columns.
# - If available, it will import your local `scripts.collect_internals` module to compute the
#   same statistics/features you used for training. If not available, it falls back to a built-in
#   minimal featurizer (works but may not match training exactly).

import argparse
import json
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

# ---------------- LM loading ----------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- Probes ----------------
import joblib
import pickle

# ---------------- Pretty print ----------------
from textwrap import indent

# ===========================
# Prompt & parsing utilities
# ===========================

FEWSHOT_FIXED_ANSWERABLE = """You are answering trivia questions.
Return only a single JSON object with fields:
- "answer": the short factual span (1–5 words, no punctuation). Do NOT output "Unknown" for this dataset; instead provide your best answer span.
- "p_true": the probability (0.0–1.0) that this answer is correct (round to two decimals).

Calibration guidance:
- Report your true probability; do NOT inflate.
- If you are unsure, still provide your best span answer and set a lower p_true (e.g., 0.15–0.35).
- If multiple plausible answers exist or the question is ambiguous, reduce p_true appropriately.

Return exactly ONE JSON object and NOTHING else.

Q: Who wrote Hamlet?
{"answer": "William Shakespeare", "p_true": 0.95}

Q: What is the capital of France?
{"answer": "Paris", "p_true": 0.92}

Q: What is the capital of South Africa?
{"answer": "Pretoria", "p_true": 0.60}

Q: Which element has the symbol 'Au'?
{"answer": "Aluminum", "p_true": 0.15}

Q: Which planet is known as the Red Planet?
{"answer": "Mars", "p_true": 0.85}

Q: {EVAL_QUESTION}
"""

# Strict-ish JSON grab: take first {...} block that parses
FIRST_JSON_RE = re.compile(r"\{[^{}]*\}")

def extract_first_json(text: str):
    """
    Extract and parse the first JSON object in the text. Returns (obj_or_none, raw_fragment_or_none)
    """
    for m in FIRST_JSON_RE.finditer(text):
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            return obj, frag
        except Exception:
            continue
    return None, None

def normalize_answer_span(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\s'-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

# ===========================
# LM backends
# ===========================

LLAMA_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_ID  = "Qwen/Qwen2.5-7B-Instruct"

def load_lm(which: str, device: str = None):
    if which.lower() in ["llama31", "llama", "backend_llama"]:
        model_id = LLAMA_ID
    elif which.lower() in ["qwen", "backend_qwen"]:
        model_id = QWEN_ID
    else:
        raise ValueError(f"Unknown --model {which}. Use llama31 or qwen.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LM] Loading {model_id} on {device} ...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    return tok, model, model_id

# ===========================
# Feature computation
# ===========================

# We’ll try to import your training-time featurizer.
COLLECT = None
try:
    # Expecting something like scripts/collect_internals.py
    from scripts.collect_internals import featurize_for_probe  # user-provided function (if exists)
    COLLECT = "featurize_for_probe"
except Exception:
    pass

# Fallback feature keys (must match training)
SCALAR_KEYS = [
    "model_confidence", "lp_mean", "seq_conf",
    "entropy_mean", "entropy_std",
    "margin_mean", "margin_min",
    "rescore_logp", "answer_len",
    "parsed_json_ok", "parsed_p_true_ok", "is_unknown",
]
VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]  # optional

@dataclass
class DemoFeatures:
    scalars: dict
    vectors: dict  # name -> np.ndarray shape [256]

def safe_softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def basic_featurizer(tokenizer, model, prompt_ids, gen_ids, logits_per_step) -> DemoFeatures:
    """
    Very lightweight fallback featurizer.
    - Uses logits to approximate per-token entropy, margins, etc.
    - Does NOT compute real hidden-state packs; fills them with zeros.
    - Matches scalar key names used in training (best effort).
    """
    # Compute per-step stats for the generated portion
    # logits_per_step: list of np.array [Vocab]
    entropies = []
    margins = []
    lp_tokens = []
    for t, (logit) in enumerate(logits_per_step):
        prob = safe_softmax(logit.astype(np.float64))
        ent = -np.sum(prob * np.log(prob + 1e-12))
        entropies.append(ent)
        sortp = np.sort(prob)
        margin = sortp[-1] - sortp[-2] if len(sortp) >= 2 else sortp[-1]
        margins.append(margin)

        tok_id = gen_ids[t]
        lp_tokens.append(np.log(prob[tok_id] + 1e-12))

    # Scalar aggregates
    entropy_mean = float(np.mean(entropies)) if entropies else 0.0
    entropy_std  = float(np.std(entropies)) if entropies else 0.0
    margin_mean  = float(np.mean(margins)) if margins else 0.0
    margin_min   = float(np.min(margins)) if margins else 0.0
    lp_mean      = float(np.mean(lp_tokens)) if lp_tokens else 0.0
    seq_conf     = float(np.exp(np.sum(lp_tokens))) if lp_tokens else 0.0  # uncalibrated pseudo-conf
    rescore_logp = float(np.sum(lp_tokens)) if lp_tokens else 0.0

    # "model_confidence" here is a placeholder; we’ll set it to parsed p_true if present
    scalars = {
        "model_confidence": 0.0,  # will set from parsed JSON if available
        "lp_mean": lp_mean,
        "seq_conf": seq_conf,
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "margin_mean": margin_mean,
        "margin_min": margin_min,
        "rescore_logp": rescore_logp,
        "answer_len": 0,  # will set after parsing
        "parsed_json_ok": 0,
        "parsed_p_true_ok": 0,
        "is_unknown": 0,
    }

    # Hidden vectors (optional): zeros to keep shape consistent if probe expects them
    vectors = {name: np.zeros(256, dtype=np.float32) for name in VECTOR_KEYS}

    return DemoFeatures(scalars=scalars, vectors=vectors)

def to_vec256(x: np.ndarray):
    """Trim/pad to 256."""
    x = np.asarray(x).astype(np.float32).ravel()
    if x.size < 256:
        x = np.pad(x, (0, 256 - x.size))
    elif x.size > 256:
        x = x[:256]
    return x

# ===========================
# Transformer probe (PyTorch)
# ===========================

class DropPath(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.empty(x.shape[0], 1, 1, device=x.device).bernoulli_(keep)
        return x * mask / keep

class EncoderBlock(torch.nn.Module):
    def __init__(self, d_model=256, nhead=8, d_ff=1024, p_drop=0.1, p_stoch=0.05):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(d_model, nhead, dropout=p_drop, batch_first=True)
        self.drop_path = DropPath(p_stoch)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(p_drop),
        )

    def forward(self, x, need_attn=False):
        x2, attn = self.mha(x, x, x, need_weights=need_attn, average_attn_weights=False)
        x = self.ln1(x + self.drop_path(x2))
        x2 = self.ff(x)
        x = self.ln2(x + self.drop_path(x2))
        return x, attn

class TinyTransformerProbe(torch.nn.Module):
    def __init__(self, n_tokens:int, d_model=256, nhead=8, num_layers=3, d_ff=1024, p_drop=0.1, p_stoch=0.05):
        super().__init__()
        self.hidden_projs = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.LayerNorm(256), torch.nn.Linear(256, d_model)
        ) for _ in range(max(0, n_tokens-1))])
        self.scalar_proj = torch.nn.Sequential(torch.nn.LayerNorm(13), torch.nn.Linear(13, d_model))
        self.token_type = torch.nn.Embedding(n_tokens, d_model)
        self.cls = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_drop = torch.nn.Dropout(p_drop)
        self.enc = torch.nn.ModuleList([EncoderBlock(d_model, nhead, d_ff, p_drop, p_stoch) for _ in range(num_layers)])
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(p_drop),
            torch.nn.Linear(d_model // 2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, tokens_256, token_mask):
        if token_mask.dim() != 1:
            token_mask = token_mask.view(-1)
        B, T, D = tokens_256.shape
        assert token_mask.numel() == T
        tok_list = []
        hid_idx = 0
        for t in range(T):
            if token_mask[t]:
                tok_list.append(self.hidden_projs[hid_idx](tokens_256[:, t, :]))
                hid_idx += 1
            else:
                scal_raw = tokens_256[:, t, :13]
                tok_list.append(self.scalar_proj(scal_raw))
        x = torch.stack(tok_list, dim=1)
        # token type + CLS
        tt = torch.arange(x.shape[1], device=x.device)
        x = x + self.token_type(tt)
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x)
        for layer in self.enc:
            x, _ = layer(x)
        pooled = x[:, 0, :]
        p = self.head(pooled).squeeze(-1)
        return p

def build_transformer_tokens_from_df(X_df: pd.DataFrame, use_hidden: bool):
    """
    Returns tokens [B,T,256], mask [T] (True: hidden pack, False: scalars), labels list.
    Includes hidden tokens only if all 256 columns exist.
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
                tokens.append(mat); labels.append(name); mask.append(True)
    # scalars as final token (padded)
    scal = X_df[SCALAR_KEYS].fillna(0.0).to_numpy(dtype=np.float32)
    if scal.shape[1] < 256:
        scal = np.pad(scal, ((0,0),(0,256-scal.shape[1])), 'constant')
    elif scal.shape[1] > 256:
        scal = scal[:, :256]
    tokens.append(scal.astype(np.float32)); labels.append("scalars_256"); mask.append(False)
    out = np.stack(tokens, axis=1)
    return out, np.array(mask, dtype=bool), labels

def xform_predict_from_blob(blob: dict, X_df: pd.DataFrame, use_hidden: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_tok, mask_np, _ = build_transformer_tokens_from_df(X_df, use_hidden)
    mask_t = torch.tensor(mask_np, dtype=torch.bool).to(device)

    arch = blob.get("arch", {})
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
    state = blob["model_state_dict"]
    # state may be Tensors; ensure on CPU
    state = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        p = model(torch.tensor(X_tok).to(device), mask_t).cpu().numpy()

    cal = blob.get("calibration")
    if cal is not None:
        cal_obj = pickle.loads(cal["payload"])
        if cal["type"] == "platt":
            p = cal_obj.predict_proba(p.reshape(-1,1))[:,1]
        else:
            p = cal_obj.transform(p)
    return p

# ===========================
# Probe loader
# ===========================

@dataclass
class LoadedProbe:
    kind: str  # "sk" or "xform"
    model: object
    feature_names: list

def load_probe_dir(probe_dir: Path) -> LoadedProbe:
    probe_dir = Path(probe_dir)
    feat_path = probe_dir / "feature_names.json"
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_names.json not found in {probe_dir}")
    feature_names = json.loads(feat_path.read_text())

    # Prefer transformer blob if present
    pt_path = probe_dir / "probe_model.pt"
    if pt_path.exists():
        blob = torch.load(pt_path, map_location="cpu")
        return LoadedProbe(kind="xform", model=blob, feature_names=feature_names)

    # else look for sklearn joblib
    jl_path = probe_dir / "probe_model.joblib"
    if jl_path.exists():
        model = joblib.load(jl_path)
        return LoadedProbe(kind="sk", model=model, feature_names=feature_names)

    raise FileNotFoundError(f"No probe model found in {probe_dir} (expected probe_model.pt or probe_model.joblib)")

# ===========================
# Feature row assembly
# ===========================

def assemble_feature_row(features: DemoFeatures, parsed_json_ok: int, parsed_p_true_ok: int, answer_span: str, p_true_self: float):
    scal = dict(features.scalars)
    scal["parsed_json_ok"] = int(parsed_json_ok)
    scal["parsed_p_true_ok"] = int(parsed_p_true_ok)
    scal["answer_len"] = int(len(answer_span.split())) if answer_span else 0
    if p_true_self is not None:
        scal["model_confidence"] = float(p_true_self)
    is_unknown = 1 if (answer_span.strip().lower() in {"unknown", ""}) else 0
    scal["is_unknown"] = is_unknown
    vecs = {k: to_vec256(v) for k, v in (features.vectors or {}).items()}
    return scal, vecs

def df_from_row(scalars: dict, vectors: dict, columns: list):
    row = dict(scalars)
    # Attach vector columns (name_0..255) if present
    for name, vec in (vectors or {}).items():
        for i in range(256):
            row[f"{name}_{i}"] = float(vec[i])
    # Build df and reindex to training feature order
    df = pd.DataFrame([row])
    df = df.reindex(columns=columns)
    return df

# ===========================
# Generation & logits capture
# ===========================

@torch.no_grad()
def generate_and_collect(tokenizer, model, prompt: str, max_new_tokens=64, temperature=0.2, top_p=0.95):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Generate with logits; we run step-by-step to capture token-level logits
    gen_ids = []
    logits_per_step = []
    past_key_values = None
    cur_ids = input_ids
    for step in range(max_new_tokens):
        out = model(cur_ids, use_cache=True, past_key_values=past_key_values, output_logits=True)
        logits = out.logits[:, -1, :].squeeze(0)  # [V]
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        # nucleus sampling
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cum > top_p).nonzero(as_tuple=True)[0]
            if cutoff.numel() > 0:
                last = cutoff[0].item()
                sorted_probs = sorted_probs[: last + 1]
                sorted_idx = sorted_idx[: last + 1]
                probs = sorted_probs / sorted_probs.sum()
                next_id = sorted_idx[torch.multinomial(probs, num_samples=1)].item()
            else:
                next_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_id = torch.multinomial(probs, num_samples=1).item()

        gen_ids.append(next_id)
        logits_per_step.append(logits.detach().float().cpu().numpy())

        next_token = torch.tensor([[next_id]], device=device)
        cur_ids = torch.cat([cur_ids, next_token], dim=1)
        past_key_values = out.past_key_values

        # simple stop on newline or close brace if the model produced a JSON
        if next_id == tokenizer.eos_token_id:
            break
        if len(gen_ids) > 4 and tokenizer.decode(gen_ids[-1:]).strip().endswith("}"):
            # give it a couple extra tokens to finish
            if len(gen_ids) > 8:
                break

    full_ids = torch.cat([input_ids, torch.tensor([gen_ids], device=device)], dim=1)
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    return text, gen_ids, logits_per_step, inputs["input_ids"][0].cpu().tolist()

# ===========================
# Main
# ===========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="llama31 or qwen")
    ap.add_argument("--probe_dir", type=str, required=True, help="Path to a single trained probe dir (e.g., probes/backend_llama/probe_mlp)")
    ap.add_argument("--use_hidden", action="store_true", help="Must match how the probe was trained")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    # Load LM
    tok, lm, model_id = load_lm(args.model)

    # Load probe
    probe = load_probe_dir(Path(args.probe_dir))
    print(f"[Probe] Loaded from {args.probe_dir} | kind={probe.kind}")

    # Interactive loop
    print("\nType your trivia question (or Ctrl-C to quit).\n")
    while True:
        try:
            q = input("Q: ").strip()
        except KeyboardInterrupt:
            print("\nBye.")
            return
        if not q:
            continue

        prompt = FEWSHOT_FIXED_ANSWERABLE.replace("{EVAL_QUESTION}", q)
        print("\n=== Prompt (few-shot, enforced JSON) ===")
        print(prompt)

        # Generate
        text, gen_ids, logits_per_step, prompt_ids = generate_and_collect(
            tok, lm, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p
        )
        print("\n=== Model output (raw) ===")
        tail = text.split("Q: {EVAL_QUESTION}")[-1] if "{EVAL_QUESTION}" in text else text
        print(text)

        # Extract JSON
        obj, frag = extract_first_json(text)
        if obj:
            ans = normalize_answer_span(obj.get("answer", ""))
            p_true_self = obj.get("p_true", None)
            try:
                p_true_self = float(p_true_self) if p_true_self is not None else None
            except Exception:
                p_true_self = None
            parsed_json_ok = 1
            parsed_p_true_ok = 1 if p_true_self is not None and 0.0 <= p_true_self <= 1.0 else 0
        else:
            ans = ""
            p_true_self = None
            parsed_json_ok = 0
            parsed_p_true_ok = 0

        print("\n=== Extracted JSON ===")
        if obj:
            print(json.dumps({"answer": ans, "p_true": p_true_self}, ensure_ascii=False))
        else:
            print("(none)")

        # Features
        if COLLECT == "featurize_for_probe":
            try:
                # Use user's training-time featurizer if available
                from scripts.collect_internals import featurize_for_probe
                feat = featurize_for_probe(
                    tokenizer=tok, model=lm,
                    prompt_text=prompt, full_text=text,
                    prompt_ids=prompt_ids, gen_token_ids=gen_ids,
                    logits_per_step=logits_per_step,
                    use_hidden=args.use_hidden,
                    backend=args.model
                )
                # Expected to return dict with scalar keys + vector packs if requested
                scalars = {k: feat.get(k, 0.0) for k in SCALAR_KEYS}
                vectors = {}
                if args.use_hidden:
                    for name in VECTOR_KEYS:
                        if name in feat and feat[name] is not None:
                            vectors[name] = to_vec256(feat[name])
                        else:
                            vectors[name] = np.zeros(256, dtype=np.float32)
            except Exception as e:
                print(f"[WARN] featurize_for_probe failed; using fallback. ({e})")
                demo = basic_featurizer(tok, lm, prompt_ids, gen_ids, logits_per_step)
                scalars, vectors = assemble_feature_row(demo, parsed_json_ok, parsed_p_true_ok, ans, p_true_self)
        else:
            demo = basic_featurizer(tok, lm, prompt_ids, gen_ids, logits_per_step)
            # fill in parsed flags, answer_len, model_confidence
            scalars, vectors = assemble_feature_row(demo, parsed_json_ok, parsed_p_true_ok, ans, p_true_self)

        # Build DF row in training column order
        df = df_from_row(scalars, vectors if args.use_hidden else {}, probe.feature_names)

        # Probe inference
        if probe.kind == "sk":
            p_probe = float(probe.model.predict_proba(df.values)[:, 1][0])
        else:
            # transformer blob
            p_arr = xform_predict_from_blob(probe.model, df, use_hidden=args.use_hidden)
            p_probe = float(p_arr[0])

        # Present
        print("\n=== Parsed self-report ===")
        print(f"p_true: {p_true_self if parsed_p_true_ok else None} | parsed_json_ok={bool(parsed_json_ok)} parsed_p_true_ok={bool(parsed_p_true_ok)}")
        print("\n=== Probe ===")
        print(f"P(correct): {p_probe:.3f}")

        # Optional: quick decode stats echo
        print("\n=== Decode stats (fallback approx) ===")
        dbg = {k: scalars.get(k) for k in ["answer_len", "lp_mean", "seq_conf", "entropy_mean", "margin_mean", "rescore_logp"]}
        dbg_str = "  ".join([f"{k}={dbg[k]:.3f}" if isinstance(dbg[k], (int,float)) else f"{k}={dbg[k]}" for k in dbg])
        print(dbg_str)

        # Final line
        print("\n--- Result ---")
        print(f"Answer: {ans!r}")
        print(f"LM self p_true: {p_true_self if parsed_p_true_ok else 'N/A'}")
        print(f"Probe P(correct): {p_probe:.3f}\n")

if __name__ == "__main__":
    main()
