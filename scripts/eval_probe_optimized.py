#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZED Probe Evaluation: Load model once, run all probes together

Usage:
    python -m scripts.eval_probe \
        --model llama \
        --probe_dirs backend_llama/probe_mlp,backend_llama/probe_logreg,backend_llama/probe_tree \
        --use_hidden \
        --datasets triviaqa,hotpotqa \
        --output results/llama_triviaqa_all_probes.json

Key optimization: Model inference runs ONCE per example, features extracted ONCE,
then all probes evaluate in parallel on the same features.
"""

import argparse
import json
import os
import re
import torch
import torch.nn as nn
import time
import joblib
import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

# Model imports
from src.models.qwen7b import Qwen7B
from src.models.llama31_8b import Llama31_8B
from src.data.datasets import load_qa, squad_em, squad_f1

# ============================================================================
# FEATURE FILTERING (must match train_probe.py)
# ============================================================================

SCALAR_KEYS_ALL = [
    "model_confidence", "lp_mean", "seq_conf",
    "entropy_mean", "entropy_std",
    "margin_mean", "margin_min",
    "rescore_logp", "answer_len",
    "parsed_json_ok", "parsed_p_true_ok", "is_unknown",
]

NON_GENERALIZABLE_FEATURES = {
    "answer_len",      # Dataset-specific
    "is_unknown",      # Task-specific
    "rescore_logp",    # Format-specific
    "parsed_json_ok",  # Format-specific
    "parsed_p_true_ok" # Format-specific
}

def get_scalar_keys(exclude_non_generalizable=False):
    """Get scalar keys, optionally excluding non-generalizable features"""
    if exclude_non_generalizable:
        return [k for k in SCALAR_KEYS_ALL if k not in NON_GENERALIZABLE_FEATURES]
    return SCALAR_KEYS_ALL

# Transformer probe architecture (must match train_probe.py)
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
        return x, attn

class TinyTransformerProbe(nn.Module):
    def __init__(self, n_tokens:int, d_model=256, nhead=8, num_layers=3, d_ff=1024,
                 p_drop=0.1, p_stoch=0.05):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens
        
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
        if token_mask.dim() != 1:
            token_mask = token_mask.view(-1)
        B, T, D = tokens_256.shape
        
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
        
        tt = torch.arange(T, device=x.device)
        x = x + self.token_type(tt)
        
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x)
        
        attn_list = [] if return_attn else None
        for layer in self.encoders:
            x, attn = layer(x, need_attn=return_attn)
            if return_attn:
                attn_list.append(attn)
        
        cls_out = x[:, 0, :]
        prob = self.head(cls_out).squeeze(-1)
        
        if return_attn:
            return prob, attn_list
        return prob

# Speed optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Optimal F1 thresholds from training
OPTIMAL_THRESHOLDS = {
    "llama": {
        "mlp": 0.327,
        "logreg": 0.308,
        "logreg_cal": 0.322,
        "tree": 0.296,
        "xform": 0.359,
    },
    "qwen": {
        "mlp": 0.314,
        "logreg": 0.351,
        "logreg_cal": 0.347,
        "tree": 0.269,
        "xform": 0.211,
    }
}

# Dataset test split mapping
DATASET_TEST_SPLITS = {
    "triviaqa": "validation",
    "hotpotqa": "validation",
    "hotpot_qa": "validation",
    "squad_v2": "validation",
    "gsm8k": "test",
    "mmlu": "validation",
}

# Prompts (same as collect_internals.py)
FEWSHOT_PROMPT_DEFAULT = """You are answering trivia questions.
Return only a single JSON object with fields:
- "answer": the short factual span (1–5 words, no punctuation)
- "p_true": the probability (0.0–1.0) that this answer is correct (round to two decimals)

Calibration guidance:
- Report your true probability; do NOT inflate.
- Overconfidence is penalized by proper scoring (Brier). If unsure, choose a lower value.
- If multiple plausible answers exist or the question is ambiguous, reduce p_true appropriately.

Q: Who wrote Hamlet?
{{"answer": "William Shakespeare", "p_true": 0.95}}

Q: What is the capital of France?
{{"answer": "Paris", "p_true": 0.92}}

Q: What is the capital of South Africa?
{{"answer": "Pretoria", "p_true": 0.60}}

Q: Which element has the symbol 'Au'?
{{"answer": "Aluminum", "p_true": 0.15}}

Q: Who authored the Voynich Manuscript?
{{"answer": "Unknown", "p_true": 0.20}}

Q: Which planet is known as the Red Planet?
{{"answer": "Mars", "p_true": 0.85}}

Q: {EVAL_QUESTION}
"""

FEWSHOT_PROMPT_SQUADV2 = """You are answering extractive QA with possible unanswerable questions (SQuAD v2).
Return only a single JSON object with fields:
- "answer": the short factual span (1–5 words, no punctuation). If unanswerable, output "Unknown".
- "p_true": the probability (0.0–1.0) that this answer is correct (round to two decimals)

Calibration guidance:
- Predict "Unknown" when there is no sufficient answer in the context.
- Report your true probability; do NOT inflate.
- Overconfidence is penalized by proper scoring (Brier). If unsure, choose a lower value.

Q: Who wrote Hamlet?
{{"answer": "William Shakespeare", "p_true": 0.95}}

Q: In 2007, who was the prime minister of Canada? (unanswerable)
{{"answer": "Unknown", "p_true": 0.15}}

Q: {EVAL_QUESTION}
"""

def build_prompt(question: str, dataset: str) -> str:
    """Build prompt with dataset-specific formatting"""
    if dataset.lower() in ("squadv2", "squad_v2", "squad2"):
        return FEWSHOT_PROMPT_SQUADV2.format(EVAL_QUESTION=question)
    else:
        return FEWSHOT_PROMPT_DEFAULT.format(EVAL_QUESTION=question)

def safe_json_extract(text: str):
    t = text.strip()
    
    # Normalize Unicode fancy quotes to ASCII quotes
    # Models sometimes output " " (U+201C/U+201D) instead of " (U+0022)
    t = t.replace('\u201c', '"')  # LEFT DOUBLE QUOTATION MARK
    t = t.replace('\u201d', '"')  # RIGHT DOUBLE QUOTATION MARK
    t = t.replace('\u2018', "'")  # LEFT SINGLE QUOTATION MARK
    t = t.replace('\u2019', "'")  # RIGHT SINGLE QUOTATION MARK
    
    if t.startswith("```"):
        t = t.lstrip("`")
        if "\n" in t:
            t = t.split("\n", 1)[1].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    try:
        if "\\\"" in t or "\\\\\"" in t:
            t_try = bytes(t, "utf-8").decode("unicode_escape")
            if "{" in t_try and "}" in t_try:
                t = t_try
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None

def extract_answer_and_prob(text: str):
    obj = safe_json_extract(text)
    if isinstance(obj, dict):
        norm = {str(k).strip().lower(): v for k, v in obj.items()}
        ans = (norm.get("answer") or norm.get("final") or norm.get("prediction") or "").strip()
        p = None
        for key in ["p_true", "confidence", "conf", "prob", "probability"]:
            if key in norm:
                try:
                    p = float(norm[key])
                except Exception:
                    try:
                        p = float(str(norm[key]).replace("%", "")) / 100.0
                    except Exception:
                        pass
                break
        if p is not None:
            p = float(max(0.0, min(1.0, p if p <= 1.0 else p/100.0)))
        return ans, p
    return text.strip(), None

def pack_256(vec: torch.Tensor):
    """L2-normalize then take first 256 dims."""
    if vec is None:
        return None
    v = torch.nn.functional.normalize(vec.float(), dim=-1)
    v = v[:256] if v.shape[-1] >= 256 else torch.nn.functional.pad(v, (0, 256 - v.shape[-1]))
    return [float(x) for x in v.cpu()]

def initialize_model(model_name: str, max_new_tokens: int = 64):
    """Initialize the model"""
    if model_name in ("llama", "llama31"):
        model = Llama31_8B(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            dtype="float16",
            device_map=None,
            max_new_tokens=max_new_tokens,
            cache_dir=None
        )
    elif model_name == "qwen":
        model = Qwen7B(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            dtype="float16",
            device_map=None,
            max_new_tokens=max_new_tokens,
            cache_dir=None
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")
    
    print(f"[Model] Initialized: {model_name}")
    return model

def load_probes(probe_dirs: List[str], use_hidden: bool, model_name: str) -> List[Dict]:
    """Load multiple probes at once"""
    probes = []
    
    # Normalize model name for threshold lookup
    model_key = "llama" if model_name in ("llama", "llama31") else model_name
    
    for probe_dir in probe_dirs:
        probe_path = Path(probe_dir)
        if not probe_path.exists():
            print(f"[WARNING] Probe directory not found: {probe_dir}")
            continue
        
        # Extract probe type from directory name
        probe_type = probe_path.name.replace("probe_", "")
        
        # Try to load probe model
        model_files = list(probe_path.glob("probe_model.*"))
        if not model_files:
            print(f"[WARNING] No probe model found in {probe_dir}")
            continue
        
        model_file = model_files[0]
        
        # Load probe based on extension
        if model_file.suffix == ".joblib":
            probe_model = joblib.load(model_file)
        elif model_file.suffix == ".pt":
            checkpoint = torch.load(model_file, map_location="cpu")
            # Transformer checkpoint: reconstruct from architecture + state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                arch = checkpoint["arch"]
                # Determine n_tokens from feature_names or use_hidden
                if use_hidden:
                    n_tokens = 5  # 4 hidden packs + 1 scalar
                else:
                    n_tokens = 1  # scalar only
                
                # Reconstruct model
                probe_model = TinyTransformerProbe(
                    n_tokens=n_tokens,
                    d_model=arch["d_model"],
                    nhead=arch["nhead"],
                    num_layers=arch["num_layers"],
                    d_ff=arch["d_ff"],
                    p_drop=arch["p_drop"],
                    p_stoch=arch["p_stoch"]
                )
                probe_model.load_state_dict(checkpoint["model_state_dict"])
                probe_model.eval()
                
                if torch.cuda.is_available():
                    probe_model = probe_model.cuda()
                
                # Store train_mask and calibration for inference
                train_mask = np.array(checkpoint.get("train_mask", [True]*(n_tokens-1) + [False]))
                probe_model._train_mask = train_mask
                probe_model._calibration = checkpoint.get("calibration")
            else:
                print(f"[WARNING] Unknown transformer checkpoint format in {model_file}")
                continue
        else:
            print(f"[WARNING] Unknown probe model format: {model_file}")
            continue
        
        # Get threshold - try both model_key and original model_name
        threshold = OPTIMAL_THRESHOLDS.get(model_key, {}).get(probe_type, 
                    OPTIMAL_THRESHOLDS.get(model_name, {}).get(probe_type, 0.5))
        
        # Load feature names
        feature_file = probe_path / "feature_names.json"
        feature_names = None
        if feature_file.exists():
            with open(feature_file) as f:
                feature_names = json.load(f)
        
        # Detect if probe was trained with excluded features
        n_features_expected = None
        exclude_non_generalizable = False
        
        try:
            if hasattr(probe_model, 'n_features_in_'):
                n_features_expected = probe_model.n_features_in_
            elif hasattr(probe_model, 'named_steps'):
                if 'impute' in probe_model.named_steps:
                    imputer = probe_model.named_steps['impute']
                    if hasattr(imputer, 'n_features_in_'):
                        n_features_expected = imputer.n_features_in_
            
            if n_features_expected is not None:
                n_scalar_all = len(SCALAR_KEYS_ALL)
                n_scalar_gen = len([k for k in SCALAR_KEYS_ALL if k not in NON_GENERALIZABLE_FEATURES])
                n_hidden = 4 * 256 if use_hidden else 0
                
                n_features_all = n_scalar_all + n_hidden  # 1036 with hidden
                n_features_gen = n_scalar_gen + n_hidden  # 1031 with hidden
                
                if n_features_expected == n_features_gen:
                    exclude_non_generalizable = True
                    print(f"[Probe] {probe_type}: trained WITHOUT non-generalizable features ({n_features_expected})")
                elif n_features_expected == n_features_all:
                    print(f"[Probe] {probe_type}: trained WITH all features ({n_features_expected})")
                else:
                    print(f"[Probe] {probe_type}: unexpected feature count {n_features_expected}")
        except Exception as e:
            pass  # Silently continue if detection fails
        
        probe_info = {
            "type": probe_type,
            "model": probe_model,
            "threshold": threshold,
            "use_hidden": use_hidden,
            "feature_names": feature_names,
            "path": str(probe_path),
            "exclude_non_generalizable": exclude_non_generalizable
        }
        
        probes.append(probe_info)
        print(f"[Probe] Loaded: {probe_type} (threshold={threshold:.3f})")
    
    return probes

def extract_features(model, input_ids, gen, new_ids, prompt, answer, p_model, use_hidden) -> Dict:
    """Extract features from model output (same as collect_internals.py)"""
    
    # Token-level stats from scores
    scores = gen.scores or []
    T = len(scores)
    gen_ids_seq = new_ids[0].tolist()
    content_len = max(1, T - 1) if T > 1 else T
    
    step_ent = []
    margins = []
    step_logp = []
    for t in range(T):
        logits = scores[t][0].float()
        top2 = torch.topk(logits, k=2).values
        margins.append(float(top2[0] - top2[1]))
        
        probs = torch.softmax(logits, dim=-1)
        ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        step_ent.append(ent)
        
        tok_id = gen_ids_seq[t] if t < len(gen_ids_seq) else None
        if tok_id is not None:
            lsm = torch.log_softmax(logits, dim=-1)
            step_logp.append(float(lsm[tok_id].item()))
    
    entropy_mean = float(sum(step_ent[:content_len]) / content_len) if step_ent else None
    entropy_last = float(step_ent[-1]) if step_ent else None
    entropy_std = float(torch.tensor(step_ent[:content_len]).std().item()) if len(step_ent[:content_len]) > 1 else None
    
    margin_mean = float(sum(margins[:content_len]) / content_len) if margins else None
    margin_last = float(margins[-1]) if margins else None
    margin_min = float(min(margins[:content_len])) if content_len > 0 and margins else None
    
    lp_mean = float(sum(step_logp[:content_len]) / content_len) if step_logp else None
    seq_conf = math.exp(lp_mean) if lp_mean is not None else None
    
    # Hidden states (if requested)
    h_last_256 = None
    h_pool_256 = None
    h_last_mid_256 = None
    h_pool_mid_256 = None
    
    if use_hidden:
        device = next(model.model.parameters()).device
        with torch.no_grad():
            full_ids = torch.cat([input_ids.to(device), new_ids.to(device)], dim=-1)
            attn = torch.ones_like(full_ids)
            out = model.model(input_ids=full_ids, attention_mask=attn,
                            output_hidden_states=True, use_cache=False, return_dict=True)
            hs_final = out.hidden_states[-1][0]
            gen_len = new_ids.shape[-1]
            h_ans = hs_final[-gen_len:]
            h_last = h_ans[-1] if h_ans.shape[0] > 0 else None
            h_pool = h_ans.mean(dim=0) if h_ans.shape[0] > 0 else None
            
            mid_ix = len(out.hidden_states) // 2
            hs_mid = out.hidden_states[mid_ix][0]
            h_ans_mid = hs_mid[-gen_len:]
            h_last_mid = h_ans_mid[-1] if h_ans_mid.shape[0] > 0 else None
            h_pool_mid = h_ans_mid.mean(dim=0) if h_ans_mid.shape[0] > 0 else None
        
        h_last_256 = pack_256(h_last) if h_last is not None else None
        h_pool_256 = pack_256(h_pool) if h_pool is not None else None
        h_last_mid_256 = pack_256(h_last_mid) if h_last_mid is not None else None
        h_pool_mid_256 = pack_256(h_pool_mid) if h_pool_mid is not None else None
    
    # Rescore canonical JSON
    safe_ans = answer.replace('"', '\\"')
    pt = p_model if p_model is not None else 0.00
    canon_json = f'{{"answer":"{safe_ans}","p_true":{pt:.2f}}}'
    
    device = next(model.model.parameters()).device
    with torch.no_grad():
        enc_prompt = model.tok(prompt, add_special_tokens=False, return_tensors="pt")
        enc_target = model.tok(canon_json, add_special_tokens=False, return_tensors="pt")
        input_ids_rescore = torch.cat([enc_prompt["input_ids"], enc_target["input_ids"]], dim=-1).to(device)
        attn_rescore = torch.ones_like(input_ids_rescore)
        out_rescore = model.model(input_ids=input_ids_rescore, attention_mask=attn_rescore, use_cache=False, return_dict=True)
        logits_rescore = out_rescore.logits[:, :-1, :]
        target_slice = input_ids_rescore[:, enc_prompt["input_ids"].shape[-1]:]
        logits_tgt = logits_rescore[:, -target_slice.shape[-1]:, :]
        step_logps = []
        for t in range(target_slice.shape[-1]):
            lsm = torch.log_softmax(logits_tgt[0, t].float(), dim=-1)
            step_logps.append(float(lsm[int(target_slice[0, t])].item()))
        rescore_lp = sum(step_logps) / max(1, len(step_logps))
    
    gen_len = new_ids.shape[-1]
    
    features = {
        "model_confidence": p_model,
        "lp_mean": lp_mean,
        "seq_conf": seq_conf,
        "entropy_mean": entropy_mean,
        "entropy_last": entropy_last,
        "entropy_std": entropy_std,
        "margin_mean": margin_mean,
        "margin_last": margin_last,
        "margin_min": margin_min,
        "h_last_256": h_last_256,
        "h_pool_256": h_pool_256,
        "h_last_mid_256": h_last_mid_256,
        "h_pool_mid_256": h_pool_mid_256,
        "rescore_logp": rescore_lp,
        "answer_len": int(gen_len),
        "parsed_json_ok": int(answer is not None and len(answer) > 0),
        "parsed_p_true_ok": int(p_model is not None),
        "is_unknown": int(answer.strip().lower() == "unknown"),
    }
    
    return features

def run_probe_inference(probe: Dict, features: Dict) -> float:
    """Run probe inference on extracted features"""
    probe_model = probe["model"]
    probe_type = probe["type"]
    feature_names = probe["feature_names"]
    
    # Special handling for transformer probes
    if probe_type == "xform" and isinstance(probe_model, TinyTransformerProbe):
        # Get scalar keys (filtered if needed)
        exclude_non_generalizable = probe.get("exclude_non_generalizable", False)
        SCALAR_KEYS = get_scalar_keys(exclude_non_generalizable)
        VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]
        
        # Build tokens_256 tensor
        tokens = []
        if probe["use_hidden"]:
            for k in VECTOR_KEYS:
                vec = features.get(k, [0.0]*256)
                if vec is None:
                    vec = [0.0]*256
                # Pad to 256 if needed
                vec = list(vec) + [0.0]*(256-len(vec)) if len(vec) < 256 else vec[:256]
                tokens.append(vec)
        
        # Add scalars (padded to 256, but only first N used where N=len(SCALAR_KEYS))
        scalar_vals = []
        for k in SCALAR_KEYS:
            val = features.get(k, 0.0)
            scalar_vals.append(float(val if val is not None else 0.0))
        scalar_vals += [0.0] * (256 - len(scalar_vals))  # Pad to 256
        tokens.append(scalar_vals)
        
        # Create tensor: [1, n_tokens, 256]
        tokens_tensor = torch.tensor([tokens], dtype=torch.float32)
        
        # Create token_mask
        train_mask = probe_model._train_mask
        token_mask = torch.tensor(train_mask, dtype=torch.bool)
        
        # Run transformer inference
        with torch.no_grad():
            if torch.cuda.is_available():
                tokens_tensor = tokens_tensor.cuda()
                token_mask = token_mask.cuda()
            prob = probe_model(tokens_tensor, token_mask).item()
        
        # Apply calibration if present
        cal_blob = probe_model._calibration
        if cal_blob is not None and "isotonic" in cal_blob:
            iso_reg = cal_blob["isotonic"]
            prob = float(iso_reg.predict([prob])[0])
        
        return float(prob)
    
    # Standard sklearn/MLP probes
    # Get scalar keys (filtered if needed)
    exclude_non_generalizable = probe.get("exclude_non_generalizable", False)
    SCALAR_KEYS = get_scalar_keys(exclude_non_generalizable)
    VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]
    
    feat_dict = {}
    for k in SCALAR_KEYS:
        feat_dict[k] = features.get(k, np.nan)
    
    if probe["use_hidden"]:
        for name in VECTOR_KEYS:
            vec = features.get(name)
            if vec is not None:
                for j, v in enumerate(vec):
                    feat_dict[f"{name}_{j}"] = float(v)
    
    # Use DataFrame to ensure column order matches training
    # (sklearn probes were trained on DataFrame columns which are alphabetically sorted)
    df = pd.DataFrame([feat_dict])
    
    if hasattr(probe_model, "predict_proba"):
        # sklearn classifiers
        prob = probe_model.predict_proba(df.values)[:, 1][0]
    elif isinstance(probe_model, torch.nn.Module):
        # PyTorch models (non-transformer)
        with torch.no_grad():
            feat_tensor = torch.tensor(df.values, dtype=torch.float32)
            if torch.cuda.is_available():
                feat_tensor = feat_tensor.cuda()
            prob = probe_model(feat_tensor).item()
    else:
        raise ValueError(f"Unknown probe model type: {type(probe_model)}")
    
    return float(prob)

def evaluate_on_dataset(model, probes: List[Dict], dataset_name: str, limit: int = None):
    """
    Evaluate ALL probes on a dataset
    Key optimization: Run model inference ONCE, extract features ONCE,
    then run all probes on the same features
    """
    split = DATASET_TEST_SPLITS.get(dataset_name, "validation")
    
    print(f"\n[{dataset_name}] Loading dataset (split: {split})...")
    ds = load_qa(dataset_name, split, limit)
    print(f"[{dataset_name}] Loaded {len(ds)} examples")
    
    # Initialize results for each probe
    probe_results = {probe["type"]: [] for probe in probes}
    
    # Disable tqdm in non-interactive environments
    import sys
    disable_tqdm = not sys.stdout.isatty()
    
    for idx, ex in enumerate(tqdm(ds, desc=f"Evaluating {dataset_name}", disable=disable_tqdm)):
        if disable_tqdm and (idx + 1) % 100 == 0:
            print(f"[{dataset_name}] Processed {idx + 1}/{len(ds)} examples")
        
        question = ex["question"]
        gold_answers = ex["answers"]
        
        # === RUN MODEL INFERENCE ONCE ===
        # Use dataset-specific prompt
        prompt = build_prompt(question, dataset_name)
        inp, gen = model.generate_with_states(prompt)
        
        start_pos = inp["input_ids"].shape[-1]
        new_ids = gen.sequences[:, start_pos:]
        raw_text = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
        
        answer, p_model = extract_answer_and_prob(raw_text)
        if not answer:
            answer = raw_text
        
        # Compute EM (ground truth)
        em = 1 if squad_em(answer, gold_answers) else 0
        
        # === EXTRACT FEATURES ONCE ===
        try:
            # Use the most permissive use_hidden (if ANY probe needs hidden states)
            use_hidden = any(p["use_hidden"] for p in probes)
            features = extract_features(model, inp["input_ids"], gen, new_ids,
                                       prompt, answer, p_model, use_hidden)
        except Exception as e:
            print(f"[WARNING] Feature extraction failed on example {idx}: {e}")
            # Skip this example for all probes
            continue
        
        # === RUN ALL PROBES ON SAME FEATURES ===
        for probe in probes:
            try:
                probe_prob = run_probe_inference(probe, features)
                probe_pred = 1 if probe_prob > probe["threshold"] else 0
            except Exception as e:
                print(f"[WARNING] Probe {probe['type']} failed on example {idx}: {e}")
                probe_prob = None
                probe_pred = None
            
            probe_results[probe["type"]].append({
                "question": question,
                "answer": answer,
                "gold_answers": gold_answers,
                "em": em,
                "probe_prob": probe_prob,
                "probe_pred": probe_pred,
            })
    
    # Compute metrics for each probe at multiple thresholds
    all_metrics = []
    thresholds_to_test = [0.3, 0.5, 0.7]
    
    for probe in probes:
        results = probe_results[probe["type"]]
        valid_results = [r for r in results if r["probe_pred"] is not None]
        
        if not valid_results:
            print(f"[{dataset_name}] [{probe['type']}] No valid predictions!")
            continue
        
        y_true = [r["em"] for r in valid_results]
        y_probs = [r["probe_prob"] for r in valid_results]
        model_accuracy = sum(y_true) / len(y_true) if len(y_true) > 0 else 0.0
        
        # Compute metrics at multiple thresholds
        threshold_results = []
        for thresh in thresholds_to_test:
            y_pred = [1 if p > thresh else 0 for p in y_probs]
            
            # Confusion matrix
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
            
            accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            threshold_results.append({
                "threshold": thresh,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            })
        
        metrics = {
            "dataset": dataset_name,
            "probe_type": probe["type"],
            "n_examples": len(valid_results),
            "model_accuracy": model_accuracy,
            "threshold_results": threshold_results,
        }
        
        all_metrics.append(metrics)
    
    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple probes on test splits (OPTIMIZED)")
    parser.add_argument("--model", choices=["llama", "llama31", "qwen"], required=True,
                       help="Model name (llama/llama31 and qwen both supported)")
    parser.add_argument("--probe_dirs", type=str, required=True,
                       help="Comma-separated list of probe directories")
    parser.add_argument("--use_hidden", action="store_true",
                       help="Use hidden states (must match probe training)")
    parser.add_argument("--datasets", type=str,
                       default="triviaqa,hotpotqa,squad_v2,gsm8k,mmlu",
                       help="Comma-separated list of datasets")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit examples per dataset (for testing)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results (one file per dataset+probe)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("OPTIMIZED PROBE EVALUATION (Load model once, run all probes)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Probe dirs: {args.probe_dirs}")
    print(f"Use hidden states: {args.use_hidden}")
    print(f"Datasets: {args.datasets}")
    if args.limit:
        print(f"Limit: {args.limit} examples per dataset")
    print("=" * 80)
    
    # Initialize model ONCE
    model = initialize_model(args.model, args.max_new_tokens)
    
    # Load ALL probes ONCE
    probe_dirs = [d.strip() for d in args.probe_dirs.split(",")]
    probes = load_probes(probe_dirs, args.use_hidden, args.model)
    
    if not probes:
        print("[ERROR] No probes loaded!")
        return
    
    print(f"\n[Loaded {len(probes)} probes]")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate on each dataset
    datasets = [d.strip() for d in args.datasets.split(",")]
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        all_metrics = evaluate_on_dataset(model, probes, dataset_name, args.limit)
        
        # Save results per probe
        for metrics in all_metrics:
            probe_type = metrics["probe_type"]
            output_file = os.path.join(
                args.output_dir,
                f"{args.model}_{dataset_name}_{probe_type}_eval.json"
            )
            
            # Format to match analyze_results.py expectations
            output_data = {
                "model": args.model,
                "datasets": [metrics]  # Single dataset, but in array format
            }
            
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"  [{probe_type}] Saved: {output_file}")
            print(f"    F1={metrics['probe_f1']:.3f}, "
                  f"Acc={metrics['probe_accuracy']:.3f}, "
                  f"P={metrics['probe_precision']:.3f}, "
                  f"R={metrics['probe_recall']:.3f}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()