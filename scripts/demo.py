#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for running LLM inference with probe predictions

Example usage:
    python -m scripts.demo --model llama31 --probe_dir probes/backend_llama/probe_mlp --use_hidden --interactive

"""
import argparse
import json
import re
import torch
import time
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Model imports
from src.models.qwen7b import Qwen7B
from src.models.llama31_8b import Llama31_8B

# For transformer probe
import torch.nn as nn

# Speed optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Prompts (identical to collect_internals.py)
FEWSHOT_PROMPT = """You are answering trivia questions.
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

def safe_json_extract(text: str) -> Optional[Dict]:
    """Extract JSON from model output (same as collect_internals)"""
    t = text.strip()
    
    # Strip code fences
    if t.startswith("```"):
        t = t.lstrip("`")
        if "\n" in t:
            t = t.split("\n", 1)[1].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    
    # Try to unescape
    try:
        if "\\\"" in t or "\\\\\"" in t:
            t_try = bytes(t, "utf-8").decode("unicode_escape")
            if "{" in t_try and "}" in t_try:
                t = t_try
    except Exception:
        pass
    
    # Find JSON object
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception as e:
            print(f"[warn] JSON parse failed: {e}")
    return None

def extract_answer_and_prob(text: str) -> Tuple[str, Optional[float]]:
    """Parse answer and p_true from model output"""
    obj = safe_json_extract(text)
    if isinstance(obj, dict):
        # Normalize keys
        norm = {str(k).strip().lower(): v for k, v in obj.items()}
        
        # Get answer
        ans = (norm.get("answer") or norm.get("final") or norm.get("prediction") or "").strip()
        
        # Get probability
        p = None
        for key in ["p_true", "confidence", "conf", "prob", "probability"]:
            if key in norm:
                try:
                    p = float(norm[key])
                    if p > 1.0:  # Handle percentages
                        p = p / 100.0
                    p = max(0.0, min(1.0, p))
                    break
                except Exception:
                    pass
        
        return ans, p
    
    # Fallback
    return text.strip(), None

def initialize_model(model_name: str, max_new_tokens: int = 64):
    """Initialize the specified model"""
    if model_name == "llama31":
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
    
    # Show GPU info
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")
    
    return model

def run_inference(model, question: str) -> Dict:
    """Run model inference on a question"""
    # Build prompt
    prompt = FEWSHOT_PROMPT.format(EVAL_QUESTION=question)
    
    # Generate
    t0 = time.perf_counter()
    inp, gen = model.generate_with_states(prompt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    
    # Extract generated text
    start_pos = inp["input_ids"].shape[-1]
    new_ids = gen.sequences[:, start_pos:]
    raw_text = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
    
    # Parse answer and probability
    answer, p_model = extract_answer_and_prob(raw_text)
    if not answer:
        answer = raw_text
    
    result = {
        "question": question,
        "answer": answer,
        "model_confidence": p_model,
        "raw_output": raw_text,
        "gen_tokens": new_ids.shape[-1],
        "gen_time": round(dt, 3),
        "prompt": prompt,  # Keep for Stage 2
        "input_ids": inp["input_ids"],  # Keep for Stage 2
        "generated_ids": new_ids,  # Keep for Stage 2
        "generation_output": gen,  # Keep for Stage 2
    }
    
    return result

# ================= Stage 3: Feature Extraction and Probe Inference =================

import math

def pack_256(vec: torch.Tensor):
    """L2-normalize then take first 256 dims."""
    if vec is None:
        return None
    v = torch.nn.functional.normalize(vec.float(), dim=-1)
    v = v[:256] if v.shape[-1] >= 256 else torch.nn.functional.pad(v, (0, 256 - v.shape[-1]))
    return [float(x) for x in v.cpu()]

def rescore_mean_logprob(model, tok, prompt_text: str, target_text: str):
    """Compute mean log-prob of target_text when appended to prompt_text."""
    with torch.no_grad():
        enc_prompt = tok(prompt_text, add_special_tokens=False, return_tensors="pt")
        enc_target = tok(target_text, add_special_tokens=False, return_tensors="pt")
        input_ids = torch.cat([enc_prompt["input_ids"], enc_target["input_ids"]], dim=-1).to(model.device)
        attn = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
        logits = out.logits[:, :-1, :]
        target_slice = input_ids[:, enc_prompt["input_ids"].shape[-1]:]
        logits_tgt = logits[:, -target_slice.shape[-1]:, :]
        step_logps = []
        for t in range(target_slice.shape[-1]):
            lsm = torch.log_softmax(logits_tgt[0, t].float(), dim=-1)
            step_logps.append(float(lsm[int(target_slice[0, t])].item()))
        return sum(step_logps) / max(1, len(step_logps))

def extract_features(model, result: Dict, use_hidden: bool = True) -> Dict:
    """Extract features from model output matching collect_internals.py"""
    
    # Get basic info from result
    prompt = result["prompt"]
    input_ids = result["input_ids"]
    new_ids = result["generated_ids"]
    gen = result["generation_output"]
    answer = result["answer"]
    p_model = result["model_confidence"]
    
    # Get the underlying HF model
    hf_model = model.model
    tok = model.tok
    device = next(hf_model.parameters()).device
    
    # Token-level statistics from generation scores
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
    
    # Compute statistics
    entropy_mean = float(sum(step_ent[:content_len]) / content_len) if step_ent else 0.0
    entropy_std = float(torch.tensor(step_ent[:content_len]).std().item()) if len(step_ent[:content_len]) > 1 else 0.0
    margin_mean = float(sum(margins[:content_len]) / content_len) if margins else 0.0
    margin_min = float(min(margins[:content_len])) if content_len > 0 and margins else 0.0
    lp_mean = float(sum(step_logp[:content_len]) / content_len) if step_logp else 0.0
    seq_conf = math.exp(lp_mean) if lp_mean != 0.0 else 0.0
    
    # Get hidden states if requested
    h_last_256 = None
    h_pool_256 = None
    h_last_mid_256 = None
    h_pool_mid_256 = None
    
    if use_hidden:
        with torch.no_grad():
            full_ids = torch.cat([input_ids.to(device), new_ids.to(device)], dim=-1)
            attn = torch.ones_like(full_ids)
            out = hf_model(
                input_ids=full_ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            hs_final = out.hidden_states[-1][0]  # [T, d]
            gen_len = new_ids.shape[-1]
            h_ans = hs_final[-gen_len:]
            h_last = h_ans[-1] if h_ans.shape[0] > 0 else None
            h_pool = h_ans.mean(dim=0) if h_ans.shape[0] > 0 else None
            
            mid_ix = len(out.hidden_states) // 2
            hs_mid = out.hidden_states[mid_ix][0]
            h_ans_mid = hs_mid[-gen_len:]
            h_last_mid = h_ans_mid[-1] if h_ans_mid.shape[0] > 0 else None
            h_pool_mid = h_ans_mid.mean(dim=0) if h_ans_mid.shape[0] > 0 else None
            
            h_last_256 = pack_256(h_last) if h_last is not None else [0.0] * 256
            h_pool_256 = pack_256(h_pool) if h_pool is not None else [0.0] * 256
            h_last_mid_256 = pack_256(h_last_mid) if h_last_mid is not None else [0.0] * 256
            h_pool_mid_256 = pack_256(h_pool_mid) if h_pool_mid is not None else [0.0] * 256
    
    # Rescore canonical JSON
    safe_ans = answer.replace('"', '\\"')
    pt = p_model if p_model is not None else 0.00
    canon_json = f'{{"answer":"{safe_ans}","p_true":{pt:.2f}}}'
    rescore_lp = rescore_mean_logprob(hf_model, tok, prompt, canon_json)
    
    # Build feature dict
    features = {
        "model_confidence": p_model if p_model is not None else 0.0,
        "lp_mean": lp_mean,
        "seq_conf": seq_conf,
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "margin_mean": margin_mean,
        "margin_min": margin_min,
        "rescore_logp": rescore_lp,
        "answer_len": int(new_ids.shape[-1]),
        "parsed_json_ok": int(answer is not None and len(answer) > 0),
        "parsed_p_true_ok": int(p_model is not None),
        "is_unknown": int(answer.strip().lower() == "unknown"),
    }
    
    if use_hidden:
        features["h_last_256"] = h_last_256
        features["h_pool_256"] = h_pool_256
        features["h_last_mid_256"] = h_last_mid_256
        features["h_pool_mid_256"] = h_pool_mid_256
    
    return features

def build_transformer_tokens(features: Dict, use_hidden: bool):
    """Build token tensor for transformer probe"""
    tokens = []
    
    if use_hidden:
        # Add hidden vectors as tokens
        for key in VECTOR_KEYS:
            vec = features.get(key, [0.0] * 256)
            if not isinstance(vec, list):
                vec = [0.0] * 256
            vec = np.array(vec[:256], dtype=np.float32)
            if len(vec) < 256:
                vec = np.pad(vec, (0, 256 - len(vec)), 'constant')
            tokens.append(vec)
    
    # Add scalar token (padded to 256)
    scalars = []
    for key in SCALAR_KEYS:
        scalars.append(float(features.get(key, 0.0)))
    scalar_array = np.array(scalars, dtype=np.float32)
    scalar_padded = np.pad(scalar_array, (0, 256 - len(scalars)), 'constant')
    tokens.append(scalar_padded)
    
    # Stack into [1, T, 256] tensor
    return np.stack(tokens, axis=0)[np.newaxis, :, :]

def run_probe_inference(probe_info: Dict, features: Dict) -> float:
    """Run probe inference on extracted features"""
    probe_type = probe_info["type"]
    model = probe_info["model"]
    use_hidden = probe_info["use_hidden"]
    
    if probe_type == "xform":
        # Transformer probe
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Build tokens
        tokens = build_transformer_tokens(features, use_hidden)
        
        with torch.no_grad():
            tokens_tensor = torch.tensor(tokens, dtype=torch.float32).to(device)
            prob = model(tokens_tensor).cpu().item()
        
        # Apply calibration if present
        if probe_info["calibration"] is not None:
            cal_blob = probe_info["calibration"]
            cal = pickle.loads(cal_blob["payload"])
            if cal_blob["type"] == "platt":
                prob = cal.predict_proba([[prob]])[0, 1]
            else:  # isotonic
                prob = cal.transform([prob])[0]
        
        return float(prob)
    
    else:
        # Sklearn probe
        # Build DataFrame with features
        feat_dict = {}
        for key in SCALAR_KEYS:
            feat_dict[key] = features.get(key, 0.0)
        
        if use_hidden:
            for vec_key in VECTOR_KEYS:
                vec = features.get(vec_key, [0.0] * 256)
                if not isinstance(vec, list):
                    vec = [0.0] * 256
                for i in range(256):
                    feat_dict[f"{vec_key}_{i}"] = vec[i] if i < len(vec) else 0.0
        
        df = pd.DataFrame([feat_dict])
        
        # Get probability
        prob = model.predict_proba(df.values)[0, 1]
        return float(prob)

# Transformer probe classes (from train_probe.py)
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
    def __init__(self,
                 use_hidden=True,
                 d_model=256, nhead=8, num_layers=3, d_ff=1024,
                 p_drop=0.1, p_stoch=0.05):
        super().__init__()
        self.use_hidden = use_hidden
        self.d_model = d_model

        if use_hidden:
            # Use ModuleList to match the checkpoint structure
            self.hidden_projs = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(256), nn.Linear(256, d_model)),
                nn.Sequential(nn.LayerNorm(256), nn.Linear(256, d_model)),
                nn.Sequential(nn.LayerNorm(256), nn.Linear(256, d_model)),
                nn.Sequential(nn.LayerNorm(256), nn.Linear(256, d_model))
            ])
            n_tokens = 5
        else:
            self.hidden_projs = None
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
        B, T, D = tokens_256.shape
        tok_list = []
        idx = 0
        if self.use_hidden and self.hidden_projs is not None:
            # Project each hidden vector type
            for proj in self.hidden_projs:
                tok_list.append(proj(tokens_256[:, idx, :]))
                idx += 1
        scal_raw = tokens_256[:, -1, :13]
        tok_list.append(self.scalar_proj(scal_raw))
        x = torch.stack(tok_list, dim=1)
        return x

    def forward(self, tokens_256, return_attn=False):
        x = self._project_tokens(tokens_256)
        B, T, D = x.shape
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
        pooled = x[:, 0, :]
        p = self.head(pooled).squeeze(-1)
        return (p, attn_list) if return_attn else p

# Feature keys from train_probe.py
SCALAR_KEYS = [
    "model_confidence", "lp_mean", "seq_conf",
    "entropy_mean", "entropy_std",
    "margin_mean", "margin_min",
    "rescore_logp", "answer_len",
    "parsed_json_ok", "parsed_p_true_ok", "is_unknown",
]
VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]

def load_probe(probe_dir: str, use_hidden: bool = True):
    """Load a trained probe from directory"""
    probe_path = Path(probe_dir)
    
    # Determine probe type from directory name or by checking files
    if "xform" in probe_dir or probe_path.name == "probe_xform":
        probe_type = "xform"
    elif "logreg_cal" in probe_dir or probe_path.name == "probe_logreg_cal":
        probe_type = "logreg_cal"
    elif "logreg" in probe_dir or probe_path.name == "probe_logreg":
        probe_type = "logreg"
    elif "mlp" in probe_dir or probe_path.name == "probe_mlp":
        probe_type = "mlp"
    elif "tree" in probe_dir or probe_path.name == "probe_tree":
        probe_type = "tree"
    else:
        # Try to infer from files
        if (probe_path / "probe_model.pt").exists():
            probe_type = "xform"
        else:
            probe_type = "mlp"  # Default
    
    print(f"[Probe] Loading {probe_type} probe from {probe_dir}")
    
    if probe_type == "xform":
        # Load transformer probe
        checkpoint = torch.load(probe_path / "probe_model.pt", map_location="cpu")
        arch = checkpoint.get("arch", {})
        
        model = TinyTransformerProbe(
            use_hidden=use_hidden,
            d_model=arch.get("d_model", 256),
            nhead=arch.get("nhead", 8),
            num_layers=arch.get("num_layers", 3),
            d_ff=arch.get("d_ff", 1024),
            p_drop=arch.get("p_drop", 0.1),
            p_stoch=arch.get("p_stoch", 0.05),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Load calibration if present
        cal_blob = checkpoint.get("calibration", None)
        
        return {
            "type": probe_type,
            "model": model,
            "calibration": cal_blob,
            "use_hidden": use_hidden
        }
    else:
        # Load sklearn probe
        model_file = probe_path / "probe_model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Probe model not found at {model_file}")
        
        model = joblib.load(model_file)
        
        return {
            "type": probe_type,
            "model": model,
            "calibration": None,
            "use_hidden": use_hidden
        }

def main():
    parser = argparse.ArgumentParser(description="LLM inference with probe predictions - Stage 3")
    parser.add_argument("--model", choices=["llama31", "qwen"], required=True)
    parser.add_argument("--probe_dir", type=str, help="Path to probe directory")
    parser.add_argument("--use_hidden", action="store_true", help="Use hidden states")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--interactive", action="store_true", help="Interactive question mode")
    args = parser.parse_args()
    
    print(f"\n=== Stage 3: Full Pipeline - Model + Probe ===")
    print(f"Model: {args.model}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Use hidden states: {args.use_hidden}")
    
    # Initialize model
    model = initialize_model(args.model, args.max_new_tokens)
    
    # Load probe if specified
    probe = None
    if args.probe_dir:
        try:
            probe = load_probe(args.probe_dir, args.use_hidden)
            print(f"[Probe] Successfully loaded {probe['type']} probe")
            print(f"[Probe] Use hidden states: {probe['use_hidden']}")
        except Exception as e:
            print(f"[ERROR] Failed to load probe: {e}")
            print("Continuing without probe...")
    else:
        print("[Warning] No probe directory specified. Running model only.")
    
    if args.interactive:
        print("\nEntering interactive mode. Type 'quit' to exit.\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break
            
            if not question:
                continue
            
            # Run inference
            result = run_inference(model, question)
            
            # Extract features and run probe if available
            probe_prob = None
            if probe:
                try:
                    features = extract_features(model, result, args.use_hidden)
                    probe_prob = run_probe_inference(probe, features)
                except Exception as e:
                    print(f"[Warning] Probe inference failed: {e}")
            
            # Display results
            print(f"\n--- Results ---")
            print(f"Answer: {result['answer']}")
            print(f"Model confidence: {result['model_confidence']}")
            if probe_prob is not None:
                print(f"Probe P(correct): {probe_prob:.3f}")
                # Compare with model confidence
                if result['model_confidence'] is not None:
                    diff = probe_prob - result['model_confidence']
                    if abs(diff) > 0.2:
                        if diff > 0:
                            print(f"  → Probe is MORE confident (+{diff:.2f})")
                        else:
                            print(f"  → Probe is LESS confident ({diff:.2f})")
            print(f"Raw output: {result['raw_output'][:200]}...")
            print(f"Tokens generated: {result['gen_tokens']}")
            print(f"Time: {result['gen_time']:.3f}s")
            print()
    else:
        # Test with example questions
        test_questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?", 
            "What year did World War II end?",
            "What is the smallest country in the world?",
            "Who was the first person to walk on the moon?",
            "What is the chemical symbol for gold?",
            "How many continents are there?",
            "Who invented the telephone?",
        ]
        
        print("\nTesting with example questions:\n")
        print(f"{'Question':<50} {'Answer':<25} {'Model Conf':<12} {'Probe P(correct)':<15}")
        print("-" * 102)
        
        for q in test_questions:
            result = run_inference(model, q)
            
            # Extract features and run probe if available
            probe_prob = None
            if probe:
                try:
                    features = extract_features(model, result, args.use_hidden)
                    probe_prob = run_probe_inference(probe, features)
                except Exception as e:
                    print(f"[Warning] Probe inference failed for '{q[:30]}...': {e}")
            
            # Format output
            q_short = q[:47] + "..." if len(q) > 50 else q
            a_short = result['answer'][:22] + "..." if len(result['answer']) > 25 else result['answer']
            model_conf = f"{result['model_confidence']:.2f}" if result['model_confidence'] else "None"
            probe_str = f"{probe_prob:.3f}" if probe_prob is not None else "N/A"
            
            print(f"{q_short:<50} {a_short:<25} {model_conf:<12} {probe_str:<15}")
        
        print("\n" + "=" * 102)
        print("\nNote: Probe P(correct) estimates the probability that the model's answer is actually correct,")
        print("based on internal model states and behavioral patterns learned from training data.")

if __name__ == "__main__":
    main()
