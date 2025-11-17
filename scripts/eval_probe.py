#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate probe performance across all datasets' test splits

Usage:
    python -m scripts.eval_probe \
        --model llama31 \
        --probe_dir backend_llama/probe_mlp \
        --use_hidden \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --limit 100

This script:
1. Loads a trained probe
2. Runs inference on test splits of specified datasets
3. Compares probe predictions (using optimal threshold) to EM
4. Reports accuracy, precision, recall, F1 for each dataset
"""


"""
# Llama-3.1-8B with MLP probe on all datasets
python -m scripts.eval_probe --model llama31 --probe_dir backend_llama/probe_mlp --use_hidden --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu

# Quick test with 50 examples per dataset
python -m scripts.eval_probe --model llama31 --probe_dir backend_llama/probe_mlp --use_hidden --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu --limit 50

# Qwen with MLP probe
python -m scripts.eval_probe --model qwen --probe_dir backend_qwen/probe_mlp --use_hidden --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu
"""

import argparse
import json
import os
import torch
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

# Model imports
from src.models.qwen7b import Qwen7B
from src.models.llama31_8b import Llama31_8B
from src.data.datasets import load_qa, squad_em, squad_f1

# Speed optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Optimal F1 thresholds from training
OPTIMAL_THRESHOLDS = {
    "llama31": {
        "mlp": 0.359,
        "logreg": 0.332,
        "logreg_cal": 0.368,
        "tree": 0.179,
        "xform": 0.227,
    },
    "qwen": {
        "mlp": 0.353,
        "logreg": 0.293,
        "logreg_cal": 0.372,
        "tree": 0.228,
        "xform": 0.264,
    }
}

# Dataset test split mapping
DATASET_TEST_SPLITS = {
    "triviaqa": "test",
    "hotpotqa": "test",
    "hotpot_qa": "test",
    "squad_v2": "test",
    "squad2": "test",
    "gsm8k": "test",
    "mmlu": "test",
    "nq_open": "test",
}

# Prompts (same as collect_internals.py)
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

def safe_json_extract(text: str):
    t = text.strip()
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
                    if p > 1.0:
                        p = p / 100.0
                    p = max(0.0, min(1.0, p))
                    break
                except Exception:
                    pass
        return ans, p
    return text.strip(), None

def initialize_model(model_name: str, max_new_tokens: int = 64):
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
    
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")
    
    return model

def pack_256(vec: torch.Tensor):
    if vec is None:
        return None
    v = torch.nn.functional.normalize(vec.float(), dim=-1)
    v = v[:256] if v.shape[-1] >= 256 else torch.nn.functional.pad(v, (0, 256 - v.shape[-1]))
    return [float(x) for x in v.cpu()]

def rescore_mean_logprob(model, tok, prompt_text: str, target_text: str):
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

def extract_features(model, inp, gen, new_ids, prompt, answer, p_model, use_hidden=True):
    import math
    
    device = next(model.model.parameters()).device
    tok = model.tok
    hf_causal_lm = model.model
    
    # Token-level statistics
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
    entropy_std = float(torch.tensor(step_ent[:content_len]).std().item()) if len(step_ent[:content_len]) > 1 else None
    margin_mean = float(sum(margins[:content_len]) / content_len) if margins else None
    margin_min = float(min(margins[:content_len])) if content_len > 0 and margins else None
    lp_mean = float(sum(step_logp[:content_len]) / content_len) if step_logp else None
    seq_conf = math.exp(lp_mean) if lp_mean is not None else None
    
    # Hidden states
    h_last_256 = None
    h_pool_256 = None
    h_last_mid_256 = None
    h_pool_mid_256 = None
    
    if use_hidden:
        with torch.no_grad():
            full_ids = torch.cat([inp.to(device), new_ids.to(device)], dim=-1)
            attn = torch.ones_like(full_ids)
            out = hf_causal_lm(input_ids=full_ids, attention_mask=attn,
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
    
    # Rescore
    safe_ans = answer.replace('"', '\\"')
    pt = p_model if p_model is not None else 0.00
    canon_json = f'{{"answer":"{safe_ans}","p_true":{pt:.2f}}}'
    rescore_lp = rescore_mean_logprob(hf_causal_lm, tok, prompt, canon_json)
    
    features = {
        "model_confidence": p_model,
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
        "h_last_256": h_last_256,
        "h_pool_256": h_pool_256,
        "h_last_mid_256": h_last_mid_256,
        "h_pool_mid_256": h_pool_mid_256,
    }
    
    return features

def run_probe_inference(probe, features):
    probe_type = probe["type"]
    model = probe["model"]
    use_hidden = probe["use_hidden"]
    
    SCALAR_KEYS = [
        "model_confidence", "lp_mean", "seq_conf",
        "entropy_mean", "entropy_std",
        "margin_mean", "margin_min",
        "rescore_logp", "answer_len",
        "parsed_json_ok", "parsed_p_true_ok", "is_unknown",
    ]
    VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]
    
    feat_dict = {}
    for k in SCALAR_KEYS:
        feat_dict[k] = features.get(k, np.nan)
    
    if use_hidden:
        for name in VECTOR_KEYS:
            vec = features.get(name)
            if vec is not None:
                for j, v in enumerate(vec):
                    feat_dict[f"{name}_{j}"] = float(v)
    
    if probe_type == "xform":
        # Would need transformer probe class - skip for now
        raise NotImplementedError("Transformer probe not implemented in this script")
    else:
        # Sklearn probe
        df = pd.DataFrame([feat_dict])
        prob = model.predict_proba(df.values)[:, 1][0]
        return float(prob)

def load_probe(probe_dir: str, use_hidden: bool = True, model_name: str = None):
    probe_path = Path(probe_dir)
    
    # Determine probe type
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
        probe_type = "mlp"
    
    print(f"[Probe] Loading {probe_type} probe from {probe_dir}")
    
    # Get threshold
    threshold = None
    if model_name and model_name in OPTIMAL_THRESHOLDS:
        threshold = OPTIMAL_THRESHOLDS[model_name].get(probe_type)
        if threshold:
            print(f"[Probe] Using optimal F1 threshold: {threshold:.3f}")
    
    if probe_type == "xform":
        raise NotImplementedError("Transformer probe evaluation not implemented")
    else:
        model_file = probe_path / "probe_model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Probe model not found at {model_file}")
        
        model = joblib.load(model_file)
        
        return {
            "type": probe_type,
            "model": model,
            "use_hidden": use_hidden,
            "threshold": threshold,
        }

def evaluate_on_dataset(model, probe, dataset_name: str, limit: int = None):
    """Evaluate probe on a single dataset"""
    
    # Get test split
    split = DATASET_TEST_SPLITS.get(dataset_name, "test")
    
    print(f"\n[{dataset_name}] Loading test split: {split}")
    try:
        ds = load_qa(dataset_name, split, limit)
    except Exception as e:
        print(f"[ERROR] Failed to load {dataset_name}: {e}")
        return None
    
    print(f"[{dataset_name}] Loaded {len(ds)} examples")
    
    threshold = probe.get("threshold", 0.5)
    use_hidden = probe.get("use_hidden", True)
    
    results = []
    
    for idx, ex in enumerate(tqdm(ds, desc=f"Evaluating {dataset_name}")):
        question = ex["question"]
        gold_answers = ex["answers"]
        
        # Generate answer
        prompt = FEWSHOT_PROMPT.format(EVAL_QUESTION=question)
        inp, gen = model.generate_with_states(prompt)
        
        start_pos = inp["input_ids"].shape[-1]
        new_ids = gen.sequences[:, start_pos:]
        raw_text = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
        
        answer, p_model = extract_answer_and_prob(raw_text)
        if not answer:
            answer = raw_text
        
        # Compute EM
        em = 1 if squad_em(answer, gold_answers) else 0
        
        # Extract features and run probe
        try:
            features = extract_features(model, inp["input_ids"], gen, new_ids, 
                                       prompt, answer, p_model, use_hidden)
            probe_prob = run_probe_inference(probe, features)
            probe_pred = 1 if probe_prob > threshold else 0
        except Exception as e:
            print(f"[WARNING] Probe failed on example {idx}: {e}")
            probe_prob = None
            probe_pred = None
        
        results.append({
            "question": question,
            "answer": answer,
            "gold_answers": gold_answers,
            "em": em,
            "probe_prob": probe_prob,
            "probe_pred": probe_pred,
        })
    
    # Compute metrics
    valid_results = [r for r in results if r["probe_pred"] is not None]
    
    if not valid_results:
        print(f"[{dataset_name}] No valid probe predictions!")
        return None
    
    y_true = [r["em"] for r in valid_results]
    y_pred = [r["probe_pred"] for r in valid_results]
    
    # Compute confusion matrix elements
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Model accuracy (EM)
    model_accuracy = sum(y_true) / len(y_true) if len(y_true) > 0 else 0.0
    
    metrics = {
        "dataset": dataset_name,
        "n_examples": len(valid_results),
        "model_accuracy": model_accuracy,
        "probe_accuracy": accuracy,
        "probe_precision": precision,
        "probe_recall": recall,
        "probe_f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate probe on test splits")
    parser.add_argument("--model", choices=["llama31", "qwen"], required=True)
    parser.add_argument("--probe_dir", type=str, required=True,
                       help="Path to probe directory")
    parser.add_argument("--use_hidden", action="store_true",
                       help="Use hidden states (must match probe training)")
    parser.add_argument("--datasets", type=str, 
                       default="triviaqa,hotpotqa,squad_v2,gsm8k,mmlu",
                       help="Comma-separated list of datasets")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit examples per dataset (for testing)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    args = parser.parse_args()
    
    print("=" * 80)
    print("PROBE EVALUATION ON TEST SPLITS")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Probe: {args.probe_dir}")
    print(f"Use hidden states: {args.use_hidden}")
    print(f"Datasets: {args.datasets}")
    if args.limit:
        print(f"Limit: {args.limit} examples per dataset")
    print("=" * 80)
    
    # Initialize model
    model = initialize_model(args.model, args.max_new_tokens)
    
    # Load probe
    probe = load_probe(args.probe_dir, args.use_hidden, model_name=args.model)
    print(f"[Probe] Type: {probe['type']}")
    print(f"[Probe] Threshold: {probe.get('threshold', 0.5):.3f}")
    
    # Evaluate on each dataset
    datasets = [d.strip() for d in args.datasets.split(",")]
    all_metrics = []
    
    for dataset_name in datasets:
        metrics = evaluate_on_dataset(model, probe, dataset_name, args.limit)
        if metrics:
            all_metrics.append(metrics)
    
    # Print summary table using tabulate
    from tabulate import tabulate
    
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    # Build table data
    table_data = []
    for m in all_metrics:
        table_data.append([
            m['dataset'],
            m['n_examples'],
            f"{m['model_accuracy']:.3f}",
            f"{m['probe_accuracy']:.3f}",
            f"{m['probe_precision']:.3f}",
            f"{m['probe_recall']:.3f}",
            f"{m['probe_f1']:.3f}",
        ])
    
    # Add average row if multiple datasets
    if len(all_metrics) > 1:
        avg_model_acc = np.mean([m['model_accuracy'] for m in all_metrics])
        avg_probe_acc = np.mean([m['probe_accuracy'] for m in all_metrics])
        avg_precision = np.mean([m['probe_precision'] for m in all_metrics])
        avg_recall = np.mean([m['probe_recall'] for m in all_metrics])
        avg_f1 = np.mean([m['probe_f1'] for m in all_metrics])
        
        table_data.append([
            "─" * 15,  # separator
            "─" * 6,
            "─" * 7,
            "─" * 7,
            "─" * 7,
            "─" * 7,
            "─" * 7,
        ])
        table_data.append([
            "AVERAGE",
            "",
            f"{avg_model_acc:.3f}",
            f"{avg_probe_acc:.3f}",
            f"{avg_precision:.3f}",
            f"{avg_recall:.3f}",
            f"{avg_f1:.3f}",
        ])
    
    headers = ["Dataset", "N", "Model Acc", "Probe Acc", "Precision", "Recall", "F1"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 100)
    
    # Print detailed confusion matrices
    print("\n" + "=" * 100)
    print("CONFUSION MATRICES (per dataset)")
    print("=" * 100)
    
    for m in all_metrics:
        print(f"\n{m['dataset'].upper()} (threshold={m['threshold']:.3f}):")
        cm_data = [
            ["", "Pred: Correct", "Pred: Wrong"],
            [f"True: Correct", f"{m['tp']} (TP)", f"{m['fn']} (FN)"],
            [f"True: Wrong", f"{m['fp']} (FP)", f"{m['tn']} (TN)"],
        ]
        print(tabulate(cm_data, tablefmt="grid"))
    
    print("=" * 100)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "model": args.model,
                "probe_dir": args.probe_dir,
                "use_hidden": args.use_hidden,
                "datasets": all_metrics,
            }, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()