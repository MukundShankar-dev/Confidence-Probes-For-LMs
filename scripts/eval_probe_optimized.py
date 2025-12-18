#!/usr/bin/env python3
"""
Evaluation script that EXACTLY matches training pipeline
Loads datasets, runs inference, evaluates probes, creates visualizations
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, precision_recall_curve, roc_curve,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.qwen7b import Qwen7B
from src.models.llama31_8b import Llama31_8B
from src.data.datasets import load_qa, squad_em

console = Console()
sns.set_style("whitegrid")

# EXACT feature definitions from training script
SCALAR_KEYS = [
    "entropy_mean", "entropy_std",
    "margin_mean", "margin_min",
    "lp_mean", "seq_conf",
    "answer_len", "is_unknown",
]

VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]

SUPER_GENERALIZABLE_FEATURES = {
    "entropy_mean", "entropy_std",
    "margin_mean", "margin_min",
    "lp_mean", "seq_conf",
}

DATASET_SPLITS = {
    "triviaqa": "validation",
    "hotpotqa": "validation",
    "squad_v2": "validation",
    "gsm8k": "test",
    "mmlu": "validation"
}

PROMPTS = {
    "triviaqa": "Answer the following question with a short factual answer (1-5 words).\n\nQuestion: {question}\nAnswer:",
    "hotpotqa": "Answer the following question with a short factual answer (1-5 words).\n\nQuestion: {question}\nAnswer:",
    "squad_v2": "Answer the following question based on the context. If unanswerable, respond with 'Unanswerable'.\n\nQuestion: {question}\nAnswer:",
    "gsm8k": "Solve this math problem step by step. Provide only the final numerical answer.\n\nQuestion: {question}\nAnswer:",
    "mmlu": "Answer the following multiple choice question.\n\n{question}\nAnswer:"
}


def filter_scalar_keys(only_generalizable=False, super_generalizable=False):
    """EXACT copy from training script"""
    if super_generalizable or only_generalizable:
        return [k for k in SCALAR_KEYS if k in SUPER_GENERALIZABLE_FEATURES]
    else:
        return SCALAR_KEYS


def to_1d(arr):
    """EXACT copy from training script"""
    if arr is None:
        return None
    a = np.array(arr, dtype=float).ravel()
    if a.size < 256:
        a = np.pad(a, (0, 256 - a.size))
    elif a.size > 256:
        a = a[:256]
    return a


def pack_256(vec):
    """Pack hidden state to 256 dims"""
    if vec is None:
        return None
    if vec.shape[0] <= 256:
        padded = torch.zeros(256, device=vec.device, dtype=vec.dtype)
        padded[:vec.shape[0]] = vec
        return padded.cpu().numpy().tolist()
    else:
        stride = vec.shape[0] // 256
        pooled = torch.nn.functional.avg_pool1d(
            vec.unsqueeze(0).unsqueeze(0),
            kernel_size=stride,
            stride=stride
        ).squeeze()
        if pooled.shape[0] > 256:
            pooled = pooled[:256]
        elif pooled.shape[0] < 256:
            padded = torch.zeros(256, device=vec.device, dtype=vec.dtype)
            padded[:pooled.shape[0]] = pooled
            pooled = padded
        return pooled.cpu().numpy().tolist()


def compute_entropy(probs):
    probs = torch.clamp(probs, min=1e-10)
    return -torch.sum(probs * torch.log(probs)).item()


def compute_margin(logits):
    top2 = torch.topk(logits, k=2).values
    return (top2[0] - top2[1]).item()


def collect_features(model, question, dataset_name):
    """Collect features - matching collection script approach"""
    prompt_template = PROMPTS.get(dataset_name, PROMPTS["triviaqa"])
    prompt = prompt_template.format(question=question)
    
    # Generate
    inp, gen = model.generate_with_states(prompt)
    
    # Extract answer
    start_pos = inp["input_ids"].shape[-1]
    new_ids = gen.sequences[:, start_pos:]
    generated_text = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
    
    # Get hidden states
    device = next(model.model.parameters()).device
    with torch.no_grad():
        full_ids = torch.cat([inp["input_ids"].to(device), new_ids.to(device)], dim=-1)
        attn_mask = torch.ones_like(full_ids)
        out = model.model(
            input_ids=full_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )
        
        gen_len = new_ids.shape[-1]
        
        # Last layer
        hs_final = out.hidden_states[-1][0]
        h_ans = hs_final[-gen_len:]
        h_last = h_ans[-1] if h_ans.shape[0] > 0 else None
        h_pool = h_ans.mean(dim=0) if h_ans.shape[0] > 0 else None
        
        # Middle layer
        mid_ix = len(out.hidden_states) // 2
        hs_mid = out.hidden_states[mid_ix][0]
        h_ans_mid = hs_mid[-gen_len:]
        h_last_mid = h_ans_mid[-1] if h_ans_mid.shape[0] > 0 else None
        h_pool_mid = h_ans_mid.mean(dim=0) if h_ans_mid.shape[0] > 0 else None
    
    # Pack hidden states
    h_last_256 = pack_256(h_last) if h_last is not None else None
    h_pool_256 = pack_256(h_pool) if h_pool is not None else None
    h_last_mid_256 = pack_256(h_last_mid) if h_last_mid is not None else None
    h_pool_mid_256 = pack_256(h_pool_mid) if h_pool_mid is not None else None
    
    # Compute probability features
    scores = gen.scores if hasattr(gen, 'scores') else []
    if len(scores) > 0:
        all_logits = torch.stack([s[0] for s in scores], dim=0)
        all_probs = torch.softmax(all_logits, dim=-1)
        
        entropies, margins, lps, top_probs = [], [], [], []
        for i, (logit_row, prob_row) in enumerate(zip(all_logits, all_probs)):
            entropies.append(compute_entropy(prob_row))
            margins.append(compute_margin(logit_row))
            chosen_id = new_ids[0, i].item()
            lps.append(float(torch.log(prob_row[chosen_id] + 1e-10)))
            top_probs.append(float(prob_row.max()))
        
        entropy_mean = float(torch.tensor(entropies).mean())
        entropy_std = float(torch.tensor(entropies).std())
        margin_mean = float(torch.tensor(margins).mean())
        margin_min = float(torch.tensor(margins).min())
        lp_mean = float(torch.tensor(lps).mean())
        seq_conf = float(torch.tensor(top_probs).mean())
    else:
        entropy_mean = entropy_std = margin_mean = margin_min = lp_mean = seq_conf = 0.0
    
    return {
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "margin_mean": margin_mean,
        "margin_min": margin_min,
        "lp_mean": lp_mean,
        "seq_conf": seq_conf,
        "answer_len": len(generated_text.split()),
        "is_unknown": generated_text.lower() in ("unknown", "unanswerable"),
        "h_last_256": h_last_256,
        "h_pool_256": h_pool_256,
        "h_last_mid_256": h_last_mid_256,
        "h_pool_mid_256": h_pool_mid_256,
        "parsed_answer": generated_text,
    }


def build_feature_row(features, use_hidden, only_generalizable, super_generalizable):
    """Build feature row EXACTLY like training script"""
    # Filter scalars
    scalar_keys = filter_scalar_keys(only_generalizable, super_generalizable)
    
    # Override hidden if super_generalizable
    if super_generalizable:
        use_hidden = False
    
    # Build feature dict
    feat_dict = {}
    for k in scalar_keys:
        feat_dict[k] = features.get(k, np.nan)
    
    if use_hidden:
        for vec_name in VECTOR_KEYS:
            vec = to_1d(features.get(vec_name))
            if vec is not None:
                for j, v in enumerate(vec):
                    feat_dict[f"{vec_name}_{j}"] = float(v)
            else:
                for j in range(256):
                    feat_dict[f"{vec_name}_{j}"] = np.nan
    
    return feat_dict


def plot_results(results_df, output_dir, dataset_name):
    """Create visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = results_df['y_true'].values
    y_pred = results_df['y_pred'].values
    p_correct = results_df['p_correct'].values
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, p_correct)
    auc_roc = roc_auc_score(y_true, p_correct)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_roc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_{dataset_name}.png', dpi=150)
    plt.close()
    
    # PR
    prec, rec, _ = precision_recall_curve(y_true, p_correct)
    auc_pr = average_precision_score(y_true, p_correct)
    
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, linewidth=2, label=f'AP = {auc_pr:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {dataset_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'pr_{dataset_name}.png', dpi=150)
    plt.close()
    
    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, p_correct, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect')
    plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Frequency')
    plt.title(f'Calibration - {dataset_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'calibration_{dataset_name}.png', dpi=150)
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=['Incorrect', 'Correct'],
                yticklabels=['Incorrect', 'Correct'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_{dataset_name}.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "llama"])
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--probe_dir", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    probe_dir = Path(args.probe_dir)
    output_dir = Path(args.output_dir)
    datasets = args.datasets.split(",")
    
    console.print(Panel.fit(
        f"[bold cyan]Probe Evaluation[/bold cyan]\n"
        f"Model: {args.model_id}\n"
        f"Probe: {probe_dir.name}\n"
        f"Datasets: {', '.join(datasets)}",
        border_style="cyan"
    ))
    
    # Load model
    console.print("\n[bold]Loading Model[/bold]")
    if args.model == "qwen":
        model = Qwen7B(args.model_id, dtype="float16", device_map="auto")
    else:
        model = Llama31_8B(args.model_id, dtype="float16", device_map="auto")
    console.print(f"  [green]✓[/green] Loaded")
    
    # Load probe + metadata
    console.print("\n[bold]Loading Probe[/bold]")
    probe = joblib.load(probe_dir / "probe_model.joblib")
    
    metadata_path = probe_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        use_hidden = metadata.get("use_hidden", True)
        only_generalizable = "generalizable" in probe_dir.name.lower()
        super_generalizable = metadata.get("super_generalizable", False)
        optimal_threshold = metadata.get("optimal_threshold", 0.5)
    else:
        use_hidden = "super_gen" not in probe_dir.name
        only_generalizable = "generalizable" in probe_dir.name.lower()
        super_generalizable = "super_gen" in probe_dir.name
        optimal_threshold = 0.5
    
    console.print(f"  Use hidden: {use_hidden}")
    console.print(f"  Only generalizable: {only_generalizable}")
    console.print(f"  Super generalizable: {super_generalizable}")
    console.print(f"  [cyan]Threshold: {optimal_threshold:.3f}[/cyan]")
    
    # DEBUG: Check what features will be created
    scalar_keys = filter_scalar_keys(only_generalizable, super_generalizable)
    _use_hidden = False if super_generalizable else use_hidden
    n_scalar = len(scalar_keys)
    n_vector = 4 * 256 if _use_hidden else 0
    n_total = n_scalar + n_vector
    console.print(f"  [yellow]DEBUG: Will create {n_total} features ({n_scalar} scalars + {n_vector} hidden)[/yellow]")
    console.print(f"  [yellow]DEBUG: Scalar keys: {scalar_keys}[/yellow]")
    
    # Evaluate each dataset
    all_results = []
    
    for dataset_name in datasets:
        console.print(f"\n[bold cyan]Dataset: {dataset_name}[/bold cyan]")
        
        split = DATASET_SPLITS.get(dataset_name, "validation")
        ds = load_qa(dataset_name, split, limit=args.limit)
        console.print(f"  Loaded {len(ds):,} examples")
        
        predictions = []
        for ex in tqdm(ds, desc=f"  Processing"):
            question = ex["question"]
            gold_answers = ex["answers"]
            
            # Collect features
            features = collect_features(model, question, dataset_name)
            
            # Check correctness
            parsed_answer = features["parsed_answer"]
            correct = any(str(g).lower() in parsed_answer.lower() for g in gold_answers if parsed_answer)
            
            # Build feature vector
            feat_dict = build_feature_row(features, use_hidden, only_generalizable, super_generalizable)
            df = pd.DataFrame([feat_dict])
            
            # Predict
            if hasattr(probe, "predict_proba"):
                p_correct = probe.predict_proba(df.values)[:, 1][0]
            else:
                p_correct = probe.predict(df.values)[0]
            
            predictions.append({
                "question": question,
                "pred_answer": parsed_answer,
                "y_true": int(correct),
                "p_correct": float(p_correct),
            })
        
        # Results
        results_df = pd.DataFrame(predictions)
        y_true = results_df['y_true'].values
        p_correct = results_df['p_correct'].values
        y_pred = (p_correct >= optimal_threshold).astype(int)
        results_df['y_pred'] = y_pred
        
        # Metrics
        metrics = {
            "dataset": dataset_name,
            "n": len(y_true),
            "threshold": float(optimal_threshold),
            "model_accuracy": float(y_true.mean()),
            "probe_accuracy": float(accuracy_score(y_true, y_pred)),
            "probe_precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "probe_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "probe_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, p_correct)),
            "auc_pr": float(average_precision_score(y_true, p_correct)),
            "brier": float(brier_score_loss(y_true, p_correct)),
        }
        all_results.append(metrics)
        
        console.print(f"\n  [bold]Model Acc:[/bold] {metrics['model_accuracy']:.3f}")
        console.print(f"  [bold]Probe Acc:[/bold] {metrics['probe_accuracy']:.3f}")
        console.print(f"  [bold]F1:[/bold] {metrics['probe_f1']:.3f}")
        console.print(f"  [bold]AUC-ROC:[/bold] {metrics['auc_roc']:.3f}")
        
        # Save
        dataset_output = output_dir / dataset_name
        results_df.to_csv(dataset_output / "predictions.csv", index=False)
        plot_results(results_df, dataset_output, dataset_name)
    
    # Summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    console.print("\n[bold cyan]Results Summary[/bold cyan]")
    table = Table(show_header=True)
    table.add_column("Dataset")
    table.add_column("N", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("LM Acc", justify="right")
    table.add_column("Probe Acc", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("AUC-ROC", justify="right")
    
    for r in all_results:
        table.add_row(
            r["dataset"],
            f"{r['n']:,}",
            f"{r['threshold']:.3f}",
            f"{r['model_accuracy']:.3f}",
            f"{r['probe_accuracy']:.3f}",
            f"{r['probe_f1']:.3f}",
            f"{r['auc_roc']:.3f}"
        )
    
    console.print(table)
    console.print(f"\n[green]✓ Complete! Results in {output_dir}[/green]")


if __name__ == "__main__":
    main()