"""
Efficient cross-dataset probe evaluation

Key idea:
- For each dataset, do one rollout per example (generate + internal feature extraction).
- Reuse the same extracted feature dict to score all probes (4 transfer probes),
  rebuilding X per probe using that probe's feature_names.json.

Output structure:
  output_dir/
    <eval_dataset>/
      <probe_train_dataset>/
        predictions.csv
        roc_<eval_dataset>.png
        pr_<eval_dataset>.png
        confusion_<eval_dataset>.png
        metrics.json
"""

import argparse
from typing import Optional, List, Dict, Tuple
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
)

import matplotlib.pyplot as plt

from src.models.qwen7b import Qwen7B
from src.models.llama31_8b import Llama31_8B
from src.data.datasets import load_qa, squad_em

console = Console()

DATASET_SPLITS = {
    "triviaqa": "validation",
    "hotpotqa": "validation",
    "squad_v2": "validation",
    "squadv2": "validation",
    "gsm8k": "test",
    "mmlu": "validation",
}

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

PROMPT_OPEN_QA = """Answer the following question with a short factual answer (1-5 words).

Q: Who wrote Hamlet?
A: William Shakespeare

Q: What is the capital of France?
A: Paris

Q: Which planet is known as the Red Planet?
A: Mars

Q: {QUESTION}
A:"""

PROMPT_MMLU = """Answer the following multiple choice question by outputting only the letter (A, B, C, or D) of the correct answer.

Question: What is the capital of France?
A. London
B. Berlin
C. Paris
D. Madrid
Answer: C

Question: Who wrote Romeo and Juliet?
A. Charles Dickens
B. William Shakespeare
C. Mark Twain
D. Jane Austen
Answer: B

{QUESTION}
Answer:"""

PROMPT_GSM8K = """Solve the following math problem. Show your reasoning and then provide the final numerical answer.

Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
A: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast, so 16 - 3 = 13 eggs remain. She uses 4 for muffins, so 13 - 4 = 9 eggs remain. She sells these 9 eggs for $2 each, so 9 * 2 = $18. The answer is 18.

Q: {QUESTION}
A:"""

PROMPT_SQUADV2 = """Answer the question based on the context. If the question cannot be answered based on the context, respond with "unanswerable".

Context: The Amazon rainforest is a moist broadleaf forest in South America. The majority of the forest is in Brazil.
Question: Which country contains most of the Amazon rainforest?
Answer: Brazil

Context: Super Bowl 50 was held on February 7, 2016 at Levi's Stadium in Santa Clara, California.
Question: Where was Super Bowl 50 held?
Answer: Levi's Stadium

Context: {CONTEXT}
Question: {QUESTION}
Answer:"""

def build_prompt(question: str, dataset: str, context: Optional[str] = None) -> str:
    d = dataset.lower()
    if d == "mmlu":
        return PROMPT_MMLU.format(QUESTION=question)
    if d == "gsm8k":
        return PROMPT_GSM8K.format(QUESTION=question)
    if d in ("squadv2", "squad_v2", "squad2"):
        return PROMPT_SQUADV2.format(CONTEXT=context or "", QUESTION=question)
    return PROMPT_OPEN_QA.format(QUESTION=question)

def extract_answer_standard(text: str, dataset: str) -> str:
    text = (text or "").strip()
    d = dataset.lower()

    if d == "mmlu":
        m = re.search(r"\b([A-D])\b", text)
        if m:
            return m.group(1)
        return text.split()[0] if text else ""

    if d == "gsm8k":
        m = re.search(r"(?:answer is|equals?)\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        return nums[-1] if nums else text

    if d in ("squadv2", "squad_v2", "squad2"):
        text = re.sub(r"^(?:Answer|A):\s*", "", text, flags=re.IGNORECASE)
        first = text.split("\n")[0].strip()
        return first[:100]

    text = re.sub(r"^(?:Answer|A):\s*", "", text, flags=re.IGNORECASE)
    first = text.split("\n")[0].strip()
    first = re.sub(r"[.!?]+$", "", first)
    return first[:100]

def pack_256(vec: Optional[torch.Tensor]):
    if vec is None:
        return None
    v = torch.nn.functional.normalize(vec.float(), dim=-1)
    if v.shape[-1] >= 256:
        v = v[:256]
    else:
        v = torch.nn.functional.pad(v, (0, 256 - v.shape[-1]))
    return [float(x) for x in v.detach().cpu()]

def compute_entropy(probs: torch.Tensor) -> float:
    if probs.numel() == 0:
        return 0.0
    eps = 1e-10
    p = torch.clamp(probs, min=eps)
    return float(-torch.sum(p * torch.log(p)))

def compute_margin(logits_flat: torch.Tensor) -> float:
    if logits_flat.numel() < 2:
        return 0.0
    top2 = torch.topk(logits_flat, k=2, largest=True).values
    return float(top2[0] - top2[1])

def collect_features(model, question: str, dataset: str, context: Optional[str] = None):
    """
    ONE rollout per example:
    - generate via model.generate_with_states(prompt)
    - logit stats from gen.scores
    - teacher-forced forward over (prompt + generated) for hidden states
    """
    prompt = build_prompt(question, dataset, context)

    try:
        result = model.generate_with_states(prompt)
        if result is None:
            raise RuntimeError("generate_with_states returned None")
        inp, gen = result
    except Exception as e:
        raise RuntimeError(f"generate_with_states failed: {e}\nPrompt head: {prompt[:200]}...")

    start_pos = inp["input_ids"].shape[-1]
    new_ids = gen.sequences[:, start_pos:]
    generated_text = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
    answer = extract_answer_standard(generated_text, dataset)

    device = next(model.model.parameters()).device
    with torch.no_grad():
        full_ids = torch.cat([inp["input_ids"].to(device), new_ids.to(device)], dim=-1)
        attn_mask = torch.ones_like(full_ids)

        out = model.model(
            input_ids=full_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        gen_len = new_ids.shape[-1]
        hs_final = out.hidden_states[-1][0]  # [T, d]
        h_ans = hs_final[-gen_len:]
        h_last = h_ans[-1] if h_ans.shape[0] > 0 else None
        h_pool = h_ans.mean(dim=0) if h_ans.shape[0] > 0 else None

        mid_ix = len(out.hidden_states) // 2
        hs_mid = out.hidden_states[mid_ix][0]
        h_ans_mid = hs_mid[-gen_len:]
        h_last_mid = h_ans_mid[-1] if h_ans_mid.shape[0] > 0 else None
        h_pool_mid = h_ans_mid.mean(dim=0) if h_ans_mid.shape[0] > 0 else None

    h_last_256 = pack_256(h_last)
    h_pool_256 = pack_256(h_pool)
    h_last_mid_256 = pack_256(h_last_mid)
    h_pool_mid_256 = pack_256(h_pool_mid)

    scores = getattr(gen, "scores", []) or []
    if len(scores) > 0:
        all_logits = torch.stack([s[0] for s in scores], dim=0)  # [L, vocab]
        all_probs = torch.softmax(all_logits, dim=-1)

        entropies, margins, lps, top_probs = [], [], [], []
        for i, (logit_row, prob_row) in enumerate(zip(all_logits, all_probs)):
            entropies.append(compute_entropy(prob_row))
            margins.append(compute_margin(logit_row))
            chosen_id = int(new_ids[0, i].item())
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
        "answer_len": len(answer.split()),
        "is_unknown": answer.lower() in ("unknown", "unanswerable"),
        "h_last_256": h_last_256,
        "h_pool_256": h_pool_256,
        "h_last_mid_256": h_last_mid_256,
        "h_pool_mid_256": h_pool_mid_256,
        "parsed_answer": answer,
        "raw_generation": generated_text,
    }

def filter_scalar_keys(only_generalizable: bool, super_generalizable: bool):
    if super_generalizable or only_generalizable:
        return [k for k in SCALAR_KEYS if k in SUPER_GENERALIZABLE_FEATURES]
    return list(SCALAR_KEYS)

def to_1d(arr):
    if arr is None:
        return None
    a = np.array(arr, dtype=float).ravel()
    if a.size < 256:
        a = np.pad(a, (0, 256 - a.size))
    elif a.size > 256:
        a = a[:256]
    return a

def build_feature_row_dict(features: dict,
                           use_hidden: bool,
                           only_generalizable: bool,
                           super_generalizable: bool) -> dict:
    if super_generalizable:
        use_hidden = False

    scalar_keys = filter_scalar_keys(only_generalizable, super_generalizable)
    feat = {}
    for k in scalar_keys:
        feat[k] = features.get(k, np.nan)

    if use_hidden:
        for vec_name in VECTOR_KEYS:
            vec = to_1d(features.get(vec_name))
            if vec is None:
                for j in range(256):
                    feat[f"{vec_name}_{j}"] = np.nan
            else:
                for j, v in enumerate(vec):
                    feat[f"{vec_name}_{j}"] = float(v)
    return feat

def infer_flags_from_feature_names(feature_names: list[str]):
    has_hidden = any(
        name.startswith("h_last_256_") or name.startswith("h_pool_256_") or
        name.startswith("h_last_mid_256_") or name.startswith("h_pool_mid_256_")
        for name in feature_names
    )
    has_answer_len = "answer_len" in feature_names
    has_is_unknown = "is_unknown" in feature_names

    only_generalizable = not (has_answer_len or has_is_unknown)
    super_generalizable = only_generalizable and (not has_hidden)
    use_hidden = has_hidden
    return use_hidden, only_generalizable, super_generalizable

def build_feature_dataframe(features: dict,
                            feature_names: List[str]) -> pd.DataFrame:
    use_hidden, only_generalizable, super_generalizable = infer_flags_from_feature_names(feature_names)
    row = build_feature_row_dict(features, use_hidden, only_generalizable, super_generalizable)
    df = pd.DataFrame([row])

    for c in feature_names:
        if c not in df.columns:
            df[c] = np.nan
    df = df[feature_names]
    return df

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def is_correct(dataset: str, pred_answer: str, gold_answers) -> bool:
    d = dataset.lower()
    pred = normalize_text(pred_answer)

    if gold_answers is None:
        return False

    if d == "mmlu":
        m = re.search(r"\b([A-D])\b", pred_answer.strip())
        p = m.group(1) if m else (pred_answer.strip()[:1].upper() if pred_answer else "")
        for g in gold_answers:
            if str(g).strip().upper() == p:
                return True
        return False

    if d == "gsm8k":
        def last_number(x):
            xs = re.findall(r"-?\d+(?:\.\d+)?", str(x))
            return xs[-1] if xs else None
        pnum = last_number(pred_answer)
        for g in gold_answers:
            gnum = last_number(g)
            if pnum is not None and gnum is not None and pnum == gnum:
                return True
        for g in gold_answers:
            if normalize_text(str(g)) in pred:
                return True
        return False

    if d in ("squadv2", "squad_v2", "squad2"):
        try:
            for g in gold_answers:
                if squad_em(pred_answer, str(g)) == 1:
                    return True
        except Exception:
            pass
        for g in gold_answers:
            if normalize_text(str(g)) in pred:
                return True
        return False

    for g in gold_answers:
        if normalize_text(str(g)) in pred:
            return True
    return False

def save_plots(results_df: pd.DataFrame, out_dir: Path, dataset_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = results_df["y_true"].values
    p = results_df["p_correct"].values

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, p)
        auc_roc = roc_auc_score(y_true, p)
    else:
        fpr, tpr, auc_roc = [0, 1], [0, 1], float("nan")

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {dataset_name} (AUC={auc_roc:.3f})" if auc_roc == auc_roc else f"ROC - {dataset_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_{dataset_name}.png", dpi=150)
    plt.close()

    if len(np.unique(y_true)) > 1:
        prec, rec, _ = precision_recall_curve(y_true, p)
        auc_pr = average_precision_score(y_true, p)
    else:
        prec, rec, auc_pr = [1, 0], [0, 1], float("nan")

    plt.figure(figsize=(7, 5))
    plt.plot(rec, prec, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {dataset_name} (AP={auc_pr:.3f})" if auc_pr == auc_pr else f"PR - {dataset_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"pr_{dataset_name}.png", dpi=150)
    plt.close()

    cm = confusion_matrix(results_df["y_true"].values, results_df["y_pred"].values)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm)
    plt.xticks([0, 1], ["Incorrect", "Correct"])
    plt.yticks([0, 1], ["Incorrect", "Correct"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion - {dataset_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_{dataset_name}.png", dpi=150)
    plt.close()

def load_probe(probe_dir: Path) -> Tuple[object, float, List[str]]:
    probe = joblib.load(probe_dir / "probe_model.joblib")

    metadata_path = probe_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    threshold = float(metadata.get("optimal_threshold", 0.5))

    fn_path = probe_dir / "feature_names.json"
    if not fn_path.exists():
        raise FileNotFoundError(f"Missing feature_names.json in {probe_dir}")
    feature_names = json.loads(fn_path.read_text())

    return probe, threshold, feature_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "llama"])
    parser.add_argument("--model_id", type=str, required=True)

    parser.add_argument("--datasets", type=str, required=True,
                        help="Comma-separated eval datasets (e.g. triviaqa,hotpotqa,squadv2,gsm8k,mmlu)")

    parser.add_argument("--probe_map_json", type=str, required=True,
                        help="Path to JSON mapping {train_dataset: probe_dir}")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug_first_n", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    probe_map = json.loads(Path(args.probe_map_json).read_text())
    probe_map = {k.strip(): Path(v) for k, v in probe_map.items()}

    console.print(Panel.fit(
        f"[bold cyan]Cross-Dataset Probe Eval (efficient)[/bold cyan]\n"
        f"Model: {args.model_id}\n"
        f"Eval datasets: {', '.join(datasets)}\n"
        f"Probe map: {args.probe_map_json}\n"
        f"Output: {output_dir}",
        border_style="cyan"
    ))

    # Load model once
    console.print("\n[bold]Loading Model[/bold]")
    if args.model == "qwen":
        model = Qwen7B(args.model_id, dtype="float16", device_map="auto")
    else:
        model = Llama31_8B(args.model_id, dtype="float16", device_map="auto")
    console.print("  [green]Loaded[\green]")

    # Load all probes once
    console.print("\n[bold]Loading Probes[/bold]")
    probes = {}
    for train_ds, pdir in probe_map.items():
        probe, thr, feature_names = load_probe(pdir)
        probes[train_ds] = {
            "dir": pdir,
            "probe": probe,
            "threshold": thr,
            "feature_names": feature_names,
        }
        console.print(f"{train_ds}: {pdir} (thr={thr:.3f}, nfeat={len(feature_names)})")

    all_metrics = []

    for eval_ds in datasets:
        console.print(f"\n[bold cyan]Eval Dataset: {eval_ds}[/bold cyan]")
        split = DATASET_SPLITS.get(eval_ds.lower(), "validation")
        ds = load_qa(eval_ds, split, limit=args.limit)
        console.print(f"  Loaded {len(ds):,} examples ({split})")

        # Choose transfer probes: all train_ds != eval_ds
        transfer_train_datasets = [td for td in probes.keys() if td.lower() != eval_ds.lower()]
        if len(transfer_train_datasets) == 0:
            console.print("  [yellow]No transfer probes for this dataset (probe map only contains itself).[/yellow]")
            continue

        # Accumulate rows per probe
        rows_by_train_ds = {td: [] for td in transfer_train_datasets}

        for ex in tqdm(ds, desc="  Rollout+feats (once) + score 4 probes", leave=False):
            question = ex.get("question", "")
            gold_answers = ex.get("answers", [])
            context = ex.get("context", None) or ex.get("passage", None) or ex.get("paragraph", None)

            feats = collect_features(model, question, eval_ds, context=context)
            pred_answer = feats["parsed_answer"]
            correct = is_correct(eval_ds, pred_answer, gold_answers)

            for train_ds in transfer_train_datasets:
                P = probes[train_ds]
                X = build_feature_dataframe(feats, feature_names=P["feature_names"])
                probe = P["probe"]
                if hasattr(probe, "predict_proba"):
                    p_correct = float(probe.predict_proba(X)[:, 1][0])
                else:
                    p_correct = float(probe.predict(X)[0])

                rows_by_train_ds[train_ds].append({
                    "question": question,
                    "pred_answer": pred_answer,
                    "y_true": int(correct),
                    "p_correct": p_correct,
                    "gold_answers": gold_answers,
                })

        # Save outputs per transfer probe (same artifacts as before)
        for train_ds, rows in rows_by_train_ds.items():
            results_df = pd.DataFrame(rows)
            if len(results_df) == 0:
                continue

            y_true = results_df["y_true"].values
            p = results_df["p_correct"].values
            thr = float(probes[train_ds]["threshold"])
            y_pred = (p >= thr).astype(int)
            results_df["y_pred"] = y_pred

            out = output_dir / eval_ds / train_ds
            out.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(out / "predictions.csv", index=False)
            save_plots(results_df, out, eval_ds)

            model_acc = float(np.mean(y_true)) if len(y_true) else 0.0
            probe_acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
            prec = float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0
            rec = float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0
            f1 = float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0

            if len(np.unique(y_true)) > 1:
                auc_roc = float(roc_auc_score(y_true, p))
                auc_pr = float(average_precision_score(y_true, p))
                brier = float(brier_score_loss(y_true, p))
                ll = float(log_loss(y_true, np.clip(p, 1e-7, 1-1e-7)))
            else:
                auc_roc = float("nan")
                auc_pr = float("nan")
                brier = float("nan")
                ll = float("nan")

            metrics = {
                "model": args.model,
                "model_id": args.model_id,
                "eval_dataset": eval_ds,
                "train_dataset": train_ds,
                "split": split,
                "n": int(len(y_true)),
                "threshold": thr,
                "model_accuracy": model_acc,
                "probe_accuracy": probe_acc,
                "probe_precision": prec,
                "probe_recall": rec,
                "probe_f1": f1,
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "brier": brier,
                "log_loss": ll,
                "probe_dir": str(probes[train_ds]["dir"]),
            }
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
            all_metrics.append(metrics)

            console.print(
                f"{eval_ds} <- {train_ds} | "
                f"Acc={model_acc:.3f} ProbeAcc={probe_acc:.3f} F1={f1:.3f} "
                f"AUC={auc_roc:.3f}" if auc_roc == auc_roc else
                f"{eval_ds} <- {train_ds} | Acc={model_acc:.3f} ProbeAcc={probe_acc:.3f} F1={f1:.3f}"
            )

        # Optional debug print for first N examples (same as before) — omitted for brevity

    # Summary table + JSON
    console.print("\n[bold cyan]Summary[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Eval", style="cyan")
    table.add_column("TrainProbe", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("ModelAcc", justify="right")
    table.add_column("ProbeAcc", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("AUC-ROC", justify="right")
    table.add_column("AUC-PR", justify="right")

    for m in all_metrics:
        table.add_row(
            str(m["eval_dataset"]),
            str(m["train_dataset"]),
            str(m["n"]),
            f'{m["model_accuracy"]:.3f}',
            f'{m["probe_accuracy"]:.3f}',
            f'{m["probe_f1"]:.3f}',
            f'{m["auc_roc"]:.3f}' if m["auc_roc"] == m["auc_roc"] else "n/a",
            f'{m["auc_pr"]:.3f}' if m["auc_pr"] == m["auc_pr"] else "n/a",
        )
    console.print(table)

    (output_dir / "summary.json").write_text(json.dumps(all_metrics, indent=2))
    console.print(f"\nWrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
