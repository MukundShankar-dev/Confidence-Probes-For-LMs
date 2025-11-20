#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe-based answer ranking for MMLU multiple choice questions

This script:
1. Loads MMLU test set
2. For each question, generates answers for all choices
3. Uses probe to rank choices by confidence
4. Selects the highest-ranked answer
5. Reports accuracy compared to ground truth

Usage:
    python -m scripts.rank_and_answer \
        --model llama31 \
        --probe_dir backend_llama31/probe_mlp \
        --use_hidden \
        --limit 100

    # Full MMLU test set
    python -m scripts.rank_and_answer \
        --model llama31 \
        --probe_dir backend_llama31/probe_mlp \
        --use_hidden
"""

import argparse
import json
import os
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()

# Model imports
from src.models.qwen7b import Qwen7B
from src.models.llama31_8b import Llama31_8B
from src.data.datasets import load_qa

# Speed optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def parse_mmlu_question(question_text: str) -> Tuple[str, Dict[str, str], str]:
    """
    Parse MMLU question format where choices are embedded in the question.
    
    Example input:
    "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
    A. 0
    B. 4
    C. 2
    D. 6"
    
    Returns:
        (question, choices_dict, None) where choices_dict = {"A": "0", "B": "4", "C": "2", "D": "6"}
    """
    import re
    
    # Split by newlines
    lines = question_text.strip().split('\n')
    
    # First line(s) before choices = question
    question_lines = []
    choices = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this line is a choice (starts with A., B., C., or D.)
        match = re.match(r'^([A-D])\.\s*(.+)$', line)
        if match:
            letter = match.group(1)
            text = match.group(2).strip()
            choices[letter] = text
        else:
            # It's part of the question
            question_lines.append(line)
    
    question = ' '.join(question_lines)
    
    return question, choices


def parse_mmlu_answer(answers_list: List[str]) -> str:
    """
    Parse answer from MMLU format.
    
    Examples:
        ['B', '4', 'B. 4'] -> 'B'
        ['A'] -> 'A'
    
    Returns the letter (A, B, C, or D)
    """
    for ans in answers_list:
        ans = str(ans).strip()
        # Check if it's just a letter
        if ans in ['A', 'B', 'C', 'D']:
            return ans
        # Check if it starts with a letter followed by .
        if len(ans) >= 2 and ans[0] in ['A', 'B', 'C', 'D'] and ans[1] in ['.', ' ']:
            return ans[0]
    
    # Fallback: try to extract any A, B, C, or D
    for ans in answers_list:
        for char in ans:
            if char in ['A', 'B', 'C', 'D']:
                return char
    
    return None


# MMLU prompt template
MMLU_PROMPT_TEMPLATE = """Answer the following multiple choice question. Return only a JSON object with:
- "answer": the letter of the correct choice (A, B, C, or D)
- "p_true": your confidence (0.0-1.0) that this answer is correct

Question: {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Your response:"""


def initialize_model(model_name: str, max_new_tokens: int = 32):
    """Initialize the language model"""
    console.print(f"[cyan]🤖 Loading model: {model_name}...[/cyan]")
    
    if model_name == "llama31":
        model = Llama31_8B(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            max_new_tokens=max_new_tokens,
            device_map="auto"
        )
    elif model_name == "qwen":
        model = Qwen7B(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            max_new_tokens=max_new_tokens,
            device_map="auto"
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    console.print(f"[green]✓ Model loaded successfully[/green]")
    return model


def pack_256(tensor):
    """Pack tensor to 256 dimensions"""
    if tensor is None:
        return None
    vec = tensor.detach().cpu().float().numpy().ravel()
    if vec.size < 256:
        vec = np.pad(vec, (0, 256 - vec.size))
    elif vec.size > 256:
        vec = vec[:256]
    return vec


def rescore_mean_logprob(model, tok, prompt_text, target_text):
    """Compute mean log probability of target given prompt"""
    import torch
    
    device = next(model.parameters()).device
    enc_prompt = tok(prompt_text, add_special_tokens=False, return_tensors="pt")
    enc_target = tok(target_text, add_special_tokens=False, return_tensors="pt")
    input_ids = torch.cat([enc_prompt["input_ids"], enc_target["input_ids"]], dim=-1).to(device)
    attn = torch.ones_like(input_ids)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
    
    logits = out.logits[:, :-1, :]
    target_slice = input_ids[:, enc_prompt["input_ids"].shape[-1]:]
    logits_tgt = logits[:, -target_slice.shape[-1]:, :]
    
    step_logps = []
    for t in range(target_slice.shape[-1]):
        lsm = torch.log_softmax(logits_tgt[0, t].float(), dim=-1)
        step_logps.append(float(lsm[int(target_slice[0, t])].item()))
    
    return sum(step_logps) / max(1, len(step_logps))


def extract_features_for_choice(model, question: str, choice_letter: str, 
                                choice_text: str, all_choices: Dict[str, str],
                                use_hidden: bool = True):
    """
    Generate answer for a specific choice and extract probe features.
    
    CRITICAL: Must use the SAME prompt format as training!
    
    During training, the prompt was:
        Q: What is the capital of France?
    
    And model generated:
        {"answer": "Paris", "p_true": 0.92}
    
    We need to match this format exactly, but bias towards a specific choice.
    
    Args:
        model: The language model
        question: The question text
        choice_letter: "A", "B", "C", or "D" - the choice we're testing
        choice_text: The text of this choice
        all_choices: Dict with all choices {"A": ..., "B": ..., etc}
        use_hidden: Whether to extract hidden states
    
    Returns:
        Dict of features for probe
    """
    import math
    import json
    import re
    
    # Use the SAME format as training (FEWSHOT_PROMPT style from eval_probe.py)
    # But include context about all choices
    prompt = f"""You are answering trivia questions.
Return only a single JSON object with fields:
- "answer": the short factual span (1–5 words, no punctuation)
- "p_true": the probability (0.0–1.0) that this answer is correct (round to two decimals)

Q: Who wrote Hamlet?
{{"answer": "William Shakespeare", "p_true": 0.95}}

Q: What is the capital of France?
{{"answer": "Paris", "p_true": 0.92}}

Q: {question}
Choices: A. {all_choices["A"]}, B. {all_choices["B"]}, C. {all_choices["C"]}, D. {all_choices["D"]}

If the correct answer is choice {choice_letter} ({choice_text}), respond:
"""
    
    # Let model generate the full JSON
    inp, gen = model.generate_with_states(prompt)
    
    start_pos = inp["input_ids"].shape[-1]
    new_ids = gen.sequences[:, start_pos:]
    raw_text = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
    
    # Try to extract answer and p_true from generated text
    answer = choice_text  # We know what answer should be
    p_model = None
    
    try:
        # Try to parse as JSON
        if raw_text.strip().startswith("{"):
            parsed = json.loads(raw_text)
            if "p_true" in parsed:
                p_model = float(parsed["p_true"])
            if "answer" in parsed:
                answer = parsed["answer"]
        else:
            # Look for p_true value in text
            match = re.search(r'"p_true":\s*([0-9.]+)', raw_text)
            if match:
                p_model = float(match.group(1))
    except:
        pass
    
    if p_model is None:
        p_model = 0.5  # Default if can't parse
    
    # Now extract features using the same method as eval_probe
    device = next(model.model.parameters()).device
    tok = model.tok
    hf_causal_lm = model.model
    
    # Token-level statistics from generation
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
            full_ids = torch.cat([inp["input_ids"].to(device), new_ids.to(device)], dim=-1)
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
    
    # DEBUG: Print what was actually generated
    import os
    if os.environ.get('DEBUG_GENERATION') == '1':
        console.print(f"[dim]Generated text: '{raw_text}'[/dim]")
        console.print(f"[dim]Generated tokens: {new_ids.shape[-1]}[/dim]")
    
    # Rescore probability - use the actual answer text, not just the letter
    canon_json = f'{{"answer": "{answer}", "p_true": {p_model:.2f}}}'
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
        "parsed_json_ok": 1 if "{" in raw_text else 0,
        "parsed_p_true_ok": 1 if p_model is not None else 0,
        "is_unknown": 1 if "unknown" in answer.lower() else 0,
        "h_last_256": h_last_256,
        "h_pool_256": h_pool_256,
        "h_last_mid_256": h_last_mid_256,
        "h_pool_mid_256": h_pool_mid_256,
    }
    
    return features


def build_feature_dataframe(features: Dict, use_hidden: bool = True) -> pd.DataFrame:
    """Convert feature dict to DataFrame for probe"""
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
    
    return pd.DataFrame([feat_dict])


def rank_choices_with_probe(model, probe, question: str, choices: Dict[str, str],
                            use_hidden: bool = True, debug: bool = False) -> List[Tuple[str, float]]:
    """
    Rank multiple choice options by probe confidence
    
    Args:
        model: Language model
        probe: Trained probe (joblib model)
        question: Question text
        choices: {"A": "...", "B": "...", "C": "...", "D": "..."}
        use_hidden: Whether to use hidden states
        debug: Print debug info
    
    Returns:
        List of (choice_letter, probe_confidence) sorted by confidence (high to low)
    """
    results = []
    
    for choice_letter in ["A", "B", "C", "D"]:
        choice_text = choices[choice_letter]
        
        # Extract features for this choice
        features = extract_features_for_choice(
            model=model,
            question=question,
            choice_letter=choice_letter,
            choice_text=choice_text,
            all_choices=choices,
            use_hidden=use_hidden
        )
        
        if debug:
            console.print(f"[dim]Choice {choice_letter}: lp_mean={features.get('lp_mean', 'N/A')}, "
                         f"entropy_mean={features.get('entropy_mean', 'N/A')}[/dim]")
            # Check hidden states
            h_last = features.get('h_last_256')
            if h_last is not None:
                console.print(f"[dim]  h_last_256: shape={len(h_last) if hasattr(h_last, '__len__') else 'scalar'}, "
                             f"sample values: {h_last[:3] if hasattr(h_last, '__len__') else h_last}[/dim]")
            else:
                console.print(f"[red]  h_last_256: None![/red]")
        
        # Convert to DataFrame
        df = build_feature_dataframe(features, use_hidden=use_hidden)
        
        if debug:
            console.print(f"[dim]  DataFrame shape: {df.shape}[/dim]")
            console.print(f"[dim]  NaN count: {df.isna().sum().sum()}[/dim]")
            console.print(f"[dim]  Sample columns: {list(df.columns[:5])}[/dim]")
            console.print(f"[dim]  Sample values: {df.iloc[0, :5].tolist()}[/dim]")
        
        # Get probe confidence
        probe_conf = probe.predict_proba(df.values)[:, 1][0]
        
        if debug:
            console.print(f"[dim]  → Probe confidence: {probe_conf:.8f}[/dim]")
        
        results.append((choice_letter, float(probe_conf)))
    
    # Sort by confidence (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def load_probe(probe_dir: str) -> object:
    """Load trained probe from directory"""
    probe_path = Path(probe_dir)
    model_file = probe_path / "probe_model.joblib"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Probe model not found at {model_file}")
    
    console.print(f"[cyan]🔬 Loading probe from {probe_dir}...[/cyan]")
    probe = joblib.load(model_file)
    console.print(f"[green]✓ Probe loaded successfully[/green]")
    
    return probe


def evaluate_mmlu_with_ranking(model, probe, limit: int = None, use_hidden: bool = True):
    """
    Evaluate on MMLU test set using probe-based answer ranking
    
    Note: This script is designed for MMLU multiple choice format.
    For other QA datasets, use eval_probe.py instead.
    
    Args:
        model: Language model
        probe: Trained probe
        limit: Optional limit on number of examples
        use_hidden: Whether to use hidden states
    
    Returns:
        Dict with results
    """
    console.print(Panel.fit(
        "[bold cyan]MMLU Evaluation with Probe-Based Answer Ranking[/bold cyan]\n"
        f"Strategy: Rank all 4 choices by probe confidence\n"
        f"Expected: ~4-6% improvement over direct generation",
        title="🎯 Probe Ranking",
        border_style="cyan"
    ))
    
    # Load MMLU test set
    console.print(f"\n[cyan]📚 Loading MMLU test set...[/cyan]")
    
    # Load using the standard load_qa function
    if limit:
        mmlu_data = load_qa("mmlu", "test", limit)
        console.print(f"[yellow]⚠ Limited to {limit} examples for testing[/yellow]")
    else:
        mmlu_data = load_qa("mmlu", "test", None)
    
    console.print(f"[green]✓ Loaded {len(mmlu_data)} examples[/green]")
    
    # Check data format
    if len(mmlu_data) > 0:
        sample = mmlu_data[0]
        console.print(f"[dim]Data format: question with embedded choices, answers list[/dim]")
        console.print(f"[dim]Example: {sample['question'][:80]}...[/dim]")
    
    # Track results
    results = []
    correct = 0
    total = 0
    skipped = 0
    
    # Process each question with rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[accuracy]:.1%}[/cyan]"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(
            "[cyan]Evaluating...", 
            total=len(mmlu_data),
            accuracy=0.0
        )
        
        for idx, item in enumerate(mmlu_data):
            question_raw = item.get("question", "")
            answers_raw = item.get("answers", [])
            
            # Parse MMLU format where choices are in the question text
            try:
                question, choices = parse_mmlu_question(question_raw)
                correct_answer = parse_mmlu_answer(answers_raw)
                
                if not choices or len(choices) != 4:
                    console.print(f"[yellow]⚠ Question {idx + 1}: Could not parse 4 choices, skipping[/yellow]")
                    skipped += 1
                    progress.update(task, advance=1, accuracy=correct / max(1, total))
                    continue
                
                if not correct_answer:
                    console.print(f"[yellow]⚠ Question {idx + 1}: Could not parse answer, skipping[/yellow]")
                    skipped += 1
                    progress.update(task, advance=1, accuracy=correct / max(1, total))
                    continue
                
            except Exception as e:
                console.print(f"[yellow]⚠ Question {idx + 1}: Parse error: {e}[/yellow]")
                skipped += 1
                progress.update(task, advance=1, accuracy=correct / max(1, total))
                continue
            
            # Rank choices by probe confidence
            try:
                ranked = rank_choices_with_probe(
                    model=model,
                    probe=probe,
                    question=question,
                    choices=choices,
                    use_hidden=use_hidden,
                    debug=(idx == 0)  # Debug first question only
                )
            except Exception as e:
                console.print(f"[red]ERROR ranking question {idx + 1}: {e}[/red]")
                skipped += 1
                progress.update(task, advance=1, accuracy=correct / max(1, total))
                continue
            
            # Select highest-ranked choice
            predicted_answer = ranked[0][0]
            predicted_conf = ranked[0][1]
            
            # Check correctness
            is_correct = (predicted_answer == correct_answer)
            if is_correct:
                correct += 1
            total += 1
            
            results.append({
                "question": question,
                "choices": choices,
                "ground_truth": correct_answer,
                "predicted": predicted_answer,
                "confidence": predicted_conf,
                "correct": is_correct,
                "all_rankings": ranked
            })
            
            # Update progress
            current_acc = correct / max(1, total)
            progress.update(task, advance=1, accuracy=current_acc)
            
            # Print sample every 25 examples
            if (idx + 1) % 25 == 0 and idx < 100:
                console.print(f"\n[dim]Sample {idx + 1}:[/dim]")
                console.print(f"  Q: {question[:60]}...")
                console.print(f"  Choices: {choices}")
                console.print(f"  Rankings: {', '.join([f'{c}:{conf:.6f}' for c, conf in ranked])}")  # Show 6 decimals
                console.print(f"  Predicted: {predicted_answer} | Truth: {correct_answer} | {'✓' if is_correct else '✗'}")
                
                # Debug: check if all confidences are actually the same
                if len(set(conf for _, conf in ranked)) == 1:
                    console.print(f"  [yellow]⚠ All choices have identical probe confidence![/yellow]")
    
    # Final results
    final_accuracy = correct / total if total > 0 else 0.0
    
    # Results table
    results_table = Table(show_header=True, header_style="bold green")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", justify="right", style="yellow")
    
    results_table.add_row("Total Examples", f"{len(mmlu_data):,}")
    results_table.add_row("Evaluated", f"{total:,}")
    results_table.add_row("Skipped", f"{skipped:,}")
    results_table.add_row("Correct", f"{correct:,}")
    results_table.add_row("Incorrect", f"{total - correct:,}")
    results_table.add_row("Accuracy", f"{final_accuracy:.3f} ({100 * final_accuracy:.1f}%)")
    
    console.print("\n")
    console.print(Panel.fit(results_table, title="📊 Final Results", border_style="green"))
    
    if skipped > 0:
        console.print(f"[yellow]⚠ Note: {skipped} questions were skipped (not multiple choice format)[/yellow]")
    
    return {
        "total": total,
        "correct": correct,
        "skipped": skipped,
        "accuracy": final_accuracy,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="MMLU evaluation with probe-based ranking")
    parser.add_argument("--model", choices=["llama31", "qwen"], required=True,
                       help="Model to use")
    parser.add_argument("--probe_dir", type=str, required=True,
                       help="Path to trained probe directory")
    parser.add_argument("--use_hidden", action="store_true",
                       help="Use hidden states (must match probe training)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of examples (for testing)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for detailed results")
    
    args = parser.parse_args()
    
    # Initialize model
    model = initialize_model(args.model, max_new_tokens=32)
    
    # Load probe
    probe = load_probe(args.probe_dir)
    
    # Run evaluation
    results = evaluate_mmlu_with_ranking(
        model=model,
        probe=probe,
        limit=args.limit,
        use_hidden=args.use_hidden
    )
    
    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[Output] Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()