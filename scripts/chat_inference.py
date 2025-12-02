#!/usr/bin/env python3
"""
Chat-Style Confidence Estimation

Tests if probes transfer to conversational/chat prompts.

Usage:
    python chat_inference.py \
        --model llama \
        --probe_dir backend_llama31/probe_mlp \
        --interactive

    python chat_inference.py \
        --model llama \
        --probe_dir backend_llama31/probe_mlp \
        --questions "questions.txt" \
        --output chat_results.json
"""

import argparse
import json
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from src.models.llama31_8b import Llama31_8B
from src.models.qwen7b import Qwen7B

console = Console()

# ============================================================================
# RELATIVE CONFIDENCE SCORING
# ============================================================================
# Maps probe's actual output distribution (0.0-0.5, mean=0.18) to intuitive 0-1 scale

PROBE_PERCENTILES = {
    'p01': 0.000,  # Bottom 1%
    'p10': 0.010,  # Bottom 10%
    'p25': 0.040,  # Bottom 25%
    'p50': 0.100,  # Median
    'p75': 0.200,  # Top 25%
    'p90': 0.350,  # Top 10%
    'p99': 0.600,  # Top 1%
}

def relative_confidence(raw_prob, scale='0-100'):
    """
    Convert raw probe probability to relative confidence score.
    
    Based on observed validation distribution where most predictions
    are in 0.0-0.3 range with mean=0.18.
    
    Examples:
        0.186 (Paris) → 79/100 (High)
        0.119 (Abu Dhabi) → 61/100 (Medium-High)
        0.025 (wrong math) → 18/100 (Low)
        0.000 (impossible) → 0/100 (Very Low)
    """
    p = PROBE_PERCENTILES
    
    if raw_prob <= p['p01']:
        rel = 0.0
    elif raw_prob <= p['p10']:
        rel = 0.1 * (raw_prob - p['p01']) / (p['p10'] - p['p01'])
    elif raw_prob <= p['p25']:
        rel = 0.1 + 0.15 * (raw_prob - p['p10']) / (p['p25'] - p['p10'])
    elif raw_prob <= p['p50']:
        rel = 0.25 + 0.25 * (raw_prob - p['p25']) / (p['p50'] - p['p25'])
    elif raw_prob <= p['p75']:
        rel = 0.5 + 0.25 * (raw_prob - p['p50']) / (p['p75'] - p['p50'])
    elif raw_prob <= p['p90']:
        rel = 0.75 + 0.15 * (raw_prob - p['p75']) / (p['p90'] - p['p75'])
    elif raw_prob <= p['p99']:
        rel = 0.9 + 0.09 * (raw_prob - p['p90']) / (p['p99'] - p['p90'])
    else:
        rel = 0.99 + 0.01 * min(1.0, (raw_prob - p['p99']) / (1.0 - p['p99']))
    
    if scale == '0-100':
        return rel * 100.0
    else:
        return rel

# Chat prompt templates
CHAT_TEMPLATES = {
    'training': """You are answering trivia questions. Return only a single JSON object with keys exactly "answer" and "confidence". Do not include any other keys or text. The key must be spelled "confidence" (not "conference").

Question: {question}

Answer (JSON format):""",
    
    'casual': """Hey! I have a quick question for you.

{question}

Thanks!""",
    
    'assistant': """You are a helpful AI assistant. Please answer the following question to the best of your ability.

Question: {question}

Answer:""",
    
    'cot': """Let's think through this step by step.

Question: {question}

Let me work through this carefully:""",
    
    'concise': """{question}""",
    
    'instructive': """<|im_start|>system
You are a knowledgeable assistant who provides accurate, concise answers.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
""",
}

def initialize_model(model_name):
    """Initialize model"""
    console.print(f"[bold cyan]Loading model: {model_name}[/bold cyan]")
    
    if model_name in ["llama", "llama31"]:
        model = Llama31_8B()
        device = next(model.model.parameters()).device
        console.print(f"✓ Llama-3.1-8B loaded on [green]{device}[/green]")
    elif model_name == "qwen":
        model = Qwen7B()
        device = next(model.model.parameters()).device
        console.print(f"✓ Qwen2.5-7B loaded on [green]{device}[/green]")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def load_probe(probe_dir, use_hidden=True, calibration_method=None):
    """Load probe and optional calibration"""
    probe_path = Path(probe_dir)
    
    # Determine probe type
    probe_type = None
    for ptype in ["xform", "mlp", "logreg_cal", "logreg", "tree", "lasso", "elasticnet", "forest"]:
        if ptype in probe_dir or probe_path.name == f"probe_{ptype}":
            probe_type = ptype
            break
    
    if probe_type is None:
        probe_type = "mlp"
    
    # Load model
    if probe_type == "xform":
        console.print("[yellow]⚠ Transformer probe not fully supported in chat mode[/yellow]")
        return None
    
    model_file = probe_path / "probe_model.joblib"
    if not model_file.exists():
        console.print(f"[red]✗ Probe model not found at {model_file}[/red]")
        raise FileNotFoundError(f"Probe model not found at {model_file}")
    
    probe_model = joblib.load(model_file)
    
    # Load calibration if requested
    calibration_model = None
    if calibration_method:
        if calibration_method == "isotonic":
            cal_file = probe_path / "isotonic_calibration.joblib"
        elif calibration_method == "platt":
            cal_file = probe_path / "platt_calibration.joblib"
        else:
            raise ValueError(f"Unknown calibration method: {calibration_method}")
        
        if cal_file.exists():
            calibration_model = joblib.load(cal_file)
            console.print(f"✓ Loaded [cyan]{probe_type}[/cyan] probe with [green]{calibration_method}[/green] calibration")
        else:
            console.print(f"[yellow]⚠ Calibration file not found: {cal_file}[/yellow]")
            console.print(f"✓ Loaded [cyan]{probe_type}[/cyan] probe (no calibration)")
    else:
        console.print(f"✓ Loaded [cyan]{probe_type}[/cyan] probe")
    
    return {
        "type": probe_type,
        "model": probe_model,
        "use_hidden": use_hidden,
        "calibration": calibration_model,
        "calibration_method": calibration_method
    }

def extract_features_chat(model, question, prompt_template, use_hidden=True):
    """Extract features from chat-style generation"""
    
    # Format prompt
    prompt = prompt_template.format(question=question)
    
    # Generate (same as collect_internals.py)
    inp, gen = model.generate_with_states(prompt)
    
    # Extract generated text
    start = inp["input_ids"].shape[-1]
    new_ids = gen.sequences[:, start:]
    raw_text = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
    
    # Parse JSON answer (matching eval_probe_optimized.py)
    answer = None
    model_confidence = None
    parsed_json_ok = 0.0
    parsed_p_true_ok = 0.0
    is_unknown = 0.0
    
    try:
        import json as json_lib
        import re
        # Try to extract JSON from raw text
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json_lib.loads(json_str)
            parsed_json_ok = 1.0
            
            # Extract answer text
            if 'answer' in parsed:
                answer = str(parsed['answer']) if parsed['answer'] is not None else ""
            
            # Extract confidence (named 'confidence' in model output, but will use 'p_true' in rescore)
            if 'confidence' in parsed:
                model_confidence = float(parsed['confidence'])
                parsed_p_true_ok = 1.0
            
            # Check for unknown
            if answer and answer.strip().lower() == 'unknown':
                is_unknown = 1.0
    except:
        pass  # Couldn't parse JSON
    
    # Fallback: use raw text as answer if parsing failed
    if answer is None:
        answer = raw_text
        parsed_json_ok = 0.0
        parsed_p_true_ok = 0.0
    
    # Build input for rescoring
    input_ids = inp["input_ids"]
    
    # Token-level stats
    scores = gen.scores or []
    T = len(scores)
    
    step_logp = []
    step_ent = []
    margins = []
    
    gen_ids_seq = new_ids[0].tolist()
    
    for t in range(T):
        logits = scores[t][0].float()
        
        # Margin
        top2 = torch.topk(logits, k=2).values
        margins.append(float(top2[0] - top2[1]))
        
        # Entropy
        probs = torch.softmax(logits, dim=-1)
        ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        step_ent.append(ent)
        
        # Log prob
        tok_id = gen_ids_seq[t] if t < len(gen_ids_seq) else None
        if tok_id is not None:
            lsm = torch.log_softmax(logits, dim=-1)
            step_logp.append(float(lsm[tok_id].item()))
    
    # Aggregate stats (exclude final token for content stats)
    content_len = max(1, T - 1) if T > 1 else T
    
    lp_mean = float(sum(step_logp[:content_len]) / content_len) if step_logp else None
    seq_conf = np.exp(lp_mean) if lp_mean is not None else None
    
    entropy_mean = float(sum(step_ent[:content_len]) / content_len) if step_ent else None
    entropy_std = float(np.std(step_ent[:content_len])) if len(step_ent[:content_len]) > 1 else None
    
    margin_mean = float(sum(margins[:content_len]) / content_len) if margins else None
    margin_min = float(min(margins[:content_len])) if content_len > 0 and margins else None
    
    # Hidden states
    h_last_256 = None
    h_pool_256 = None
    h_last_mid_256 = None
    h_pool_mid_256 = None
    
    if use_hidden:
        device = next(model.model.parameters()).device
        console.print(f"\n[yellow]DEBUG: Extracting hidden states (use_hidden=True)[/yellow]")
        console.print(f"  Device: {device}")
        console.print(f"  input_ids shape: {input_ids.shape}")
        console.print(f"  new_ids shape: {new_ids.shape}")
        
        with torch.no_grad():
            full_ids = torch.cat([input_ids.to(device), new_ids.to(device)], dim=-1)
            console.print(f"  full_ids shape: {full_ids.shape}")
            attn = torch.ones_like(full_ids)
            out = model.model(input_ids=full_ids, attention_mask=attn,
                            output_hidden_states=True, use_cache=False, return_dict=True)
            
            console.print(f"  Number of hidden state layers: {len(out.hidden_states)}")
            
            # Last layer (matching eval exactly)
            hs_final = out.hidden_states[-1][0]  # [seq_len, hidden_dim]
            console.print(f"  hs_final shape: {hs_final.shape}")
            gen_len = new_ids.shape[-1]
            h_ans = hs_final[-gen_len:]  # Last gen_len tokens
            console.print(f"  h_ans shape: {h_ans.shape}")
            h_last = h_ans[-1] if h_ans.shape[0] > 0 else None
            h_pool = h_ans.mean(dim=0) if h_ans.shape[0] > 0 else None
            console.print(f"  h_last shape: {h_last.shape if h_last is not None else 'None'}")
            console.print(f"  h_pool shape: {h_pool.shape if h_pool is not None else 'None'}")
            
            # Mid layer
            mid_ix = len(out.hidden_states) // 2
            hs_mid = out.hidden_states[mid_ix][0]
            h_ans_mid = hs_mid[-gen_len:]
            h_last_mid = h_ans_mid[-1] if h_ans_mid.shape[0] > 0 else None
            h_pool_mid = h_ans_mid.mean(dim=0) if h_ans_mid.shape[0] > 0 else None
            
            # Pack to 256 dimensions (matching eval's pack_256 function exactly!)
            def pack_256(vec):
                if vec is None:
                    console.print(f"    [red]pack_256: vec is None![/red]")
                    return None
                # L2 normalize (CRITICAL - matches eval!)
                v = torch.nn.functional.normalize(vec.float(), dim=-1)
                # Take first 256 dims
                v = v[:256] if v.shape[-1] >= 256 else torch.nn.functional.pad(v, (0, 256 - v.shape[-1]))
                packed = [float(x) for x in v.cpu()]
                console.print(f"    pack_256: Created normalized list of {len(packed)} values")
                return packed
            
            console.print(f"\n[yellow]  Packing to 256 dimensions...[/yellow]")
            h_last_256 = pack_256(h_last)
            h_pool_256 = pack_256(h_pool)
            h_last_mid_256 = pack_256(h_last_mid)
            h_pool_mid_256 = pack_256(h_pool_mid)
    
    # Rescore canonical JSON (matching eval exactly)
    rescore_logp = None
    if model_confidence is not None:
        try:
            # Use p_true (not confidence) to match training data
            safe_ans = answer.replace('"', '\\"') if answer else ""
            pt = model_confidence
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
                rescore_logp = sum(step_logps) / max(1, len(step_logps))
        except:
            pass  # Rescoring failed, leave as None
    
    # Features dict (matching eval exactly)
    features = {
        'model_confidence': model_confidence,
        'lp_mean': lp_mean,
        'seq_conf': seq_conf,
        'entropy_mean': entropy_mean,
        'entropy_std': entropy_std,
        'margin_mean': margin_mean,
        'margin_min': margin_min,
        'rescore_logp': rescore_logp,
        'answer_len': int(new_ids.shape[-1]),
        'parsed_json_ok': parsed_json_ok,
        'parsed_p_true_ok': parsed_p_true_ok,
        'is_unknown': is_unknown,
        'h_last_256': h_last_256,
        'h_pool_256': h_pool_256,
        'h_last_mid_256': h_last_mid_256,
        'h_pool_mid_256': h_pool_mid_256,
    }
    
    return answer, features

def run_probe_inference(probe, features):
    """Run probe inference"""
    
    probe_model = probe["model"]
    use_hidden = probe["use_hidden"]
    
    # CRITICAL: Features must be in THIS exact order (not alphabetical!)
    # This matches how train_probe.py builds features
    SCALAR_KEYS = [
        "model_confidence", "lp_mean", "seq_conf",
        "entropy_mean", "entropy_std",
        "margin_mean", "margin_min",
        "rescore_logp", "answer_len",
        "parsed_json_ok", "parsed_p_true_ok", "is_unknown",
    ]
    VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]
    
    # DEBUG: Print extracted features
    console.print("\n[yellow]DEBUG: Extracted Features[/yellow]")
    for k in SCALAR_KEYS:
        val = features.get(k, np.nan)
        console.print(f"  {k}: {val}")
    
    # Check hidden states
    if use_hidden:
        console.print("\n[yellow]DEBUG: Hidden States[/yellow]")
        for key in VECTOR_KEYS:
            vec = features.get(key)
            if vec is None:
                console.print(f"  {key}: [red]None (ERROR!)[/red]")
            elif isinstance(vec, list):
                console.print(f"  {key}: [green]List of {len(vec)} values[/green] (sample: {vec[:3]}...)")
            else:
                console.print(f"  {key}: [yellow]Unexpected type: {type(vec)}[/yellow]")
    
    # Build feature array in CORRECT ORDER (not alphabetical!)
    feat_array = []
    
    # Add scalars in order
    for k in SCALAR_KEYS:
        feat_array.append(features.get(k, np.nan))
    
    # Add vectors in order
    if use_hidden:
        for name in VECTOR_KEYS:
            vec = features.get(name)
            if vec is not None and len(vec) == 256:
                feat_array.extend(vec)
            else:
                # Missing vector - fill with zeros
                console.print(f"[red]WARNING: {name} missing or wrong length, filling with zeros[/red]")
                feat_array.extend([0.0] * 256)
    
    # Convert to numpy array
    feat_array = np.array(feat_array, dtype=np.float32).reshape(1, -1)
    
    console.print(f"\n[yellow]DEBUG: Feature array shape: {feat_array.shape}[/yellow]")
    console.print(f"[yellow]DEBUG: Expected shape: (1, {12 + (1024 if use_hidden else 0)})[/yellow]")
    console.print(f"[yellow]DEBUG: First 12 values: {feat_array[0, :12]}[/yellow]")
    
    # Predict
    prob = probe_model.predict_proba(feat_array)[:, 1][0]
    
    console.print(f"\n[yellow]DEBUG: Raw probe probability: {prob:.6f}[/yellow]")
    
    # Apply calibration if available
    if probe.get("calibration") is not None:
        cal_model = probe["calibration"]
        cal_method = probe.get("calibration_method", "unknown")
        raw_prob = prob
        
        if cal_method == "isotonic":
            prob = float(cal_model.predict([prob])[0])
        elif cal_method == "platt":
            prob = float(cal_model.predict_proba([[prob]])[0, 1])
        
        prob = np.clip(prob, 0.0, 1.0)
        console.print(f"[yellow]DEBUG: Calibrated probability ({cal_method}): {prob:.6f} (Δ={prob-raw_prob:+.6f})[/yellow]\n")
    else:
        console.print()
    
    return float(prob)

def interactive_mode(model, probe):
    """Interactive chat mode"""
    
    print("\n" + "="*80)
    print("INTERACTIVE CHAT MODE")
    print("="*80)
    print("Type your questions. The system will:")
    print("  1. Generate an answer")
    print("  2. Estimate confidence that the answer is correct")
    print()
    print("Commands:")
    print("  /template <name>  - Change prompt template (casual, assistant, cot, etc.)")
    print("  /quit             - Exit")
    print("="*80 + "\n")
    
    current_template = "casual"
    
    while True:
        question = input("You: ").strip()
        
        if not question:
            continue
        
        if question == "/quit":
            break
        
        if question.startswith("/template "):
            template_name = question.split()[1]
            if template_name in CHAT_TEMPLATES:
                current_template = template_name
                print(f"✓ Switched to '{template_name}' template\n")
            else:
                print(f"✗ Unknown template. Available: {', '.join(CHAT_TEMPLATES.keys())}\n")
            continue
        
        # Generate answer
        print("\nGenerating answer...")
        answer, features = extract_features_chat(
            model, question, 
            CHAT_TEMPLATES[current_template],
            use_hidden=probe["use_hidden"]
        )
        
        # Estimate confidence
        confidence = run_probe_inference(probe, features)
        rel_conf = relative_confidence(confidence, scale='0-100')
        
        # Display
        print(f"\nA: {answer}")
        print(f"\n{'─'*80}")
        print(f"Confidence Score: {confidence:.3f} (raw) → {rel_conf:.1f}/100 (relative)")
        
        if rel_conf >= 75:
            print("  ✓ High confidence - likely correct")
        elif rel_conf >= 50:
            print("  ⚠ Medium confidence - verify if important")
        elif rel_conf >= 25:
            print("  ⚠ Low confidence - likely uncertain")
        else:
            print("  ✗ Very low confidence - likely incorrect")
        print(f"{'─'*80}\n")

def batch_mode(model, probe, questions_file, output_file):
    """Batch evaluation on questions file"""
    
    # Load questions
    with open(questions_file) as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(questions)} questions")
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question[:50]}...")
        
        # Test multiple templates
        for template_name, template in CHAT_TEMPLATES.items():
            answer, features = extract_features_chat(
                model, question, template,
                use_hidden=probe["use_hidden"]
            )
            
            confidence = run_probe_inference(probe, features)
            
            results.append({
                'question': question,
                'template': template_name,
                'answer': answer,
                'confidence': confidence,
                'features': {k: v for k, v in features.items() 
                           if not isinstance(v, list)}  # Exclude vectors
            })
            
            print(f"  [{template_name:12s}] conf={confidence:.3f}")
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {output_file}")
    
    # Analysis
    analyze_template_variance(results)

def analyze_template_variance(results):
    """Analyze how confidence varies across templates"""
    
    print("\n" + "="*80)
    print("TEMPLATE SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Group by question
    by_question = {}
    for r in results:
        q = r['question']
        if q not in by_question:
            by_question[q] = []
        by_question[q].append(r)
    
    # Variance analysis
    variances = []
    for question, results_list in by_question.items():
        confidences = [r['confidence'] for r in results_list]
        variance = np.var(confidences)
        variances.append(variance)
        
        if variance > 0.1:
            print(f"\n⚠ High variance ({variance:.3f}) for: {question[:60]}...")
            for r in results_list:
                print(f"  {r['template']:12s}: {r['confidence']:.3f}")
    
    print(f"\n{'─'*80}")
    print(f"Average confidence variance across templates: {np.mean(variances):.4f}")
    print(f"Max variance: {np.max(variances):.4f}")
    
    if np.mean(variances) > 0.05:
        print("\n⚠ WARNING: Probes are sensitive to prompt formatting!")
        print("  → May need prompt-specific calibration")
        print("  → Training on diverse prompts could improve robustness")
    else:
        print("\n✓ Probes are relatively robust to prompt variations")

def main():
    parser = argparse.ArgumentParser(description="Chat-style confidence estimation")
    parser.add_argument("--model", default="llama", choices=["llama", "llama31", "qwen"])
    parser.add_argument("--probe_dir", required=True, help="Path to probe directory")
    parser.add_argument("--use_hidden", action="store_true", default=True)
    parser.add_argument("--calibration", choices=["isotonic", "platt"], default=None,
                       help="Use calibrated probabilities (requires calibration files)")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--questions", help="File with questions (one per line)")
    parser.add_argument("--output", default="chat_results.json")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CHAT-STYLE CONFIDENCE ESTIMATION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Probe: {args.probe_dir}")
    if args.calibration:
        print(f"Calibration: {args.calibration}")
    
    # Initialize
    model = initialize_model(args.model)
    probe = load_probe(args.probe_dir, args.use_hidden, args.calibration)
    
    if probe is None:
        print("✗ Failed to load probe")
        return
    
    # Run
    if args.interactive:
        interactive_mode(model, probe)
    elif args.questions:
        batch_mode(model, probe, args.questions, args.output)
    else:
        print("\nError: Must specify either --interactive or --questions")

if __name__ == "__main__":
    main()