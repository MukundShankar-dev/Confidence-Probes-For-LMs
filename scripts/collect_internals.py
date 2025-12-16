#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect internals for a confidence probe - STANDARD PROMPTS VERSION
- Uses dataset-appropriate standard prompts (no JSON)
- Captures token-level scores, margins, entropies from logits
- Runs one teacher-forced forward pass to grab hidden states
- Extracts answer from standard format output
- Writes compact features + label (EM) to JSONL

Key change: Uses standard prompts for each dataset type, not JSON format
"""
import argparse, json, os, re, math, time, sys
import torch
from tqdm import tqdm
from transformers.utils import logging as hf_logging

# === your project modules ===
from src.data.datasets import load_qa, squad_em, squad_f1
from src.models.gpt_oss import GPTOSS
from src.models.qwen7b import Qwen7B
from src.models.gemma12b import Gemma12B
from src.models.llama31_8b import Llama31_8B
from src.models.llama32_11b import Llama32_11B

# ------- speed knobs -------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ========================================================================
# STANDARD PROMPTS FOR EACH DATASET TYPE
# ========================================================================

# For TriviaQA, HotpotQA - standard open-domain QA
PROMPT_OPEN_QA = """Answer the following question with a short factual answer (1-5 words).

Q: Who wrote Hamlet?
A: William Shakespeare

Q: What is the capital of France?
A: Paris

Q: Which planet is known as the Red Planet?
A: Mars

Q: {QUESTION}
A:"""

# For MMLU - standard multiple choice
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

# For GSM8K - standard math problem
PROMPT_GSM8K = """Solve the following math problem. Show your reasoning and then provide the final numerical answer.

Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
A: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast, so 16 - 3 = 13 eggs remain. She uses 4 for muffins, so 13 - 4 = 9 eggs remain. She sells these 9 eggs for $2 each, so 9 * 2 = $18. The answer is 18.

Q: {QUESTION}
A:"""

# For SQuAD v2 - extractive QA with unanswerable questions
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

def build_prompt(question: str, dataset: str, context: str = None) -> str:
    """Build standard prompt for the dataset type"""
    dataset_lower = dataset.lower()
    
    if dataset_lower == "mmlu":
        # For MMLU, question already contains the choices
        return PROMPT_MMLU.format(QUESTION=question)
    
    elif dataset_lower == "gsm8k":
        return PROMPT_GSM8K.format(QUESTION=question)
    
    elif dataset_lower in ("squadv2", "squad_v2", "squad2"):
        return PROMPT_SQUADV2.format(CONTEXT=context or "", QUESTION=question)
    
    else:
        # TriviaQA, HotpotQA, and other open-domain QA
        return PROMPT_OPEN_QA.format(QUESTION=question)

def extract_answer_standard(text: str, dataset: str) -> str:
    """Extract answer from standard format output (no JSON parsing)"""
    text = text.strip()
    
    dataset_lower = dataset.lower()
    
    if dataset_lower == "mmlu":
        # For MMLU, extract just the letter
        # Look for single letter A/B/C/D at start or after "Answer:"
        match = re.search(r'\b([A-D])\b', text)
        if match:
            return match.group(1)
        return text.split()[0] if text else ""
    
    elif dataset_lower == "gsm8k":
        # For GSM8K, extract the final number
        # Look for patterns like "The answer is 18" or just "18"
        # Try to find number after "answer is" or at the end
        answer_match = re.search(r'(?:answer is|equals?)\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1)
        
        # Fall back to last number in text
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        return text
    
    elif dataset_lower in ("squadv2", "squad_v2", "squad2"):
        # For SQuAD, take first line or sentence
        # Remove common prefixes like "Answer:" if present
        text = re.sub(r'^(?:Answer|A):\s*', '', text, flags=re.IGNORECASE)
        # Take first sentence/line
        first_line = text.split('\n')[0].strip()
        return first_line[:100]  # Cap length
    
    else:
        # For open-domain QA (TriviaQA, HotpotQA)
        # Take first line, remove common patterns
        text = re.sub(r'^(?:Answer|A):\s*', '', text, flags=re.IGNORECASE)
        first_line = text.split('\n')[0].strip()
        # Remove trailing punctuation
        first_line = re.sub(r'[.!?]+$', '', first_line)
        return first_line[:100]  # Cap length

def pack_256(vec: torch.Tensor):
    """L2-normalize then take first 256 dims."""
    if vec is None:
        return None
    v = torch.nn.functional.normalize(vec.float(), dim=-1)
    v = v[:256] if v.shape[-1] >= 256 else torch.nn.functional.pad(v, (0, 256 - v.shape[-1]))
    return [float(x) for x in v.cpu()]

def compute_margin(logits_flat):
    """Margin between top-2 tokens (or 0 if only one token)."""
    if logits_flat.numel() < 2:
        return 0.0
    top2 = torch.topk(logits_flat, k=2, largest=True).values
    return float(top2[0] - top2[1])

def compute_entropy(probs):
    """Entropy of the distribution"""
    if probs.numel() == 0:
        return 0.0
    eps = 1e-10
    probs_safe = torch.clamp(probs, min=eps)
    return float(-torch.sum(probs_safe * torch.log(probs_safe)))

def collect_features(model, question: str, dataset: str, context: str = None, device="cuda"):
    """
    Generate answer and collect all features for probe training
    Returns dict with features and generated answer
    """
    # Build standard prompt
    prompt = build_prompt(question, dataset, context)
    
    # Generate with internals
    try:
        result = model.generate_with_states(prompt)
        if result is None:
            raise ValueError("generate_with_states returned None")
        inp, gen = result
    except Exception as e:
        raise RuntimeError(f"generate_with_states failed: {e}\nPrompt: {prompt[:200]}...")
    
    # Extract generated text
    start_pos = inp["input_ids"].shape[-1]
    new_ids = gen.sequences[:, start_pos:]
    generated_text = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
    
    # Extract answer from standard format
    answer = extract_answer_standard(generated_text, dataset)
    
    # Get hidden states via teacher-forced forward pass (like original)
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
        
        # Extract hidden states from the generated portion only
        gen_len = new_ids.shape[-1]
        
        # Last layer
        hs_final = out.hidden_states[-1][0]  # [T, d]
        h_ans = hs_final[-gen_len:]
        h_last = h_ans[-1] if h_ans.shape[0] > 0 else None
        h_pool = h_ans.mean(dim=0) if h_ans.shape[0] > 0 else None
        
        # Middle layer
        mid_ix = len(out.hidden_states) // 2
        hs_mid = out.hidden_states[mid_ix][0]
        h_ans_mid = hs_mid[-gen_len:]
        h_last_mid = h_ans_mid[-1] if h_ans_mid.shape[0] > 0 else None
        h_pool_mid = h_ans_mid.mean(dim=0) if h_ans_mid.shape[0] > 0 else None
    
    # Pack hidden states to 256 dims
    h_last_256 = pack_256(h_last) if h_last is not None else None
    h_pool_256 = pack_256(h_pool) if h_pool is not None else None
    h_last_mid_256 = pack_256(h_last_mid) if h_last_mid is not None else None
    h_pool_mid_256 = pack_256(h_pool_mid) if h_pool_mid is not None else None
    
    # Compute token-level features from scores
    scores = gen.scores  # List of [1, vocab_size] tensors
    
    if len(scores) > 0:
        # Stack scores: [seq_len, vocab_size]
        all_logits = torch.stack([s[0] for s in scores], dim=0)
        all_probs = torch.softmax(all_logits, dim=-1)
        
        # Compute per-token features
        entropies = []
        margins = []
        lps = []
        top_probs = []
        
        for i, (logit_row, prob_row) in enumerate(zip(all_logits, all_probs)):
            # Entropy
            entropies.append(compute_entropy(prob_row))
            
            # Margin
            margins.append(compute_margin(logit_row))
            
            # Log prob of chosen token
            chosen_id = new_ids[0, i].item()
            lps.append(float(torch.log(prob_row[chosen_id] + 1e-10)))
            
            # Top-1 probability
            top_probs.append(float(prob_row.max()))
        
        # Aggregate features
        entropy_mean = float(torch.tensor(entropies).mean())
        entropy_std = float(torch.tensor(entropies).std())
        margin_mean = float(torch.tensor(margins).mean())
        margin_min = float(torch.tensor(margins).min())
        lp_mean = float(torch.tensor(lps).mean())
        seq_conf = float(torch.tensor(top_probs).mean())
    else:
        entropy_mean = 0.0
        entropy_std = 0.0
        margin_mean = 0.0
        margin_min = 0.0
        lp_mean = 0.0
        seq_conf = 0.0
    
    # Build feature dict
    features = {
        # Scalar features (probability-based, format-agnostic)
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "margin_mean": margin_mean,
        "margin_min": margin_min,
        "lp_mean": lp_mean,
        "seq_conf": seq_conf,
        "model_confidence": None,  # Not available in standard format
        
        # Answer metadata
        "answer_len": len(answer.split()),
        "parsed_json_ok": False,  # N/A for standard format
        "parsed_p_true_ok": False,  # N/A for standard format
        "is_unknown": answer.lower() in ("unknown", "unanswerable"),
        
        # Rescore features (optional, could compute later)
        "rescore_logp": lp_mean,  # Use mean as proxy
        
        # Hidden state features (256-dim each)
        "h_last_256": h_last_256,
        "h_pool_256": h_pool_256,
        "h_last_mid_256": h_last_mid_256,
        "h_pool_mid_256": h_pool_mid_256,
        
        # Generated output
        "raw_output": generated_text,
        "parsed_answer": answer,
    }
    
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["llama31", "llama32", "qwen", "gemma", "gpt"], required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    
    args = parser.parse_args()
    
    # Suppress transformers warnings
    hf_logging.set_verbosity_error()
    
    print("="*80)
    print("COLLECT INTERNALS - STANDARD PROMPTS VERSION")
    print("="*80)
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model_id}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Shard: {args.shard_id}/{args.num_shards}")
    print(f"Output: {args.out}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    if args.backend == "llama31":
        model = Llama31_8B(
            model_id=args.model_id,
            dtype="float16",
            device_map=None,
            max_new_tokens=args.max_new_tokens,
            cache_dir=None
        )
    elif args.backend == "llama32":
        model = Llama32_11B(
            model_id=args.model_id,
            dtype="float16",
            device_map=None,
            max_new_tokens=args.max_new_tokens,
            cache_dir=None
        )
    elif args.backend == "qwen":
        model = Qwen7B(
            model_id=args.model_id,
            dtype="float16",
            device_map=None,
            max_new_tokens=args.max_new_tokens,
            cache_dir=None
        )
    elif args.backend == "gemma":
        model = Gemma12B(
            model_id=args.model_id,
            dtype="float16",
            device_map=None,
            max_new_tokens=args.max_new_tokens,
            cache_dir=None
        )
    elif args.backend == "gpt":
        model = GPTOSS(
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    
    print(f"Model loaded on: {next(model.model.parameters()).device}")
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    dataset = load_qa(args.dataset, args.split, args.limit)
    print(f"Loaded {len(dataset)} examples")
    
    # Shard the dataset - use select() for HuggingFace Dataset compatibility
    if args.num_shards > 1:
        shard_size = len(dataset) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = start_idx + shard_size if args.shard_id < args.num_shards - 1 else len(dataset)
        
        # Use select() method for HuggingFace Dataset objects
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(start_idx, end_idx))
        else:
            dataset = dataset[start_idx:end_idx]
        
        print(f"Processing shard {args.shard_id}: examples {start_idx} to {end_idx} ({len(dataset)} rows)")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    
    # Process examples
    print("\nProcessing examples...")
    with open(args.out, "w") as fout:
        for idx in tqdm(range(len(dataset)), desc="Collecting"):
            # Access example properly regardless of dataset type
            example = dataset[idx]
            
            # Handle case where dataset[idx] returns a dict vs a weird format
            if not isinstance(example, dict):
                print(f"\nWarning: Unexpected example format at {idx}: {type(example)}")
                print(f"Example content: {example}")
                continue
            
            question = example.get("question", "")
            gold_answers = example.get("answers", [])
            context = example.get("context", None)
            
            if not question:
                print(f"\nWarning: Empty question at index {idx}")
                continue
            
            # Collect features
            try:
                features = collect_features(
                    model, 
                    question, 
                    args.dataset, 
                    context
                )
                
                # Check correctness
                parsed_answer = features["parsed_answer"]
                correct = any(
                    str(gold).lower() in parsed_answer.lower()
                    for gold in gold_answers
                    if parsed_answer
                )
                
                # Write to output
                output_record = {
                    "question": question,
                    "gold_answers": gold_answers,
                    "context": context,
                    "correct": int(correct),
                    **features
                }
                
                fout.write(json.dumps(output_record) + "\n")
                
            except Exception as e:
                print(f"\nError on example {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n✓ Done! Wrote to: {args.out}")
    print("="*80)

if __name__ == "__main__":
    main()