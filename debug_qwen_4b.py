#!/usr/bin/env python3
"""
Debug script to check Qwen3-4B model outputs
Shows first N examples with full details
"""

import sys
sys.path.insert(0, '/home/claude/downloads/723_project')

from src.models.qwen7b import Qwen7B
from src.data.datasets import load_qa

# Prompt templates (copied from eval_probe_optimized.py)
FEWSHOT_PROMPT_DEFAULT = """You are a helpful AI assistant. Answer questions accurately and provide a confidence score.

Example 1:
Question: What is the capital of France?
Response: {{"answer": "Paris", "p_true": 0.95}}

Example 2:
Question: Who wrote Romeo and Juliet?
Response: {{"answer": "William Shakespeare", "p_true": 0.98}}

Example 3:
Question: What is 15 * 24?
Response: {{"answer": "360", "p_true": 0.92}}

Now answer this question. Respond ONLY with a JSON object containing "answer" and "p_true" (a number between 0 and 1 indicating your confidence). Do NOT include any other text.

Question: {EVAL_QUESTION}
Response: """

def check_model_outputs(dataset_name="mmlu", n_examples=20):
    """Check model outputs on a dataset"""
    
    print("=" * 80)
    print(f"QWEN3-4B OUTPUT INSPECTION")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Showing first {n_examples} examples")
    print("=" * 80)
    
    # Initialize model
    print("\nInitializing Qwen3-4B-Instruct...")
    model = Qwen7B(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        dtype="float16",
        device_map=None,
        max_new_tokens=64,
        cache_dir=None
    )
    print()
    
    # Load dataset - returns tuple of (dataset_dict, split_name)
    if dataset_name == "mmlu":
        result = load_qa("mmlu", "validation", n_examples)
    elif dataset_name == "gsm8k":
        result = load_qa("gsm8k", "test", n_examples)
    elif dataset_name == "hotpotqa":
        result = load_qa("hotpotqa", "validation", n_examples)
    else:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    # Handle result - could be tuple or dict
    if isinstance(result, tuple):
        dataset = result[0]
    else:
        dataset = result
    
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset length: {len(dataset)}")
    print()
    
    # Check first n examples
    correct = 0
    total = 0
    
    # Iterate properly through HuggingFace dataset
    for i in range(min(n_examples, len(dataset))):
        example = dataset[i]
        
        # Now example should be a dict
        if not isinstance(example, dict):
            print(f"ERROR: Unexpected example format at {i}: {type(example)}")
            continue
        
        question = example.get("question", "")
        gold_answers = example.get("answers", [])
        
        if not question:
            print(f"WARNING: Empty question at index {i}")
            continue
        
        # Build prompt
        prompt = FEWSHOT_PROMPT_DEFAULT.format(EVAL_QUESTION=question)
        
        # Generate
        try:
            inp, gen = model.generate_with_states(prompt)
            start_pos = inp["input_ids"].shape[-1]
            new_ids = gen.sequences[:, start_pos:]
            raw_output = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"ERROR generating for example {i}: {e}")
            continue
        
        # Check correctness (simple substring match)
        is_correct = False
        if gold_answers and raw_output:
            is_correct = any(
                str(gold).lower() in raw_output.lower() 
                for gold in gold_answers
            )
        
        if is_correct:
            correct += 1
        total += 1
        
        # Print details
        print(f"{'='*80}")
        print(f"Example {i+1}/{n_examples}")
        print(f"{'-'*80}")
        print(f"Question: {question[:150]}{'...' if len(question) > 150 else ''}")
        print(f"\nGold answers: {gold_answers}")
        print(f"\nModel output (raw):")
        print(f"  {raw_output}")
        print(f"\nCorrect: {'✓' if is_correct else '✗'}")
        print()
    
    print("=" * 80)
    print(f"SUMMARY")
    print("=" * 80)
    if total > 0:
        print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")
    else:
        print("No examples processed!")
    print()
    
    # Show raw bytes of first output to check for unicode issues
    if total > 0 and len(dataset) > 0:
        print("=" * 80)
        print("RAW BYTES CHECK (First output):")
        print("=" * 80)
        
        first_example = dataset[0]
        if isinstance(first_example, dict):
            first_question = first_example.get("question", "")
            if first_question:
                try:
                    inp, gen = model.generate_with_states(
                        FEWSHOT_PROMPT_DEFAULT.format(EVAL_QUESTION=first_question)
                    )
                    start_pos = inp["input_ids"].shape[-1]
                    new_ids = gen.sequences[:, start_pos:]
                    raw_output = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
                    print(f"String: {raw_output[:100]}")
                    print(f"Bytes:  {raw_output[:100].encode('utf-8')}")
                except Exception as e:
                    print(f"ERROR: {e}")
        print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmlu", "gsm8k", "hotpotqa"], default="mmlu")
    parser.add_argument("--n", type=int, default=20, help="Number of examples to check")
    args = parser.parse_args()
    
    check_model_outputs(args.dataset, args.n)