# GPT-OSS Confidence & Grounding

White-box pipeline for: hidden-state probe → confidence gate → RAG/refusal. Code is currently integrated for `Llama-3.1-8B`, `Qwen-2.5-7B`, and `Gemma-3-7B`. 

Repository Structure: 
```
723_project/
├─ configs/
│  └─ default.yaml
├─ src/
│  ├─ models/            # model wrappers/backends (e.g., Qwen7B, Gemma12B, Llama31_8B, etc.)
│  └─ eval/              # metrics, scoring (e.g., evaluate_batch, squad_em (exact match), squad_f1 (Beier Score))
├─ scripts/
│  ├─ prepare_data.py   # Loads dataset once so that it is in cache for future use
│  ├─ prepare_model.py  # Loads model so that it is in cache, need to specify --model_id
│  ├─ run_baseline.py   # Runs baseline. More details below
│  └─ merge_results.py  # Used after running baseline if dataset was sharded
├─ results/
│  └─ triviaqa/         # Results from run_baseline.py (and merge_results.py)
│     ├─ gemma/
│     ├─ llama31/
│     └─ qwen/
├─ README.md            # This file :)
└─ requirements.txt
```

## Other settings (for cache)
```mkdir -p language_models/{transformers,datasets,hf_home,xdg,torch,torch_extensions,sentence-transformers}

# Hugging Face caches (needed on cluster specifically)
export HF_HOME="$PWD/language_models/hf_home"
export TRANSFORMERS_CACHE="$PWD/language_models/transformers"
export HF_DATASETS_CACHE="$PWD/language_models/datasets"
export XDG_CACHE_HOME="$PWD/language_models/xdg"

# PyTorch caches / JIT kernels
export TORCH_HOME="$PWD/language_models/torch"
export TORCH_EXTENSIONS_DIR="$PWD/language_models/torch_extensions"

# SentenceTransformers (if you use it)
export SENTENCE_TRANSFORMERS_HOME="$PWD/language_models/sentence-transformers"

# (Optional) quieter tokenizer threads
export TOKENIZERS_PARALLELISM=false
```

Run: 

```
python -m venv .venv && source .venv/bin/activate    # (can also be done via conda env and python 3.9 if preferred)
pip install -r requirements.txt
```

## Run
1. Prepare data using `python scripts/prepare_data.py --config configs/default.yaml`
2. Prepare model using `python scripts/prepare_model.py --model-id=...`. Here, `model-id` must be in `[meta-llama/Meta-Llama-3.1-8B-Instruct, google/gemma-3-12b-it, Qwen/Qwen2.5-7B-Instruct]`. You may need to log into huggingface CLI.
2. Baseline. The commands to run the baseline are below. Fix num shards to 1 and shard_id to 0 if running on single machine.
```
python -m scripts.run_baseline \
  --config configs/default.yaml \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --thinking_mode off \
  --prompt_style fewshot_fixed \
  --max_new_tokens 64 \
  --num_shards "$num_shards" \
  --shard_id "$shard_id" \
  --save_jsonl "results/triviaqa/qwen/qwen_final_only_shard${shard_id}.json"

python -m scripts.run_baseline \
  --config configs/default.yaml \
  --backend gemma \
  --model_id "google/gemma-3-12b-it" \
  --thinking_mode off \
  --prompt_style fewshot_fixed \
  --max_new_tokens 64 \
  --num_shards "$num_shards" \
  --shard_id "$shard_id" \
  --save_jsonl "results/triviaqa/gemma/gemma_final_only_shard${shard_id}.json"

python -m scripts.run_baseline \
  --config configs/default.yaml \
  --backend llama31 \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --thinking_mode off \
  --prompt_style fewshot_fixed \
  --max_new_tokens 64 \
  --num_shards $num_shards \
  --shard_id $shard_id \
  --save_jsonl results/triviaqa/llama31/8b_final_only_shard${shard_id}.json
```
3. If ran on multiple machines (sharded dataset), also run `python -m scripts.merge_results results/triviaqa/MODEL_ID`.
<!-- 3. Train probe: `python scripts/run_probe_training.py --config configs/default.yaml` (needs to be updated) -->
<!-- 4. Gate+RAG eval: `python scripts/run_gate_abstain.py --config configs/default.yaml`   (needs to be updated) -->

## Notes
`run_baseline.py` provides model output, exact match (Exact Match) score, F1 (Brier Score), ground truths, `seq_conf` (geometric mean token probability over the generated answer, excluding the last step), `conf_entropy_mean` (smoother, length-agnostic version). Taking something like `final_conf = 0.7 * seq_conf + 0.3 * conf_entropy_mean` would be beneficial as a "confidence score".
<!-- - For probing, prefer Transformers (not vLLM) to access `hidden_states`. -->
<!-- - Swap models by editing `model.model_id`. -->

## Fix list
1. Update datasets (add Google NQA and Hot Pot QA), make baseline configurable for this also.
2. Compare: model confidence scores (from outputs), entropy gains (`seq_conf` and `conf_entropy_mean`) to actual correctness (use mainly `EM`, `F1` is not entirely accurate).
            --> For example, "16 million" gets F1 of 0.5 if ground truth is "18 million", when it is actually fully incorrect.
3. Look into training probes - how, which layers, why? 
4. How do we deal with the model obstaining? Do we penalize? 