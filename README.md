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

NOTE: STEPS 2-4 ARE OPTIONAL. SKIP TO 6 IF YOU WANT.

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

4. Now that baseline has been ran, verify results (can be found in results/) and make sure they make sense.

5. Now collect model internals using the python calls which can be found in `run_collecion.slurm`.

6. Run the following command which generates train/val/test splits using all 3 datasets pooled, per model (from run_collection results): 
```
python -m scripts.make_probe_split \
  --root data/probe \
  --out_dir data/probe_splits_80_10_10 \
  --train_frac 0.80 --val_frac 0.10 --test_frac 0.10
```

7. Train all types of available probes for all current models (qwen and llama at the moment):
```
python -m scripts.train_probe --data_dir data/probe_splits_80_10_10 --models llama,qwen --use_hidden --out_dir probes/ --probe_type all --xform_dump_attn --xform_epochs 40 --xform_calibrate --xform_cal_debug
```

8. Run the interactive demo using (adjust model and probe parameters as needed):
```
python -m scripts.demo --model llama31 --probe_dir probes/backend_llama/probe_mlp --use_hidden --interactive
```

## Notes
`run_baseline.py` provides model output, exact match (Exact Match) score, F1 (Brier Score), ground truths, `seq_conf` (geometric mean token probability over the generated answer, excluding the last step), `conf_entropy_mean` (smoother, length-agnostic version). Taking something like `final_conf = 0.7 * seq_conf + 0.3 * conf_entropy_mean` would be beneficial as a "confidence score".

We currently have the following types of probes:
  - Logistic regression
  - Logistic regression with calibration
  - MLP
  - Decision Tree
  - Small transformer

From preliminary results, MLP is the best, while transformer is the most stable/consistent. The rest are garbage. 

<!-- - For probing, prefer Transformers (not vLLM) to access `hidden_states`. -->
<!-- - Swap models by editing `model.model_id`. -->

## Fix list
1. Our datasets are s.t. the LMs mostly get answers wrong. This means that we have pretty hefty class imbalance. Try to find an easy dataset which these LMs will answer all if not most correct or just make one synthetically (using gpt or something) to inflate the positive numbers where models get answers correct.

2. Figure out nice ways to visualize and interpret the stuff inside training reports (found in subdirectories of `probes/`). The `best_threshold` and `best_f1` also would probably go up when we fix (1). 

3. See if there are more complex or clever probes we can make instead of just look at specific parts of model and learn. 

4. Find out what most important statistics from the ones we collected are. Attention maps (in `probes/*/probe_xform`) are a good place to start and we can also probably expand on the logistic regressions to see which features are identified as important.

5. Make the demo nicer (we can probably do a live demo during presentation then).