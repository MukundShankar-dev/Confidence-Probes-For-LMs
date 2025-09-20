# GPT-OSS Confidence & Grounding

White-box pipeline for: hidden-state probe → confidence gate → RAG/refusal, built around GPT‑OSS‑20B.
Code currently would work on mac silicone but really don't recommend. Will work much more efficiently if ran on a GPU. 

Current specs: 
    - GPT oss 20b
    - TriviaQA task (metrics are in `results/`).

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
2. Baseline: `python scripts/run_baseline.py --config configs/default.yaml` (finally works but needs to be updated for better inference)
3. Train probe: `python scripts/run_probe_training.py --config configs/default.yaml` (needs to be updated)
4. Gate+RAG eval: `python scripts/run_gate_abstain.py --config configs/default.yaml`   (needs to be updated)

## Notes
- For probing, prefer Transformers (not vLLM) to access `hidden_states`.
- Swap models by editing `model.model_id`.

## Fix list
1. Inference (in `src/models/gpt_oss.py`) skips thinking entirely by bypassing channels and forcing `|final|`. Need to work around this and run proper inference, then extract final section only as final answer. 
2. Double check probe training, gating, etc. 
3. Check if there are more techniques we can add to this repo by looking at papers.