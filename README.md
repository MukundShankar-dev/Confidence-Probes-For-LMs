# GPT-OSS Confidence & Grounding

White-box pipeline for: hidden-state probe → confidence gate → RAG/refusal, built around GPT‑OSS‑20B.

## Install
See root `requirements.txt` then edit `configs/default.yaml`.

## Other settings (for cache)
```mkdir -p language_models/{transformers,datasets,hf_home,xdg,torch,torch_extensions,sentence-transformers}

# Hugging Face caches
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
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
1. Prepare data using `python scripts/prepare_data.py --config configs/default.yaml`
2. Baseline: `python scripts/run_baseline.py --config configs/default.yaml`
3. Train probe: `python scripts/run_probe_training.py --config configs/default.yaml`
4. Gate+RAG eval: `python scripts/run_gate_eval.py --config configs/default.yaml`

## Notes
- For probing, prefer Transformers (not vLLM) to access `hidden_states`.
- Wikipedia backend uses online API; for offline/large-scale, implement FAISS in `src/rag/retriever.py`.
- Swap models by editing `model.model_id`.