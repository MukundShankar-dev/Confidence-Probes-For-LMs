#!/usr/bin/env python3
"""
Prefetch a Hugging Face model into the HF cache (HF_HOME) so distributed jobs
don't all try to download it.

Usage:
  python prepare_model.py --model-id Qwen/Qwen2.5-7B-Instruct
  python prepare_model.py --model-id google/gemma-7b-it
  # For gated models, ensure you're logged in (huggingface-cli login) or set HF_TOKEN.
"""

import argparse
import os
from huggingface_hub import snapshot_download, login

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="HF repo id, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--revision", default=None, help="Optional branch/tag/commit (e.g. main)")
    ap.add_argument("--token", default=None, help="HF token; else uses HF_TOKEN env or existing login")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Allow continuing if some files fail (rare edge cases)")
    args = ap.parse_args()

    # If token is present, ensure we're authenticated (safe no-op if already logged in)
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    print(f"[prepare_model] HF_HOME={os.environ.get('HF_HOME', '(default)')}")
    print(f"[prepare_model] Downloading {args.model_id} (rev={args.revision or 'default'}) into cache...")

    try:
        local_path = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            resume_download=True,
            force_download=False,      # don't re-download unchanged files
            local_files_only=False,    # actually hit the hub if missing
            token=token,
        )
        print(f"[prepare_model] Cached at: {local_path}")
    except Exception as e:
        if args.allow_partial:
            print(f"[prepare_model] WARNING: partial/failed download but continuing. Error: {e}")
        else:
            raise

    # Optional: quick presence check without network
    try:
        from transformers import AutoTokenizer
        _ = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
        print("[prepare_model] Transformers can resolve tokenizer from cache (local_files_only=True).")
    except Exception as e:
        print(f"[prepare_model] NOTE: tokenizer local check failed (may be model-only repo): {e}")

if __name__ == "__main__":
    main()
