# # scripts/prepare_data.py
# import argparse
# from src.conf.config import Cfg
# from src.data.datasets import load_qa

# """Minimal data prep: ensures the dataset is downloaded/cached.
# If rag.backend == 'faiss', this is where you'd build a local index.
# """

# def main(cfg: Cfg):
#     ds = load_qa(cfg.data.dataset, cfg.data.split, cfg.data.limit)
#     print(f"Prepared dataset: {cfg.data.dataset}/{cfg.data.split} with {len(ds)} examples (limit={cfg.data.limit})")
#     if cfg.rag.backend.lower() != "faiss":
#         print("Note: RAG backend is not 'faiss'; no index built. Wikipedia backend will be used at runtime.")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
#     args = ap.parse_args(); cfg = Cfg.load(args.config); main(cfg)


#!/usr/bin/env python3
# scripts/prepare_data.py
import argparse
import sys
from pathlib import Path

from transformers.utils import logging as hf_logging
from datasets import load_dataset, DownloadMode

from src.conf.config import Cfg

hf_logging.set_verbosity_info()
hf_logging.enable_propagation()

SUPPORTED = {
    # name_in_cfg -> (hf_dataset, hf_config or None)
    "trivia_qa": ("trivia_qa", "rc"),
    "triviaqa": ("trivia_qa", "rc"),
    "hotpot_qa": ("hotpot_qa", "distractor"),
    "hotpotqa": ("hotpot_qa", "distractor"),
    "squad": ("squad", None),
    "squad_v2": ("squad_v2", None),
    "squad2": ("squad_v2", None),
    "nq_open": ("nq_open", None),
    "gsm8k": ("gsm8k", "main"),
    "mmlu": ("cais/mmlu", "all"),
    # if you ever want full NQ, uncomment next line (heavier, noisier)
    # "natural_questions": ("natural_questions", None),
}

def resolve_hf_spec(name_from_cfg: str):
    key = (name_from_cfg or "").lower()
    if key not in SUPPORTED:
        raise ValueError(
            f"Unsupported dataset for warmup: {name_from_cfg!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED))}"
        )
    return SUPPORTED[key]

def warmup(hf_name: str, hf_config: str, split: str, streaming: bool = False, force_redownload: bool = False):
    """
    Download + cache the dataset locally (Arrow cache). Does NOT touch your src/data code.
    """
    kwargs = {
        "name": hf_config,
        "split": split,
        "streaming": streaming,
    }
    if hf_config is None:
        kwargs.pop("name")

    # ensure cache is written to disk (avoid fully streaming)
    dl_mode = DownloadMode.FORCE_REDOWNLOAD if force_redownload else DownloadMode.REUSE_DATASET_IF_EXISTS

    ds = load_dataset(hf_name, **kwargs, download_mode=dl_mode)

    # Touch a few examples so the Arrow files are built & indexed (materialize)
    try:
        n = len(ds)  # triggers indexing
        _ = ds[0] if n > 0 else None
        _ = ds[min(10, max(0, n - 1))] if n > 0 else None
    except TypeError:
        # Some streaming configs don't support len(); fallback to small iteration
        it = iter(ds)
        for _ in range(16):
            try:
                next(it)
            except StopIteration:
                break

    # Report a size if available
    try:
        size = len(ds)
        print(f"[prepare_data] Warmed '{hf_name}'"
              f"{'/' + hf_config if hf_config else ''} split='{split}' | {size} examples (cached).")
    except Exception:
        print(f"[prepare_data] Warmed '{hf_name}'"
              f"{'/' + hf_config if hf_config else ''} split='{split}' (streaming, cached shards).")

def main():
    ap = argparse.ArgumentParser(description="Warm up HF datasets cache (no dataloader changes).")
    ap.add_argument("--config", default="configs/default.yaml", help="Path to YAML config (reads data.dataset & data.split).")
    ap.add_argument("--dataset", default=None, help="Override dataset name (e.g., hotpot_qa, squad, nq_open).")
    ap.add_argument("--split", default=None, help="Override split (e.g., train, validation).")
    ap.add_argument("--force-redownload", action="store_true", help="Force re-download (ignore existing cache).")
    args = ap.parse_args()

    # load your existing config
    cfg = Cfg.load(args.config)

    # resolve dataset + split (allow CLI override)
    ds_name_cfg = (args.dataset or getattr(cfg.data, "dataset", None) or "").strip()
    split = (args.split or getattr(cfg.data, "split", None) or "").strip() or "validation"

    try:
        hf_name, hf_config = resolve_hf_spec(ds_name_cfg)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    # print where config came from
    print(f"[prepare_data] Using config: {args.config}")
    print(f"[prepare_data] Dataset={ds_name_cfg} -> hf='{hf_name}'"
          f"{'/' + hf_config if hf_config else ''}, split='{split}'")

    # Warm the cache
    warmup(
        hf_name=hf_name,
        hf_config=hf_config,
        split=split,
        streaming=False,
        force_redownload=args.force_redownload,
    )

    # friendly hint about retrieval (no indexing here)
    print("[prepare_data] Done. Note: this script does not build any retrieval index; "
          "it only ensures datasets are downloaded & cached.")

if __name__ == "__main__":
    main()