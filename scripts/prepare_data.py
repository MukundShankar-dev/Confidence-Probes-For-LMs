# scripts/prepare_data.py
import argparse
from src.conf.config import Cfg
from src.data.datasets import load_qa

"""Minimal data prep: ensures the dataset is downloaded/cached.
If rag.backend == 'faiss', this is where you'd build a local index.
"""

def main(cfg: Cfg):
    ds = load_qa(cfg.data.dataset, cfg.data.split, cfg.data.limit)
    print(f"Prepared dataset: {cfg.data.dataset}/{cfg.data.split} with {len(ds)} examples (limit={cfg.data.limit})")
    if cfg.rag.backend.lower() != "faiss":
        print("Note: RAG backend is not 'faiss'; no index built. Wikipedia backend will be used at runtime.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(); cfg = Cfg.load(args.config); main(cfg)
