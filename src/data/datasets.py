from datasets import load_dataset
from typing import Dict, Iterable
import random


def NORMALIZE(s): return " ".join(s.lower().split())


def load_qa(name: str, split: str, limit: int):
    if name == "squad":
        ds = load_dataset("squad", split=split)
        # unify fields

        def _fmt(ex):
            return {
                "question": ex["question"],
                "answers": [ans["text"] for ans in ex["answers"]["answers"]] if isinstance(ex["answers"], dict) and "answers" in ex["answers"] else ex["answers"]["text"],
                "context": ex.get("context", ""),
            }
        ds = ds.map(_fmt, remove_columns=ds.column_names)
    elif name == "nq_open":
        ds = load_dataset("nq_open", split=split)
        ds = ds.rename_columns({"answer": "answers"})
    elif name == "triviaqa":
        ds = load_dataset("trivia_qa", "unfiltered", split=split)

        def _fmt(ex):
            return {
                "question": ex["question"],
                "answers": [ex["answer"]["value"]],
                "context": "",
            }
        ds = ds.map(_fmt, remove_columns=ds.column_names)

    else:
        raise ValueError(f"Unknown dataset {name}")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))
        return ds


def exact_match(pred: str, refs: Iterable[str]) -> bool:
    p = NORMALIZE(pred)
    for r in refs:
        if NORMALIZE(r) in p or p in NORMALIZE(r):
            return True
    return False
