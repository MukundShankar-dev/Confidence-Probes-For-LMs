from datasets import load_dataset
from typing import Dict, Iterable
import random
import re, string

def _normalize(s: str) -> str:
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
    def lower(t): return t.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def squad_em(pred: str, refs) -> bool:
    if not isinstance(refs, (list, tuple)): refs = [refs]
    p = _normalize(pred)
    return any(p == _normalize(r) for r in refs)

def squad_f1(pred: str, refs) -> float:
    if not isinstance(refs, (list, tuple)): refs = [refs]
    def toks(s): return _normalize(s).split()
    p = toks(pred)
    best = 0.0
    for r in refs:
        g = toks(r)
        if not p and not g: best = max(best, 1.0); continue
        common = set(p) & set(g)
        num_same = sum(min(p.count(w), g.count(w)) for w in common)
        if num_same == 0: best = max(best, 0.0); continue
        prec, rec = num_same/len(p), num_same/len(g)
        f1 = 2*prec*rec/(prec+rec)
        best = max(best, f1)
    return best

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
