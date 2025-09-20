from datasets import load_dataset
from typing import Dict, Iterable, List
import re, string

# -------------------- normalization + metrics --------------------

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
        if not p and not g:
            best = max(best, 1.0); continue
        common = set(p) & set(g)
        num_same = sum(min(p.count(w), g.count(w)) for w in common)
        if num_same == 0:
            best = max(best, 0.0); continue
        prec, rec = num_same/len(p), num_same/len(g)
        f1 = 2*prec*rec/(prec+rec)
        best = max(best, f1)
    return best

# -------------------- loaders --------------------

def _dedupe_nonempty(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if not x: continue
        x = x.strip()
        if not x: continue
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def load_qa(name: str, split: str, limit: int):
    """
    Return a dataset with unified fields:
      {"question": str, "answers": List[str], "context": str}
    - SQuAD: keeps context (for RC later).
    - NQ-Open: wraps single 'answer' into a list.
    - TriviaQA (unfiltered): includes primary value + aliases.
    """
    if name == "squad":
        ds = load_dataset("squad", split=split)

        def _fmt(ex):
            # HF SQuAD schema: ex["answers"] = {"text": [..], "answer_start": [..]}
            ans_list = ex["answers"]["text"] if isinstance(ex.get("answers"), dict) and "text" in ex["answers"] else ex.get("answers", [])
            if isinstance(ans_list, str): ans_list = [ans_list]
            return {
                "question": ex["question"],
                "answers": _dedupe_nonempty(list(ans_list)),
                "context": ex.get("context", "") or "",
            }
        ds = ds.map(_fmt, remove_columns=ds.column_names)

    elif name == "nq_open":
        ds = load_dataset("nq_open", split=split)

        def _fmt(ex):
            ans = ex.get("answer", "")
            ans_list = [ans] if isinstance(ans, str) else (ans or [])
            return {
                "question": ex["question"],
                "answers": _dedupe_nonempty(list(ans_list)),
                "context": "",
            }
        ds = ds.map(_fmt, remove_columns=ds.column_names)

    elif name == "triviaqa":
        # HuggingFace dataset name is "trivia_qa", config "unfiltered"
        ds = load_dataset("trivia_qa", "unfiltered", split=split)

        def _fmt(ex):
            # ex["answer"] may be dict with keys {"value": str, "aliases": [str, ...]}
            refs = []
            ans = ex.get("answer", {})
            if isinstance(ans, dict):
                v = ans.get("value")
                if isinstance(v, str) and v: refs.append(v)
                aliases = ans.get("aliases", [])
                if isinstance(aliases, (list, tuple)):
                    refs.extend([a for a in aliases if isinstance(a, str) and a])
            elif isinstance(ans, str) and ans:
                refs.append(ans)
            return {
                "question": ex["question"],
                "answers": _dedupe_nonempty(refs),
                # keep blank for closed-book; you can wire search/evidence later if needed
                "context": "",
            }
        ds = ds.map(_fmt, remove_columns=ds.column_names)

    else:
        raise ValueError(f"Unknown dataset {name}")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds
