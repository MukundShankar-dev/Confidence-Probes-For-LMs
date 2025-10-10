from datasets import load_dataset
from typing import Dict, Iterable, List
import re, string
import unicodedata

UNICODE_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212\u2015"
UNICODE_SPACES = "\u00A0\u2007\u202F"

def _normalize(s: str) -> str:
    # 1) NFKC fold (turns many unicode punct to ASCII)
    s = unicodedata.normalize("NFKC", s)
    # 2) normalize odd spaces/dashes
    for ch in UNICODE_SPACES:
        s = s.replace(ch, " ")
    for ch in UNICODE_DASHES:
        s = s.replace(ch, " ")
    # 3) lowercase trim
    s = s.strip()
    return s

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

def _split_variants(s: str) -> Iterable[str]:
    """
    Split string into multiple answer variants by common separators; strip punctuation.
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return []
    # Common delimiters for multiple answers / aliases
    parts = re.split(r"\s*[;/,]\s*|\s+or\s+|\s+aka\s+|\s+aka\.\s+", s, flags=re.IGNORECASE)
    out = []
    for p in parts:
        t = p.strip()
        if t:
            # strip trailing punctuation like ".", "," etc.
            t = t.strip(string.punctuation + " ")
            if t:
                out.append(t)
    return out

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
    Return a HuggingFace Dataset where each example is a dict:
      { "question": str, "answers": List[str], "context": str }
    NOTE: We keep `context` as an empty string for closed-book runs.

    Supported names in this file:
      - "squad"        -> huggingface 'squad'
      - "squad_v2"     -> huggingface 'squad_v2'
      - "nq_open"      -> huggingface 'nq_open'
      - "triviaqa"     -> huggingface 'trivia_qa' (unfiltered)
      - "hotpotqa"/"hotpot_qa" -> huggingface 'hotpot_qa' ('distractor' config)
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

    elif name in {"squad_v2", "squad2", "squad2.0"}:
        ds = load_dataset("squad_v2", split=split)

        def _fmt(ex):
            ans_list = ex["answers"]["text"] if isinstance(ex.get("answers"), dict) and "text" in ex["answers"] else ex.get("answers", [])
            if isinstance(ans_list, str): ans_list = [ans_list]
            return {
                "question": ex["question"],
                "answers": _dedupe_nonempty(list(ans_list)),
                "context": ex.get("context", "") or "",
            }
        ds = ds.map(_fmt, remove_columns=ds.column_names)

    elif name in {"nq_open", "natural_questions_open", "naturalquestions_open"}:
        ds = load_dataset("nq_open", split=split)

        def _fmt(ex):
            ans_list = ex.get("answer", [])
            if isinstance(ans_list, str): ans_list = [ans_list]
            return {
                "question": ex["question"],
                "answers": _dedupe_nonempty(list(ans_list)),
                "context": "",
            }
        ds = ds.map(_fmt, remove_columns=ds.column_names)

    elif name == "triviaqa":
        ds = load_dataset("trivia_qa", "unfiltered", split=split)
        def _fmt(ex):
            ans = ex["answer"]
            cand = [ans["value"]]
            for k in ("aliases", "normalized_aliases"):
                if k in ans and ans[k]:
                    cand.extend(a for a in ans[k] if isinstance(a, str))
            # dedupe (case-insensitive), keep order
            seen = set(); out = []
            for a in cand:
                s = a.strip()
                key = s.lower()
                if s and key not in seen:
                    seen.add(key); out.append(s)
            return {"question": ex["question"], "answers": out, "context": ""}
        ds = ds.map(_fmt, remove_columns=ds.column_names)

    elif name in {"hotpotqa", "hotpot_qa"}:
        # Multi-hop HotpotQA; use 'distractor' config. Closed-book: keep context empty.
        ds = load_dataset("hotpot_qa", "distractor", split=split)
        def _fmt(ex):
            return {
                "question": ex.get("question", ""),
                "answers": _dedupe_nonempty([ex.get("answer", "")]),
                "context": "",
            }
        ds = ds.map(_fmt, remove_columns=ds.column_names)

    else:
        raise ValueError(f"Unknown dataset {name}")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds
