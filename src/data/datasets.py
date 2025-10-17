from datasets import load_dataset, concatenate_datasets
from typing import Dict, Iterable, List, Optional
import re, string
import unicodedata
import os

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
        key = x.lower()
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def _fmt_triviaqa_hf_example(ex):
    ans = ex["answer"]
    cand = [ans.get("value", "")]
    for k in ("aliases", "normalized_aliases"):
        if k in ans and ans[k]:
            cand.extend(a for a in ans[k] if isinstance(a, str))
    # dedupe (case-insensitive), keep order
    return {
        "question": ex.get("question", ""),
        "answers": _dedupe_nonempty(cand),
        "context": "",
    }


def _load_supplementary_jsonl(path: str):
    """
    Load a TriviaQA-like JSONL with fields:
      - QuestionId (ignored)
      - Question: str
      - Answer: { Value: str, Aliases: [str, ...] }  # Aliases optional
      - Evidence: [str, ...]                         # optional, ignored (we keep context="")
    Returns a dataset with unified fields: {question, answers, context}
    """
    ds = load_dataset("json", data_files={"train": path}, split="train")

    def _fmt(ex):
        q = ex.get("Question", "") or ex.get("question", "")
        ans_obj = ex.get("Answer", {}) or {}
        # Some robustness if Answer happens to be a string
        if isinstance(ans_obj, str):
            cand = [ans_obj]
        else:
            cand = []
            v = ans_obj.get("Value") or ans_obj.get("value") or ""
            if isinstance(v, str) and v:
                cand.append(v)
            aliases = ans_obj.get("Aliases") or ans_obj.get("aliases") or []
            if isinstance(aliases, (list, tuple)):
                cand.extend([a for a in aliases if isinstance(a, str)])
        return {
            "question": q,
            "answers": _dedupe_nonempty(cand) or ["Unknown"],
            "context": "",  # we stay closed-book for consistency
        }

    return ds.map(_fmt, remove_columns=ds.column_names)


def load_qa(
    name: str,
    split: str,
    limit: int,
    supplementary_data: Optional[str] = None,
    supplementary_only: bool = False,
):
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

    Supplement handling:
      - If `supplementary_only` is True, return ONLY the supplementary JSONL on the requested split.
      - If `supplementary_data` is provided, append it to the loaded split (train/validation/test).
      - If neither flag is set, no supplementary examples are used.
    """

    # ---- supplementary-only short-circuit (works on any split) ----
    if supplementary_only:
        if not supplementary_data:
            # try project default
            default_path = "src/data/supplementary_data.jsonl"
            if os.path.exists(default_path):
                supplementary_data = default_path
        if not supplementary_data or not os.path.exists(supplementary_data):
            raise FileNotFoundError(
                "Supplementary JSONL not found. Pass --supplementary_data "
                "or place it at src/data/supplementary_data.jsonl"
            )
        ds = _load_supplementary_jsonl(supplementary_data)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ds

    # ---- main dataset ----
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
            # HF: ex["answers"]["text"] is [] for unanswerables
            ans_list = []
            if isinstance(ex.get("answers"), dict):
                ans_list = ex["answers"].get("text", []) or []
            # Canonicalize no-answer so references is never empty
            if not ans_list:
                ans_list = ["Unknown"]  # sentinel aligning with prompt/metrics
            return {
                "question": ex.get("question", ""),
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
        ds = ds.map(_fmt_triviaqa_hf_example, remove_columns=ds.column_names)

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

    # ---- optional supplementary (append to ANY split when provided) ----
    if supplementary_data:
        if not os.path.exists(supplementary_data):
            raise FileNotFoundError(
                f"Supplementary JSONL not found: {supplementary_data}")
        supp_ds = _load_supplementary_jsonl(supplementary_data)
        ds = concatenate_datasets([ds, supp_ds])

    # ---- limit AFTER merge ----
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds
