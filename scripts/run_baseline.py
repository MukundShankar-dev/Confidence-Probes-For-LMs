# scripts/run_baseline.py
import argparse
import time
import torch
import os
import json
import re
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from src.conf.config import Cfg
from src.models.gpt_oss import GPTOSS
from src.models.qwen7b import Qwen7B
from src.models.gemma12b import Gemma12B
from src.models.llama31_8b import Llama31_8B
from src.models.llama32_11b import Llama32_11B
from src.data.datasets import load_qa
from src.eval.metrics import evaluate_batch, squad_em, squad_f1

from src.probe.runtime import ProbeRuntime

import math
import sys
import ast
from typing import Optional, Tuple

# Allow fast math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---------------- Prompts ----------------

ZERO_SHOT_FINAL_ONLY = (
    "Answer with only the short factual span (1–5 words). No punctuation.\n"
    "Q: {q}\nA: "
)

# For datasets that ALLOW 'Unknown' (SQuAD v2)
FEWSHOT_FIXED_SQUADV2 = """You are answering trivia and reading comprehension questions.
Return only a single JSON object with fields:
- "answer": the short factual span (1–5 words, no punctuation).
  If the question cannot be answered from the given context, use "Unknown".
- "p_true": the probability (0.0–1.0) that this answer is correct (round to two decimals).

Calibration guidance:
- Report your true probability; do NOT inflate.
- Overconfidence is penalized by proper scoring (Brier). If unsure, choose a lower value.
- If multiple plausible answers exist or the question is ambiguous, reduce p_true appropriately.
- If no answer is supported by evidence, output "Unknown" with an appropriately low p_true.

Q: Who wrote Hamlet?
{{"answer": "William Shakespeare", "p_true": 0.95}}

Q: What is the capital of France?
{{"answer": "Paris", "p_true": 0.92}}

Q: What is the capital of South Africa?
{{"answer": "Pretoria", "p_true": 0.60}}

Q: Which element has the symbol 'Au'?
{{"answer": "Aluminum", "p_true": 0.15}}

Q: Who authored the Voynich Manuscript?
{{"answer": "Unknown", "p_true": 0.20}}

Q: Which planet is known as the Red Planet?
{{"answer": "Mars", "p_true": 0.85}}

Q: {EVAL_QUESTION}
"""

# For answerable datasets (TriviaQA / HotpotQA / NQ-Open): discourage 'Unknown'
FEWSHOT_FIXED_ANSWERABLE = """You are answering trivia questions.
Return only a single JSON object with fields:
- "answer": the short factual span (1–5 words, no punctuation). Do NOT output "Unknown" for this dataset; instead provide your best answer span.
- "p_true": the probability (0.0–1.0) that this answer is correct (round to two decimals).

Calibration guidance:
- Report your true probability; do NOT inflate.
- If you are unsure, still provide your best span answer and set a lower p_true (e.g., 0.15–0.35).
- If multiple plausible answers exist or the question is ambiguous, reduce p_true appropriately.

Q: Who wrote Hamlet?
{{"answer": "William Shakespeare", "p_true": 0.95}}

Q: What is the capital of France?
{{"answer": "Paris", "p_true": 0.92}}

Q: What is the capital of South Africa?
{{"answer": "Pretoria", "p_true": 0.60}}

Q: Which element has the symbol 'Au'?
{{"answer": "Aluminum", "p_true": 0.15}}

Q: Which planet is known as the Red Planet?
{{"answer": "Mars", "p_true": 0.85}}

Q: {EVAL_QUESTION}
"""

FEWSHOT_FIXED_THINKING = """You are answering trivia questions. Be concise.

Q: Who wrote Hamlet?
Analysis:
- Briefly consider candidates; Shakespeare wrote Hamlet.
Final:
William Shakespeare

Q: What is the capital of France?
Analysis:
- European capitals; France → Paris.
Final:
Paris

Q: In which year did the Titanic sink?
Analysis:
- Titanic sank in the North Atlantic in April 1912.
Final:
1912

Q: The chemical symbol for gold is?
Analysis:
- Periodic table: gold → Au.
Final:
Au

Q: Which planet is known as the Red Planet?
Analysis:
- The Red Planet → Mars.
Final:
Mars

Q: {EVAL_QUESTION}
Analysis:
- Briefly reason about the answer.
Final:
"""

# Single, consistent global flag used below
_JSON_ERR_SEEN = False

def softmax_entropy(logits):
    # logits: torch.FloatTensor [V]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # numerical guard: clamp in log
    logp = torch.log(probs.clamp_min(1e-12))
    ent = -(probs * logp).sum().item()
    return float(ent)

def _softmax_entropy_vec(logits_vec: torch.Tensor) -> float:
    probs = torch.nn.functional.softmax(logits_vec, dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    return float(-(probs * logp).sum().item())

def _clamp01_from_any(x):
    try:
        s = str(x).strip().replace("％", "%")
        if s.endswith("%"):
            v = float(s[:-1])
        else:
            v = float(s)
    except Exception:
        return None
    if v > 1.0:
        v = v / 100.0
    return float(max(0.0, min(1.0, v)))

def _norm_answer_text(ans):
    if isinstance(ans, (list, tuple)):
        for a in ans:
            if isinstance(a, str) and a.strip():
                return a.strip()
        return str(ans[0]) if ans else ""
    return ans.strip() if isinstance(ans, str) else (str(ans) if ans is not None else "")

def extract_answer_and_model_conf(text: str):
    """
    Parse model output into (answer, p_true).
    Canonical key: 'p_true'. Also accepts confidence/probability variants and CJK keys.
    Robust to:
      - JSON
      - single-quoted Python dicts
      - JSON encoded as a STRING (need double-decode)
      - over-escaped quotes/backslashes like {\"answer\":\"...\", \\\"p_true\\\":0.70}
    """
    import json, re, sys
    global _JSON_ERR_SEEN

    ANSWER_KEYS = {
        "answer", "final", "prediction",
        "答案", "回答", "最终", "最終", "解答",
    }
    PROB_KEYS = {
        "p_true",
        # english variants
        "confidence", "conference", "conf", "confidence_score",
        "confidencelevel", "confidence_level", "prob", "probability", "p",
        # chinese/japanese
        "信心", "置信度", "置信", "概率", "可能性", "信心分数", "信心分", "自信度",
    }

    t = (text or "").strip()

    # strip code fences
    if t.startswith("```"):
        t = t.lstrip("`")
        if "\n" in t:
            t = t.split("\n", 1)[1].strip()
        if t.endswith("```"):
            t = t[:-3].strip()

    # Extract the outermost {...} slice if present
    start, end = t.find("{"), t.rfind("}")
    candidate = t[start:end+1] if (start != -1 and end != -1 and end > start) else t

    def _parse_mapping(maybe_dict_text: str):
        """
        Try a sequence of increasingly aggressive parses.
        Returns a dict or None.
        """
        s = maybe_dict_text

        # attempt up to 4 passes (handles JSON-as-string / nested escaping)
        for _ in range(4):
            # 1) strict JSON
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, str):
                    # JSON string containing JSON -> unwrap and loop
                    s = obj
                    continue
            except Exception:
                pass

            # 2) python literal (single quotes, etc.)
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, str):
                    s = obj
                    continue
            except Exception:
                pass

            # 3) naive repair of over-escaped sequences, then try JSON again
            repaired = s
            # collapse double-escaping first
            repaired = repaired.replace('\\\\\\"', '\\"')
            repaired = repaired.replace('\\\\"', '\\"')
            # then unescape \" -> "
            repaired = repaired.replace('\\"', '"')
            # and \\ -> \
            repaired = repaired.replace('\\\\', '\\')

            if repaired != s:
                s = repaired
                # loop again with the repaired string
                continue

            # nothing changed; break out
            break

        return None

    def _from_mapping(obj):
        # Try exact keys (including CJK) then lower-cased fallbacks
        ans = ""
        p_true = None

        # answer
        for k, v in obj.items():
            kl = str(k).strip().lower()
            if kl in ANSWER_KEYS or str(k) in ANSWER_KEYS:
                ans = _norm_answer_text(v)
                break
        if not ans:
            norm = {str(k).strip().lower(): v for k, v in obj.items()}
            for k in ANSWER_KEYS:
                if k in norm:
                    ans = _norm_answer_text(norm[k]); break

        # probability: prefer p_true, then variants
        for k, v in obj.items():
            if str(k).strip().lower() == "p_true":
                p_true = _clamp01_from_any(v); break
        if p_true is None:
            for k, v in obj.items():
                kl = str(k).strip().lower()
                if kl in PROB_KEYS or str(k) in PROB_KEYS:
                    p_true = _clamp01_from_any(v); break
        if p_true is None:
            norm = {str(k).strip().lower(): v for k, v in obj.items()}
            for k in ["p_true"] + [x for x in PROB_KEYS if x != "p_true"]:
                if k in norm:
                    p_true = _clamp01_from_any(norm[k]); break

        return ans, p_true

    # Try parsing as a mapping with aggressive fallbacks
    obj = _parse_mapping(candidate)
    if isinstance(obj, dict):
        return _from_mapping(obj)
    else:
        # Warn once if our best parse failed
        try:
            if not _JSON_ERR_SEEN:
                print("[warn] Could not parse model JSON; falling back to regex.", file=sys.stdout, flush=True)
                _JSON_ERR_SEEN = True
        except Exception:
            pass

    # Regex fallback (tolerates both ' and " and %/％, and some escaping)
    m = re.search(
        r"""(?ix)
        [\{\[]?        # optional leading brace
        .*?
        (["']|\\?["'])?(?:answer|答案|回答|最终|最終|解答)\1?\s*:\s*(["'])(?P<ans>.*?)\2
        .*?
        (["']|\\?["'])?(?:p_true|conf(?:idence|erence|idence_score)?|prob(?:ability)?|p|信心|置信度|置信|概率|可能性|信心分数|信心分|自信度)\4?
        \s*:\s*(?P<prob>-?\d+(?:\.\d+)?(?:%|％)?)
        .*?
        [\}\]]?        # optional trailing brace
        """,
        candidate,
        flags=re.DOTALL,
    )
    if m:
        ans = m.group("ans").strip()
        p_true = _clamp01_from_any(m.group("prob"))
        return ans, p_true

    # Legacy fallback: "text [CONF=...]" style
    m = re.search(r"(.*?)(?:\s*\[CONF\s*=\s*([0-9.]+)\s*\]\s*)?$", t)
    if m:
        ans = m.group(1).strip()
        p_true = _clamp01_from_any(m.group(2)) if m.group(2) is not None else None
        return ans, p_true

    # give up: return raw text as answer, None prob
    return t, None


def build_prompt(q: str, style: str, allow_thinking: bool, fewshot_file: str = None,
                 allow_unknown: bool = False) -> str:
    """Build the evaluation prompt based on style + whether we allow thinking, and dataset policy."""
    if allow_thinking:
        if style == "fewshot_fixed":
            return FEWSHOT_FIXED_THINKING.format(EVAL_QUESTION=q)
        elif style == "fewshot_file":
            assert fewshot_file and os.path.exists(fewshot_file), \
                f"--fewshot_file missing or not found: {fewshot_file}"
            with open(fewshot_file, "r", encoding="utf-8") as f:
                template = f.read()
            assert "{EVAL_QUESTION}" in template, \
                "fewshot_file must contain {EVAL_QUESTION} placeholder."
            return template.format(EVAL_QUESTION=q)
        else:
            return (
                "You are answering trivia questions.\n\n"
                "Q: {Q}\nAnalysis:\n- Think briefly.\nFinal:\n"
            ).format(Q=q)
    else:
        if style == "fewshot_fixed":
            template = FEWSHOT_FIXED_SQUADV2 if allow_unknown else FEWSHOT_FIXED_ANSWERABLE
            return template.format(EVAL_QUESTION=q)
        elif style == "fewshot_file":
            assert fewshot_file and os.path.exists(fewshot_file), \
                f"--fewshot_file missing or not found: {fewshot_file}"
            with open(fewshot_file, "r", encoding="utf-8") as f:
                template = f.read()
            assert "{EVAL_QUESTION}" in template, \
                "fewshot_file must contain {EVAL_QUESTION} placeholder."
            return template.format(EVAL_QUESTION=q)
        else:
            return ZERO_SHOT_FINAL_ONLY.format(q=q)

def parse_analysis_final(text: str):
    """Extract 'Analysis:' ... 'Final:' blocks from plain text."""
    t = re.sub(r"\r", "", text)
    m = re.search(r"Analysis:\s*(.*?)\s*Final:\s*(.*)", t, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return "", text.strip()
    analysis = m.group(1).strip()
    final = m.group(2).strip()
    final = re.split(r"\n(?:Analysis:|Final:)", final)[0].strip()
    return analysis, final

@torch.inference_mode()
def _rescore_answer_mean_logprob(model, prompt: str, answer: str) -> Optional[float]:
    """
    Try to compute the teacher-forced mean token logprob of `answer` under `prompt`.
    Returns None if the backend doesn't expose a raw HF model interface.
    """
    try:
        tok = model.tok
        hf_model = getattr(model, "model", None)
        if hf_model is None:
            return None

        # Tokenize prompt and answer
        enc_prompt = tok(prompt, return_tensors="pt", add_special_tokens=True)
        enc_ans = tok(answer, return_tensors="pt", add_special_tokens=False)

        # Concatenate
        input_ids = torch.cat([enc_prompt["input_ids"], enc_ans["input_ids"]], dim=1)
        attn = torch.ones_like(input_ids)

        # Labels: ignore prompt tokens, supervise only answer tokens
        labels = input_ids.clone()
        labels[:, :enc_prompt["input_ids"].shape[1]] = -100  # ignore prompt via -100

        # Move to the model device (best guess: first param device)
        dev = next(hf_model.parameters()).device
        input_ids = input_ids.to(dev)
        attn = attn.to(dev)
        labels = labels.to(dev)

        out = hf_model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
        # out.loss is mean over non -100; also compute explicit mean logprob:
        # mean_nll = loss.item(); mean_logprob = -mean_nll
        mean_logprob = -float(out.loss.item())
        return mean_logprob
    except Exception:
        return None

# --------------- Main ----------------

def main(cfg: Cfg, args):
    hf_logging.set_verbosity_info()
    hf_logging.enable_propagation()

    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")

    # ---------- Backend-specific overrides ----------
    backend = args.backend
    prompt_style_effective = args.prompt_style
    thinking_mode_effective = args.thinking_mode
    allow_thinking_effective = args.allow_thinking

    if backend in ("qwen", "gemma"):
        # Force FINAL-ONLY + FEWSHOT_FIXED for plain LMs
        prompt_style_effective = "fewshot_fixed"
        thinking_mode_effective = "off"
        allow_thinking_effective = False

    # Dataset policy: allow 'Unknown' only on SQuAD v2
    ds_name = cfg.data.dataset.lower()
    allow_unknown = ("squad" in ds_name and "v2" in ds_name) or (ds_name in {"squad_v2", "squad2", "squad2.0"})

    max_new_tokens = args.max_new_tokens or cfg.model.max_new_tokens
    force_final = not allow_thinking_effective  # final-only if we DON'T allow thinking
    reasoning = args.reasoning

    # Thinking mode dispatch
    if thinking_mode_effective == "off":
        use_chat = True
        force_final = True
        parse_harmony = False
        parse_textual = False
    elif thinking_mode_effective == "textual":
        use_chat = False
        force_final = False
        parse_harmony = False
        parse_textual = True
    else:  # "harmony"
        use_chat = True
        force_final = False
        parse_harmony = True
        parse_textual = False

    # Model backend
    if backend == "gpt":
        model_id = args.model_id or cfg.model.model_id
        model = GPTOSS(
            model_id,
            cfg.model.dtype,
            cfg.model.device_map,
            max_new_tokens=max_new_tokens,
            use_router_probs=getattr(cfg.model, "use_router_probs", True),
            cache_dir=getattr(cfg.model, "cache_dir", None),
            use_chat_template=use_chat,
            reasoning_effort=reasoning,
            force_final_prefix=force_final,          # final-only vs think-then-final (Harmony)
            final_allowance=args.final_allowance,
            analysis_cap=args.analysis_cap,
        )
    elif backend == "qwen":
        model_id = args.model_id or "Qwen/Qwen2.5-7B-Instruct"
        model = Qwen7B(
            model_id=model_id,
            dtype=getattr(cfg.model, "dtype", "float16"),
            device_map=None,                         # full GPU (no offload)
            max_new_tokens=max_new_tokens,
            cache_dir=getattr(cfg.model, "cache_dir", None),
        )
    elif backend == "gemma":
        model_id = args.model_id or "google/gemma-3-12b-it"
        model = Gemma12B(
            model_id=model_id,
            dtype=getattr(cfg.model, "dtype", "float16"),
            device_map=None,
            max_new_tokens=max_new_tokens,
            cache_dir=getattr(cfg.model, "cache_dir", None),
        )
    elif backend == "llama31":
        model_id = args.model_id or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model = Llama31_8B(
            model_id=model_id,
            dtype=getattr(cfg.model, "dtype", "float16"),
            device_map=None,
            max_new_tokens=max_new_tokens,
            cache_dir=getattr(cfg.model, "cache_dir", None),
        )
    elif backend == "llama32":
        model_id = args.model_id or "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = Llama32_11B(
            model_id=model_id,
            dtype=getattr(cfg.model, "dtype", "float16"),
            device_map=None,
            max_new_tokens=max_new_tokens,
            cache_dir=getattr(cfg.model, "cache_dir", None),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # === dataset ===
    # Allow CLI overrides for dataset/split without editing YAML (handled in __main__)
    limit = args.limit or cfg.data.limit
    ds = load_qa(cfg.data.dataset, cfg.data.split, limit)

    if args.num_shards > 1:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_id, contiguous=True)
        print(f"[shard] Using shard {args.shard_id}/{args.num_shards} with {len(ds)} examples")

    probe = ProbeRuntime(args.probe_path) if args.probe_path else None

    preds, refs, rows = [], [], []
    total_tokens, total_time = 0, 0.0

    # for Brier score accumulation
    brier_sum = 0.0
    brier_n = 0

    bar = tqdm(ds, desc="baseline", dynamic_ncols=True)
    for idx, ex in enumerate(bar):
        q, gold = ex["question"], ex["answers"]
        prompt = build_prompt(q, prompt_style_effective, allow_thinking_effective,
                              args.fewshot_file, allow_unknown=allow_unknown)

        t0 = time.perf_counter()
        inp, gen = model.generate_with_states(prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        start = inp["input_ids"].shape[-1]
        new_ids = gen.sequences[:, start:]

        # Ensure these exist regardless of branch
        raw_text = ""
        model_conf = None

        if parse_harmony:
            analysis_txt, ans = model.split_channels(new_ids[0])
            ans = ans.strip()
            raw_text = ans  # no separate raw here
        elif parse_textual:
            raw_plain = model.tok.decode(new_ids[0], skip_special_tokens=True)
            analysis_txt, ans = parse_analysis_final(raw_plain)
            ans = ans.strip()
            raw_text = raw_plain
        else:
            raw_text = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()
            analysis_txt, ans = "", raw_text
            ans_parsed, model_conf = extract_answer_and_model_conf(raw_text)
            if ans_parsed:
                ans = ans_parsed.strip()

        # -------- confidences / entropies ----------
        entropy_last = None
        entropy_mean = None              # mean over *content* steps (excludes last)
        conf_entropy = None              # exp(-entropy_last) (kept for backward-compat)
        conf_entropy_mean = None         # exp(-mean content entropy)
        seq_conf = None                  # geometric mean token prob over content tokens
        lp_mean = None                   # path mean logprob (redundant w/ seq_conf)
        margin_last = None
        margin_mean = None

        try:
            scores = gen.scores  # list length = #gen tokens; each is [1, vocab]
            T = len(scores)

            if T > 0:
                ents = [_softmax_entropy_vec(s[0].float()) for s in scores]
                entropy_last = float(ents[-1])
                conf_entropy = float(math.exp(-entropy_last))

                # treat the very last step as newline/EOS; exclude for content stats
                content_ents = ents[:-1] if T > 1 else ents
                if content_ents:
                    entropy_mean = float(sum(content_ents) / len(content_ents))
                    conf_entropy_mean = float(math.exp(-entropy_mean))

                # sequence confidence: mean token prob for generated ids, excluding last
                gen_ids_seq = new_ids[0].tolist()
                if T > 1:
                    gen_ids_seq = gen_ids_seq[:-1]
                    scores_iter = scores[:-1]
                else:
                    scores_iter = scores

                if len(gen_ids_seq) == len(scores_iter) and len(gen_ids_seq) > 0:
                    logps = []
                    margins = []
                    for logit_step, tok_id in zip(scores_iter, gen_ids_seq):
                        lsm = torch.nn.functional.log_softmax(logit_step[0].float(), dim=-1)
                        logps.append(float(lsm[tok_id].item()))
                        top2 = torch.topk(lsm, k=2).values
                        margins.append(float(top2[0] - top2[1]))
                    avg_logp = sum(logps) / len(logps)
                    seq_conf = float(math.exp(avg_logp))  # in [0,1]
                    lp_mean = float(avg_logp)
                    margin_last = float(margins[-1]) if margins else None
                    margin_mean = float(sum(margins) / len(margins)) if margins else None

            entropy_std = statistics.pstdev(content_ents) if content_ents else None
            margin_min = min(margins) if margins else None
        except Exception:
            pass

        # -------- answer re-scoring (teacher-forced) --------
        canon_json = None
        try:
            pt = model_conf if model_conf is not None else 0.0
            safe_ans = ans.replace('"', '\\"')
            canon_json = f'{{"answer":"{safe_ans}","p_true":{pt:.2f}}}'
        except Exception:
            pass
        rescore_logp = _rescore_answer_mean_logprob(model, prompt, canon_json if canon_json else ans)

        gen_len = int(new_ids.shape[-1])
        total_tokens += gen_len
        total_time += dt
        tokps = (gen_len / dt) if dt > 0 else float("inf")

        em_i = 1 if squad_em(ans, gold) else 0
        f1_i = squad_f1(ans, gold)
        preds.append(ans)
        refs.append(gold)

        # Brier accumulation if model_conf available
        if model_conf is not None:
            diff = float(model_conf) - float(em_i)
            brier_sum += diff * diff
            brier_n += 1

        if idx == 0 or args.debug_first:
            print("[debug] head:", ans[:200].replace("\n", "\\n"))
            if isinstance(gold, (list, tuple)) and gold:
                print("[debug] gold:", gold[0])
            else:
                print("[debug] gold:", gold)
            print("[debug] model_p_true:", model_conf)
            print("entropy_last:", entropy_last, "| entropy_mean_content:", entropy_mean,
                  "| conf_entropy_last:", conf_entropy, "| conf_entropy_mean:", conf_entropy_mean,
                  "| seq_conf:", seq_conf, "| lp_mean:", lp_mean,
                  "| margin_last:", margin_last, "| margin_mean:", margin_mean,
                  "| rescore_logp:", rescore_logp)
            print("[debug] tail:", ans[-200:].replace("\n", "\\n"))
            print("[debug] raw:", raw_text.replace("\n", "\\n"))

        if args.verbose and (idx % 10 == 0):
            try:
                mem = torch.cuda.max_memory_allocated() / 1e9
                bar.set_postfix(tokens=gen_len, t=f"{dt:.2f}s", tps=f"{tokps:.1f}", vram=f"{mem:.1f}GB")
            except Exception:
                bar.set_postfix(tokens=gen_len, t=f"{dt:.2f}s", tps=f"{tokps:.1f}")

        rows.append({
            "idx": idx,
            "question": q,
            "prediction": ans,
            "em": em_i,
            "f1": f1_i,

            # === confidence / entropy metrics ===
            "entropy_last": entropy_last,
            "entropy_mean": entropy_mean,                # mean over content tokens
            "conf_entropy": conf_entropy,                # exp(-H_last)  (legacy)
            "conf_entropy_mean": conf_entropy_mean,      # exp(-mean H_content)
            "seq_conf": seq_conf,                        # geometric mean token prob over content tokens
            "lp_mean": lp_mean,                          # mean logprob along path
            "margin_last": margin_last,
            "margin_mean": margin_mean,

            "model_confidence": model_conf,              # parsed p_true (if provided)
            "references": gold,
            "gen_tokens": gen_len,
            "gen_time": round(dt, 3),
            "analysis": analysis_txt,
            "raw_output": raw_text,

            # teacher-forced re-scoring
            "rescore_logp": rescore_logp,
        })

        row_feats = {
            "lp_mean": lp_mean,
            "seq_conf": seq_conf,
            "entropy_mean": entropy_mean,
            "entropy_last": entropy_last,
            "entropy_std": entropy_std,          # stddev over content tokens <-- FAIL? 
            "margin_mean": margin_mean,
            "margin_last": margin_last,
            "margin_min": margin_min,        # probe features <-- FAIL?
            "h_last_256": None,           # list[256]
            "h_pool_256": None,           # list[256]
            "h_last_mid_256": None,   # list[256]
            "h_pool_mid_256": None,   # list[256]
            "rescore_logp": rescore_logp,
            "answer_len": int(gen_len),
            "parsed_json_ok": int(ans is not None and len(ans) > 0),
            "parsed_p_true_ok": int(model_conf is not None),
            "is_unknown": int(ans.strip().lower() == "unknown"),
        }

        p_true_probe = probe.predict(row_feats) if probe else None
        rows[-1]["p_true_probe"] = p_true_probe

    overall_metrics = evaluate_batch(preds, refs)

    # Brier finalize
    if brier_n > 0:
        overall_metrics["brier"] = brier_sum / brier_n
        overall_metrics["brier_n"] = brier_n
        print(f"[brier] on {brier_n} examples: {overall_metrics['brier']:.4f}")
    else:
        print("[brier] skipped (no p_true emitted by model)")

    if total_time > 0 and len(preds) > 0:
        print(f"[baseline] avg toks/ex: {total_tokens/len(preds):.1f} | "
              f"avg time/ex: {total_time/len(preds):.2f}s | "
              f"overall toks/s: {total_tokens/total_time:.1f}")

    print(overall_metrics)
    rows.append({"overall": True, **overall_metrics})
    if args.save_jsonl:
        os.makedirs(os.path.dirname(args.save_jsonl) or ".", exist_ok=True)
        with open(args.save_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[baseline] wrote per-example outputs to {args.save_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Override dataset name (triviaqa | squad | squad_v2 | nq_open | hotpot_qa)")
    ap.add_argument("--split", type=str, default=None,
                    help="Override split (train | validation)")
    ap.add_argument("--limit", type=int, default=None, help="override dataset size")
    ap.add_argument("--save_jsonl", type=str, default=None, help="path for per-example outputs")
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--debug_first", action="store_true", default=False)

    # prompt controls
    ap.add_argument("--prompt_style", choices=["zero_shot", "fewshot_fixed", "fewshot_file"], default="fewshot_fixed")
    ap.add_argument("--fewshot_file", type=str, default=None)

    # thinking vs final-only
    ap.add_argument("--allow_thinking", action="store_true", help="Use textual Analysis/Final protocol")
    ap.add_argument("--reasoning", choices=["low", "medium", "high"], default="low")

    # token caps (forwarded to model)
    ap.add_argument("--max_new_tokens", type=int, default=None, help="override cfg.model.max_new_tokens")
    ap.add_argument("--final_allowance", type=int, default=32, help="(think mode) tokens allowed inside final")
    ap.add_argument("--analysis_cap", type=int, default=512, help="(think mode) cap tokens before final")
    ap.add_argument("--thinking_mode", choices=["off", "textual", "harmony"], default="off",
                    help="off=final-only; textual=Analysis/Final delimiters (no Harmony); harmony=think->final with Harmony")

    # backend & model id
    ap.add_argument("--backend", choices=["gpt", "qwen", "gemma", "llama31", "llama32"], default="gpt",
                    help="Select model backend.")
    ap.add_argument("--model_id", type=str, default=None, help="override model id for selected backend")

    ap.add_argument("--num_shards", type=int, default=1, help="for distributed eval (slurm)")
    ap.add_argument("--shard_id", type=int, default=0, help="shard index for distributed eval (slurm)")

    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    # Allow CLI overrides for dataset/split without editing YAML
    if hasattr(args, "dataset") and args.dataset:
        try:
            cfg.data.dataset = args.dataset.strip()
        except Exception:
            pass
    if hasattr(args, "split") and args.split:
        try:
            cfg.data.split = args.split.strip()
        except Exception:
            pass
    main(cfg, args)
