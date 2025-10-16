#!/usr/bin/env python3
"""
demo.py — Ask a question, run the LM, build *the exact same features* as in data collection,
load the trained per-backend probe, and print the model JSON + probe P(correct).

Key point: This reproduces the feature pipeline from collect_internals.py so the probe
sees the same statistics (generation-time token stats, teacher-forced hidden states,
and canonical JSON rescoring).

Usage:
  python -m scripts.demo --model llama31 --probes_dir probes/
  python -m scripts.demo --model qwen    --probes_dir probes/
Options:
  --max_new_tokens 64
  --use_hidden                 (compute 4×256 hidden-state packs; slower)
  --model_id <HF id>           (override default HF model id)
"""

import argparse, json, math, os, sys, time, re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import joblib

# ----------------------------- Config -----------------------------

MODEL_MAP = {
    "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
}
BACKEND_KEY = {"llama31": "llama", "qwen": "qwen"}

# Enforce the very same few-shot style used during data collection (note the double braces)
FEWSHOT_FIXED_FINAL_ONLY_DEFAULT = """You are answering trivia questions.
Return only a single JSON object with fields:
- "answer": the short factual span (1–5 words, no punctuation)
- "p_true": the probability (0.0–1.0) that this answer is correct (round to two decimals)

Calibration guidance:
- Report your true probability; do NOT inflate.
- Overconfidence is penalized by proper scoring (Brier). If unsure, choose a lower value.
- If multiple plausible answers exist or the question is ambiguous, reduce p_true appropriately.

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

def build_prompt(q: str) -> str:
    return FEWSHOT_FIXED_FINAL_ONLY_DEFAULT.format(EVAL_QUESTION=q)

# ----------------------------- JSON parsing (same tolerance) -----------------------------

_JSON_WARN_ONCE = False

def safe_json_extract(text: str):
    """Best-effort JSON extractor consistent with data collection."""
    global _JSON_WARN_ONCE
    t = text.strip()

    # strip code fences
    if t.startswith("```"):
        t = t.lstrip("`")
        if "\n" in t:
            t = t.split("\n", 1)[1].strip()
        if t.endswith("```"):
            t = t[:-3].strip()

    # un-escape if double-escaped
    try:
        if "\\\"" in t or "\\\\\"" in t:
            t_try = bytes(t, "utf-8").decode("unicode_escape")
            if "{" in t_try and "}" in t_try:
                t = t_try
    except Exception:
        pass

    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception as e:
            if not _JSON_WARN_ONCE:
                print(f"[warn] JSON parse failed: {e}\nraw: {t[:200]}", file=sys.stderr)
                _JSON_WARN_ONCE = True
    return None

def extract_answer_and_prob(text: str):
    """Parse {"answer": ..., "p_true": ...}. Returns (answer_text, p_model or None)."""
    obj = safe_json_extract(text)
    if isinstance(obj, dict):
        norm = {str(k).strip().lower(): v for k, v in obj.items()}
        ans = (norm.get("answer") or norm.get("final") or norm.get("prediction") or "").strip()
        p = None
        for key in ["p_true", "confidence", "conf", "prob", "probability", "信心", "confidence_score", "confidence_level"]:
            if key in norm:
                try:
                    p = float(norm[key])
                except Exception:
                    try:
                        p = float(str(norm[key]).replace("%", "")) / 100.0
                    except Exception:
                        pass
                break
        if p is not None:
            p = float(max(0.0, min(1.0, p if p <= 1.0 else p/100.0)))
        return ans, p
    return text.strip(), None

# ----------------------------- Stats identical to training -----------------------------

def token_logprobs_for_targets(logits, target_ids):
    logps = []
    for logit_step, tok_id in zip(logits, target_ids):
        lsm = torch.log_softmax(logit_step[0].float(), dim=-1)
        logps.append(float(lsm[int(tok_id)].item()))
    return logps

def compute_generation_stats(gen_scores: List[torch.Tensor], gen_ids_seq: List[int]) -> Dict[str, float]:
    """
    Mirror feature construction from collect_internals:
    - Use all T steps from generation scores
    - Compute entropy & margins per step
    - Compute per-step log-prob for the chosen token
    - For mean/std/min aggregates, exclude the final step (content_len = max(1, T-1))
    """
    T = len(gen_scores)
    content_len = max(1, T - 1) if T > 1 else T

    step_ent, margins, step_logp = [], [], []
    for t in range(T):
        logits = gen_scores[t][0].float()
        top2 = torch.topk(logits, k=2).values
        margins.append(float(top2[0] - top2[1]))

        probs = torch.softmax(logits, dim=-1)
        ent = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        step_ent.append(ent)

        tok_id = gen_ids_seq[t] if t < len(gen_ids_seq) else None
        if tok_id is not None:
            lsm = torch.log_softmax(logits, dim=-1)
            step_logp.append(float(lsm[tok_id].item()))

    entropy_mean = float(sum(step_ent[:content_len]) / content_len) if step_ent else None
    entropy_last = float(step_ent[-1]) if step_ent else None
    entropy_std = float(torch.tensor(step_ent[:content_len]).std().item()) if len(step_ent[:content_len]) > 1 else None

    margin_mean = float(sum(margins[:content_len]) / content_len) if margins else None
    margin_last = float(margins[-1]) if margins else None
    margin_min = float(min(margins[:content_len])) if content_len > 0 and margins else None

    lp_mean = float(sum(step_logp[:content_len]) / content_len) if step_logp else None
    seq_conf = math.exp(lp_mean) if lp_mean is not None else None

    return {
        "lp_mean": lp_mean,
        "seq_conf": seq_conf,
        "entropy_mean": entropy_mean,
        "entropy_last": entropy_last,
        "entropy_std": entropy_std,
        "margin_mean": margin_mean,
        "margin_last": margin_last,
        "margin_min": margin_min,
    }

def rescore_mean_logprob(hf_causal_lm, tok, prompt_text: str, target_text: str):
    """
    Teacher-forced mean log-prob for the *canonical JSON* appended to the prompt.
    """
    with torch.no_grad():
        enc_prompt = tok(prompt_text, add_special_tokens=False, return_tensors="pt")
        enc_target = tok(target_text, add_special_tokens=False, return_tensors="pt")
        input_ids = torch.cat([enc_prompt["input_ids"], enc_target["input_ids"]], dim=-1).to(hf_causal_lm.device)
        attn = torch.ones_like(input_ids)
        out = hf_causal_lm(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
        logits = out.logits[:, :-1, :]  # next-token preds
        target_slice = input_ids[:, enc_prompt["input_ids"].shape[-1]:]
        logits_tgt = logits[:, -target_slice.shape[-1]:, :]
        step_logps = []
        for t in range(target_slice.shape[-1]):
            lsm = torch.log_softmax(logits_tgt[0, t].float(), dim=-1)
            step_logps.append(float(lsm[int(target_slice[0, t])].item()))
        return sum(step_logps) / max(1, len(step_logps))

def pack_256(vec: torch.Tensor):
    """L2-normalize then take/pad to first 256 dims."""
    if vec is None:
        return None
    v = torch.nn.functional.normalize(vec.float(), dim=-1)
    if v.shape[-1] >= 256:
        v = v[:256]
    else:
        v = torch.nn.functional.pad(v, (0, 256 - v.shape[-1]))
    return [float(x) for x in v.detach().cpu()]

# ----------------------------- Model adapters -----------------------------

def load_adapter(args):
    """
    Prefer the same wrappers used in data collection. If not importable, fall back
    to a minimal HF adapter with similar generation settings.
    """
    # Try project's Llama31 adapter
    if args.model == "llama31":
        try:
            from src.models.llama31_8b import Llama31_8B  # same as collection
            mid = args.model_id or MODEL_MAP["llama31"]
            return Llama31_8B(model_id=mid, dtype="float16", device_map=None, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"[info] Could not import project adapter for llama31, falling back to HF: {e}", file=sys.stderr)

    # Try project's Qwen adapter
    if args.model == "qwen":
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
            class _StopOnStrings(torch.nn.Module):
                def __init__(self, tok, stop_strings: List[str]):
                    super().__init__()
                    self.tok = tok
                    self.stop_ids = [tok.encode(s, add_special_tokens=False) for s in stop_strings]
                    self.start_len = None
                def set_start_len(self, n): self.start_len = int(n)
                def __call__(self, input_ids, scores, **kwargs):
                    if self.start_len is None: return False
                    seq = input_ids[0].tolist()
                    for pat in self.stop_ids:
                        L = len(pat)
                        if L and len(seq) >= L and seq[-L:] == pat: return True
                    return False

            class QwenAdapter:
                def __init__(self, model_id, dtype="float16", device_map=None, max_new_tokens=64, cache_dir=None):
                    torch_dtype = getattr(torch, dtype)
                    self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch_dtype, device_map=device_map,
                        trust_remote_code=True, low_cpu_mem_usage=True, attn_implementation="sdpa"
                    ).to("cuda").eval()
                    if getattr(self.model.config, "pad_token_id", None) is None:
                        self.model.config.pad_token_id = self.tok.eos_token_id
                    self.max_new = int(max_new_tokens)
                    self._stopper = _StopOnStrings(self.tok, ["\nQ:", " Q:", "\nUser:", "\nAssistant:"])

                @torch.no_grad()
                def generate_with_states(self, prompt: str):
                    enc = self.tok(prompt, return_tensors="pt")
                    enc = {k: v.to(self.model.device) for k, v in enc.items()}
                    start = int(enc["input_ids"].shape[-1])
                    self._stopper.set_start_len(start)
                    gen = self.model.generate(
                        **enc,
                        max_new_tokens=self.max_new,
                        do_sample=False,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.05,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_hidden_states=False,
                        pad_token_id=self.tok.eos_token_id,
                        eos_token_id=self.tok.eos_token_id,
                        use_cache=True,
                        stopping_criteria=StoppingCriteriaList([self._stopper]),
                        min_new_tokens=2,
                    )
                    return enc, gen
                def split_channels(self, generated_ids: torch.Tensor):
                    return "", self.tok.decode(generated_ids, skip_special_tokens=True).strip()

            mid = args.model_id or MODEL_MAP["qwen"]
            return QwenAdapter(mid, dtype="float16", device_map=None, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"[info] Could not construct Qwen adapter, falling back to generic HF: {e}", file=sys.stderr)

    # Generic HF fallback
    from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

    class _StopOnStrings(StoppingCriteria):
        def __init__(self, tok, stop_strings: List[str]):
            self.tok = tok
            self.stop_ids = [tok.encode(s, add_special_tokens=False) for s in stop_strings]
            self.start_len = None
        def set_start_len(self, n): self.start_len = int(n)
        def __call__(self, input_ids, scores, **kwargs):
            if self.start_len is None: return False
            seq = input_ids[0].tolist()
            for pat in self.stop_ids:
                L = len(pat)
                if L and len(seq) >= L and seq[-L:] == pat:
                    return True
            return False

    class HFAdapter:
        def __init__(self, model_id, dtype="float16", device_map=None, max_new_tokens=64):
            torch_dtype = getattr(torch, dtype)
            self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            ).to("cuda").eval()
            if getattr(self.model.config, "pad_token_id", None) is None:
                self.model.config.pad_token_id = self.tok.eos_token_id
            self.max_new = int(max_new_tokens)
            self._stopper = _StopOnStrings(self.tok, ["\nQ:", " Q:", "\nUser:", "\nAssistant:"])

        @torch.no_grad()
        def generate_with_states(self, prompt: str):
            enc = self.tok(prompt, return_tensors="pt")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            start = int(enc["input_ids"].shape[-1])
            self._stopper.set_start_len(start)
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.max_new,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([self._stopper]),
                min_new_tokens=2,
            )
            return enc, gen

        def split_channels(self, generated_ids: torch.Tensor):
            text = self.tok.decode(generated_ids, skip_special_tokens=True).strip()
            return "", text

    mid = args.model_id or MODEL_MAP.get(args.model, MODEL_MAP["llama31"])
    return HFAdapter(mid, dtype="float16", device_map=None, max_new_tokens=args.max_new_tokens)

# ----------------------------- Probe features -----------------------------

SCALAR_KEYS = [
    "model_confidence", "lp_mean", "seq_conf",
    "entropy_mean", "entropy_last", "entropy_std",
    "margin_mean", "margin_last", "margin_min",
    "rescore_logp", "answer_len",
    "parsed_json_ok", "parsed_p_true_ok", "is_unknown",
]
VECTOR_KEYS = ["h_last_256", "h_pool_256", "h_last_mid_256", "h_pool_mid_256"]

def to_1d_256(arr: Optional[List[float]]) -> Optional[np.ndarray]:
    if arr is None: return None
    a = np.asarray(arr, dtype=float).ravel()
    if a.size < 256: a = np.pad(a, (0, 256 - a.size))
    elif a.size > 256: a = a[:256]
    return a

def build_feature_row(record: Dict[str, Any], feat_names: List[str]) -> List[float]:
    expanded = {}
    # scalars
    for k in SCALAR_KEYS:
        v = record.get(k, np.nan)
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            v = np.nan
        expanded[k] = float(v) if v is not np.nan else np.nan
    # vectors
    for base in VECTOR_KEYS:
        vec = to_1d_256(record.get(base))
        if vec is None:
            for i in range(256):
                expanded[f"{base}_{i}"] = np.nan
        else:
            for i, val in enumerate(vec):
                expanded[f"{base}_{i}"] = float(val)
    return [expanded.get(name, np.nan) for name in feat_names]

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama31","qwen"], required=True)
    ap.add_argument("--model_id", type=str, default=None, help="override HF model id")
    ap.add_argument("--probes_dir", type=str, default="probes")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--use_hidden", action="store_true")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Load adapter (tries project wrapper first)
    adapter = load_adapter(args)
    tok = adapter.tok
    hf_causal_lm = adapter.model
    device = next(hf_causal_lm.parameters()).device

    # Load probe
    probe_dir = Path(args.probes_dir) / f"backend_{BACKEND_KEY[args.model]}"
    if not probe_dir.exists():
        print(f"[ERR] Probe directory not found: {probe_dir}", file=sys.stderr)
        sys.exit(1)
    probe = joblib.load(probe_dir / "probe_model.joblib")
    feat_names = json.load(open(probe_dir / "feature_names.json", "r", encoding="utf-8"))

    print("Type your trivia question (Ctrl-C to quit).")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue

        prompt = build_prompt(q)

        t0 = time.perf_counter()
        inp, gen = adapter.generate_with_states(prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        start = inp["input_ids"].shape[-1]
        new_ids = gen.sequences[:, start:]
        raw_text = tok.decode(new_ids[0], skip_special_tokens=True).strip()

        # Parse model JSON
        ans_text, p_model = extract_answer_and_prob(raw_text)
        if not ans_text:
            ans_text = raw_text

        # === TOKEN-LEVEL STATS (generation-time), EXACTLY LIKE TRAINING ===
        scores = gen.scores or []
        gen_ids_seq = new_ids[0].tolist()
        stats = compute_generation_stats(scores, gen_ids_seq)

        # === HIDDEN STATES (teacher-forced over prompt+generated), EXACT MATCH ===
        with torch.no_grad():
            full_ids = torch.cat([inp["input_ids"].to(device), new_ids.to(device)], dim=-1)
            attn = torch.ones_like(full_ids)
            out = hf_causal_lm(input_ids=full_ids, attention_mask=attn,
                                output_hidden_states=True, use_cache=False, return_dict=True)
            hs_final = out.hidden_states[-1][0]  # [T, d]
            gen_len = new_ids.shape[-1]
            h_ans = hs_final[-gen_len:]
            h_last = h_ans[-1] if h_ans.shape[0] > 0 else None
            h_pool = h_ans.mean(dim=0) if h_ans.shape[0] > 0 else None

            mid_ix = len(out.hidden_states) // 2
            hs_mid = out.hidden_states[mid_ix][0]
            h_ans_mid = hs_mid[-gen_len:]
            h_last_mid = h_ans_mid[-1] if h_ans_mid.shape[0] > 0 else None
            h_pool_mid = h_ans_mid.mean(dim=0) if h_ans_mid.shape[0] > 0 else None

        h_last_256 = pack_256(h_last) if h_last is not None else None
        h_pool_256 = pack_256(h_pool) if h_pool is not None else None
        h_last_mid_256 = pack_256(h_last_mid) if h_last_mid is not None else None
        h_pool_mid_256 = pack_256(h_pool_mid) if h_pool_mid is not None else None

        # === RESCORE CANONICAL JSON (exact formatting) ===
        safe_ans = ans_text.replace('"', '\\"')
        pt = p_model if p_model is not None else 0.00
        canon_json = f'{{"answer":"{safe_ans}","p_true":{pt:.2f}}}'
        rescore_lp = rescore_mean_logprob(hf_causal_lm, tok, prompt, canon_json)

        # === Assemble feature row exactly as training ===
        row = {
            "model_confidence": p_model,
            "lp_mean": stats["lp_mean"],
            "seq_conf": stats["seq_conf"],
            "entropy_mean": stats["entropy_mean"],
            "entropy_last": stats["entropy_last"],
            "entropy_std": stats["entropy_std"],
            "margin_mean": stats["margin_mean"],
            "margin_last": stats["margin_last"],
            "margin_min": stats["margin_min"],
            "rescore_logp": rescore_lp,
            "answer_len": int(gen_len),
            "parsed_json_ok": int(ans_text is not None and len(ans_text) > 0),
            "parsed_p_true_ok": int(p_model is not None),
            "is_unknown": int(ans_text.strip().lower() == "unknown"),
            # vector packs
            "h_last_256": h_last_256 if args.use_hidden else None,
            "h_pool_256": h_pool_256 if args.use_hidden else None,
            "h_last_mid_256": h_last_mid_256 if args.use_hidden else None,
            "h_pool_mid_256": h_pool_mid_256 if args.use_hidden else None,
        }

        # probe prediction
        x = build_feature_row(row, feat_names)
        p_right = float(probe.predict_proba([x])[0, 1])

        # print
        print("\n=== Prompt (few-shot as in training) ===")
        print(prompt)
        print("\n=== Model output (raw) ===")
        print(raw_text)
        print("\n=== Canonical JSON used for rescoring ===")
        print(canon_json)
        print("\n=== Parsed self-report ===")
        print(f'p_true: {p_model if p_model is not None else "None"}')
        print("\n=== Probe ===")
        print(f"P(correct): {p_right:.3f}")
        print("\n=== Decode stats (generation-time, training-aligned) ===")
        for k in ["answer_len","lp_mean","seq_conf","entropy_mean","entropy_last","entropy_std","margin_mean","margin_last","margin_min","rescore_logp"]:
            v = row[k] if k in row else None
            if isinstance(v, float):
                print(f"{k}={v:.3f}")
            else:
                print(f"{k}={v}")
        print(f"\n[info] gen_len={gen_len}  gen_time={dt:.3f}s")

if __name__ == "__main__":
    main()
