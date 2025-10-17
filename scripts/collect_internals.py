#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect internals for a confidence probe:
- Generates one answer per example (same flow as run_baseline)
- Captures token-level scores, margins, entropies
- Runs one teacher-forced forward pass to grab hidden states
- Rescores the canonical JSON output (better than rescoring only the span)
- Writes compact features + label (EM) to JSONL

USAGE (examples):
  python -m scripts.collect_internals \
    --backend llama31 \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset triviaqa --split validation \
    --max_new_tokens 64 \
    --num_shards $num_shards --shard_id $shard_id \
    --out data/probe/triviaqa/llama31_shard${shard_id}.jsonl
"""
import argparse, json, os, re, math, time, sys
import torch
from tqdm import tqdm
from transformers.utils import logging as hf_logging

# === your project modules ===
from src.data.datasets import load_qa, squad_em, squad_f1
from src.models.gpt_oss import GPTOSS
from src.models.qwen7b import Qwen7B
from src.models.gemma12b import Gemma12B
from src.models.llama31_8b import Llama31_8B
from src.models.llama32_11b import Llama32_11B

# ------- speed knobs -------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ------- prompts -------
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

FEWSHOT_FIXED_FINAL_ONLY_SQUADV2 = """You are answering extractive QA with possible unanswerable questions (SQuAD v2).
Return only a single JSON object with fields:
- "answer": the short factual span (1–5 words, no punctuation). If unanswerable, output "Unknown".
- "p_true": the probability (0.0–1.0) that this answer is correct (round to two decimals)

Calibration guidance:
- Predict "Unknown" when there is no sufficient answer in the context.
- Report your true probability; do NOT inflate.
- Overconfidence is penalized by proper scoring (Brier). If unsure, choose a lower value.

Q: Who wrote Hamlet?
{{"answer": "William Shakespeare", "p_true": 0.95}}

Q: In 2007, who was the prime minister of Canada? (unanswerable)
{{"answer": "Unknown", "p_true": 0.15}}

Q: {EVAL_QUESTION}
"""

_JSON_WARN_ONCE = False

def build_prompt(q: str, dataset: str) -> str:
    if dataset.lower() in ("squadv2", "squad_v2", "squad2"):
        return FEWSHOT_FIXED_FINAL_ONLY_SQUADV2.format(EVAL_QUESTION=q)
    else:
        return FEWSHOT_FIXED_FINAL_ONLY_DEFAULT.format(EVAL_QUESTION=q)

def safe_json_extract(text: str):
    """Return best-effort JSON object from model text."""
    global _JSON_WARN_ONCE
    t = text.strip()

    # strip code fences
    if t.startswith("```"):
        t = t.lstrip("`")
        if "\n" in t:
            t = t.split("\n", 1)[1].strip()
        if t.endswith("```"):
            t = t[:-3].strip()

    # un-escape sequences like \" and \\\" if someone double-escaped
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
    """
    Robustly parse {"answer": "...", "p_true": 0.xx} with lots of tolerance.
    Falls back to ("<raw>", None) on failure.
    """
    obj = safe_json_extract(text)
    if isinstance(obj, dict):
        norm = {str(k).strip().lower(): v for k, v in obj.items()}
        ans = (norm.get("answer") or norm.get("final") or norm.get("prediction") or "").strip()
        # accept p_true / confidence variants
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
    # fallback: when no JSON
    return text.strip(), None

def token_logprobs_for_targets(logits, target_ids):
    """
    Given step-wise logits (list of tensors [1, V]) and target token ids (list[int]),
    return per-step log-probs for those ids. Assumes alignment.
    """
    logps = []
    for logit_step, tok_id in zip(logits, target_ids):
        lsm = torch.log_softmax(logit_step[0].float(), dim=-1)
        logps.append(float(lsm[int(tok_id)].item()))
    return logps

def rescore_mean_logprob(model, tok, prompt_text: str, target_text: str):
    """
    Compute mean log-prob of target_text when appended to prompt_text.
    Uses a single forward (teacher-forced).
    """
    with torch.no_grad():
        enc_prompt = tok(prompt_text, add_special_tokens=False, return_tensors="pt")
        enc_target = tok(target_text, add_special_tokens=False, return_tensors="pt")
        input_ids = torch.cat([enc_prompt["input_ids"], enc_target["input_ids"]], dim=-1).to(model.device)
        attn = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
        logits = out.logits[:, :-1, :]           # shift for next-token pred
        target_slice = input_ids[:, enc_prompt["input_ids"].shape[-1]:]  # the target portion
        logits_tgt = logits[:, -target_slice.shape[-1]:, :]              # align last K steps to target
        step_logps = []
        for t in range(target_slice.shape[-1]):
            lsm = torch.log_softmax(logits_tgt[0, t].float(), dim=-1)
            step_logps.append(float(lsm[int(target_slice[0, t])].item()))
        return sum(step_logps) / max(1, len(step_logps))

def pack_256(vec: torch.Tensor):
    """L2-normalize then take first 256 dims."""
    if vec is None:
        return None
    v = torch.nn.functional.normalize(vec.float(), dim=-1)
    v = v[:256] if v.shape[-1] >= 256 else torch.nn.functional.pad(v, (0, 256 - v.shape[-1]))
    return [float(x) for x in v.cpu()]

def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--dataset", choices=["triviaqa", "hotpot_qa", "squad_v2"], default="triviaqa")
    ap.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)

    # NEW: supplementary controls (same as run_baseline)
    ap.add_argument("--supplementary_data", type=str, default=None,
                    help="Path to a TriviaQA-like JSONL to append/use for ANY split "
                         "(e.g., src/data/supplementary_data.jsonl). Pass empty string \"\" to force-disable auto-pickup.")
    ap.add_argument("--supplementary_only", action="store_true", default=False,
                    help="Use only the supplementary JSONL for the selected split (no base dataset)")

    # model
    ap.add_argument("--backend", choices=["gpt", "qwen", "gemma", "llama31", "llama32"], default="llama31")
    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    # io
    ap.add_argument("--out", type=str, required=True, help="Path to write probe dataset jsonl")
    ap.add_argument("--verbose", action="store_true", default=False)
    args = ap.parse_args()

    hf_logging.set_verbosity_info()
    hf_logging.enable_propagation()

    # ---- resolve supplementary path (auto-pickup, with explicit disable) ----
    supp = args.supplementary_data
    auto_pick = True
    if isinstance(supp, str) and supp.strip() == "":
        supp = None
        auto_pick = False
    if supp is None and auto_pick:
        default_path = "src/data/supplementary_data.jsonl"
        if os.path.exists(default_path):
            supp = default_path

    # ---- load dataset (supplement supported on ANY split) ----
    ds = load_qa(
        args.dataset,
        args.split,
        args.limit,
        supplementary_data=supp,
        supplementary_only=args.supplementary_only,
    )

    # Shard AFTER merge/selection so supplement is evenly distributed too
    if args.num_shards > 1:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_id, contiguous=True)
        print(f"[shard] Using shard {args.shard_id}/{args.num_shards} with {len(ds)} examples")

    # ---- model init ----
    if args.backend == "gpt":
        model = GPTOSS(
            model_id=args.model_id,
            dtype="float16",
            device_map=None,
            max_new_tokens=args.max_new_tokens,
            use_router_probs=True,
            cache_dir=None,
            use_chat_template=True,
            reasoning_effort="low",
            force_final_prefix=True,
            final_allowance=32,
            analysis_cap=512,
        )
    elif args.backend == "qwen":
        mid = args.model_id or "Qwen/Qwen2.5-7B-Instruct"
        model = Qwen7B(model_id=mid, dtype="float16", device_map=None, max_new_tokens=args.max_new_tokens, cache_dir=None)
    elif args.backend == "gemma":
        mid = args.model_id or "google/gemma-3-12b-it"
        model = Gemma12B(model_id=mid, dtype="float16", device_map=None, max_new_tokens=args.max_new_tokens, cache_dir=None)
    elif args.backend == "llama31":
        mid = args.model_id or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model = Llama31_8B(model_id=mid, dtype="float16", device_map=None, max_new_tokens=args.max_new_tokens, cache_dir=None)
    elif args.backend == "llama32":
        mid = args.model_id or "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = Llama32_11B(model_id=mid, dtype="float16", device_map=None, max_new_tokens=args.max_new_tokens, cache_dir=None)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # sanity: show GPU
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")

    tok = model.tok
    hf_causal_lm = model.model  # underlying HF CausalLM
    device = next(hf_causal_lm.parameters()).device

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    w = open(args.out, "w", encoding="utf-8")

    bar = tqdm(ds, desc="collect", dynamic_ncols=True)
    for idx, ex in enumerate(bar):
        q, gold = ex["question"], ex["answers"]
        prompt = build_prompt(q, args.dataset)

        # ---- generate (your wrapper returns (inp, gen)) ----
        t0 = time.perf_counter()
        inp, gen = model.generate_with_states(prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        start = inp["input_ids"].shape[-1]
        new_ids = gen.sequences[:, start:]
        raw_text = model.tok.decode(new_ids[0], skip_special_tokens=True).strip()

        # parse model JSON
        ans_text, p_model = extract_answer_and_prob(raw_text)
        if not ans_text:
            ans_text = raw_text

        # token-level stats from scores
        scores = gen.scores or []
        T = len(scores)
        gen_ids_seq = new_ids[0].tolist()
        # exclude final step for "content" stats
        content_len = max(1, T - 1) if T > 1 else T

        step_ent = []
        margins = []
        step_logp = []
        for t in range(T):
            logits = scores[t][0].float()
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
        seq_conf = math.exp(lp_mean) if lp_mean is not None else None  # geometric mean token prob

        # ---- teacher-forced forward to fetch hidden states ----
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

        # ---- rescore canonical JSON ----
        # Canonicalize compact JSON (no extra spaces; stable formatting)
        safe_ans = ans_text.replace('"', '\\"')
        pt = p_model if p_model is not None else 0.00
        canon_json = f'{{"answer":"{safe_ans}","p_true":{pt:.2f}}}'
        rescore_lp = rescore_mean_logprob(hf_causal_lm, tok, prompt, canon_json)

        # ---- labels ----
        em_i = 1 if squad_em(ans_text, gold) else 0
        f1_i = squad_f1(ans_text, gold)

        row = {
            "idx": ex.get("idx", idx),
            "question": q,
            "prediction": ans_text,
            "references": gold,
            "em": em_i,
            "f1": f1_i,

            # model-decode side
            "raw_output": raw_text,
            "gen_tokens": int(gen_len),
            "gen_time": round(dt, 3),

            # parsed prob from model (may be None)
            "model_confidence": p_model,

            # token stats
            "lp_mean": lp_mean,
            "seq_conf": seq_conf,
            "entropy_mean": entropy_mean,
            "entropy_last": entropy_last,
            "entropy_std": entropy_std,
            "margin_mean": margin_mean,
            "margin_last": margin_last,
            "margin_min": margin_min,

            # hidden-state packs
            "h_last_256": h_last_256,
            "h_pool_256": h_pool_256,
            "h_last_mid_256": h_last_mid_256,
            "h_pool_mid_256": h_pool_mid_256,

            # rescoring aligned with prompt format
            "rescore_logp": rescore_lp,

            # meta
            "answer_len": int(gen_len),
            "parsed_json_ok": int(ans_text is not None and len(ans_text) > 0),
            "parsed_p_true_ok": int(p_model is not None),
            "is_unknown": int(ans_text.strip().lower() == "unknown"),
        }

        w.write(json.dumps(row, ensure_ascii=False) + "\n")

    w.close()
    print(f"[collect] wrote {args.out}")

if __name__ == "__main__":
    main()
