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
from src.data.datasets import load_qa
from src.eval.metrics import evaluate_batch, squad_em, squad_f1

# Allow fast math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---------------- Prompts ----------------

ZERO_SHOT_FINAL_ONLY = (
    "Answer with only the short factual span (1–5 words). No punctuation.\n"
    "Q: {q}\nA: "
)

FEWSHOT_FIXED_FINAL_ONLY = """You are answering trivia questions. Give only the short answer.

Q: Who wrote Hamlet?
A: William Shakespeare
Q: What is the capital of France?
A: Paris
Q: In which year did the Titanic sink?
A: 1912
Q: The chemical symbol for gold is?
A: Au
Q: Which planet is known as the Red Planet?
A: Mars
Q: {EVAL_QUESTION}
A: """

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

def build_prompt(q: str, style: str, allow_thinking: bool, fewshot_file: str = None) -> str:
    """Build the evaluation prompt based on style + whether we allow thinking."""
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
            return FEWSHOT_FIXED_FINAL_ONLY.format(EVAL_QUESTION=q)
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
        # Force FINAL-ONLY + FEWSHOT_FIXED_FINAL_ONLY for plain LMs
        prompt_style_effective = "fewshot_fixed"
        thinking_mode_effective = "off"
        allow_thinking_effective = False

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
        model_id = args.model_id or "google/gemma-12b-it"
        model = Gemma12B(
            model_id=model_id,
            dtype=getattr(cfg.model, "dtype", "float16"),
            device_map=None,                         # full GPU (no offload)
            max_new_tokens=max_new_tokens,
            cache_dir=getattr(cfg.model, "cache_dir", None),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    limit = args.limit or cfg.data.limit
    ds = load_qa(cfg.data.dataset, cfg.data.split, limit)

    if args.num_shards > 1:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_id, contiguous=True)
        print(f"[shard] Using shard {args.shard_id}/{args.num_shards} with {len(ds)} examples")

    preds, refs, rows = [], [], []
    total_tokens, total_time = 0, 0.0

    bar = tqdm(ds, desc="baseline", dynamic_ncols=True)
    for idx, ex in enumerate(bar):
        q, gold = ex["question"], ex["answers"]
        prompt = build_prompt(q, prompt_style_effective, allow_thinking_effective, args.fewshot_file)

        t0 = time.perf_counter()
        inp, gen = model.generate_with_states(prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        start = inp["input_ids"].shape[-1]
        new_ids = gen.sequences[:, start:]

        if parse_harmony:
            analysis_txt, ans = model.split_channels(new_ids[0])
            ans = ans.strip()
        elif parse_textual:
            raw_plain = model.tok.decode(new_ids[0], skip_special_tokens=True)
            analysis_txt, ans = parse_analysis_final(raw_plain)
            ans = ans.strip()
        else:
            analysis_txt, ans = "", model.tok.decode(new_ids[0], skip_special_tokens=True).strip()

        gen_len = int(new_ids.shape[-1])
        total_tokens += gen_len
        total_time += dt
        tokps = (gen_len / dt) if dt > 0 else float("inf")

        em_i = 1 if squad_em(ans, gold) else 0
        f1_i = squad_f1(ans, gold)
        preds.append(ans)
        refs.append(gold)

        if idx == 0 or args.debug_first:
            print("[debug] head:", ans[:200].replace("\n", "\\n"))
            print("[debug] tail:", ans[-200:].replace("\n", "\\n"))

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
            "references": gold,
            "gen_tokens": gen_len,
            "gen_time": round(dt, 3),
            "analysis": analysis_txt,
        })

    overall_metrics = evaluate_batch(preds, refs)

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
    ap.add_argument("--backend", choices=["gpt", "qwen", "gemma"], default="gpt",
                    help="Select model backend.")
    ap.add_argument("--model_id", type=str, default=None, help="override model id for selected backend")

    ap.add_argument("--num_shards", type=int, default=1, help="for distributed eval (slurm)")
    ap.add_argument("--shard_id", type=int, default=0, help="shard index for distributed eval (slurm)")

    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    main(cfg, args)
