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
from src.data.datasets import load_qa
from src.eval.metrics import evaluate_batch, squad_em, squad_f1

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

# Few-shot template for THINKING mode (textual delimiters)
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
        # Use textual Analysis/Final protocol (robust, model-agnostic)
        if style == "fewshot_fixed":
            return FEWSHOT_FIXED_THINKING.format(EVAL_QUESTION=q)
        elif style == "fewshot_file":
            assert fewshot_file and os.path.exists(fewshot_file), \
                f"--fewshot_file missing or not found: {fewshot_file}"
            with open(fewshot_file, "r", encoding="utf-8") as f:
                template = f.read()
            # Template must end its example with a 'Final:' block and contain {EVAL_QUESTION}
            assert "{EVAL_QUESTION}" in template, \
                "fewshot_file must contain {EVAL_QUESTION} placeholder."
            return template.format(EVAL_QUESTION=q)
        else:
            # zero_shot thinking: minimal scaffold
            return (
                "You are answering trivia questions. Be concise.\n\n"
                f"Q: {q}\n"
                "Analysis:\n"
                "- Briefly reason about the answer.\n"
                "Final:\n"
            )
    else:
        # Final-only (no thinking in prompt)
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
        else:  # zero_shot
            return ZERO_SHOT_FINAL_ONLY.format(q=q)

# --------------- Parsing ----------------

_FINAL_SPLIT_RE = re.compile(r'(?i)\bFinal:\s*')

def parse_analysis_final(text: str):
    """
    Extract (analysis, final) from text using the 'Analysis:' / 'Final:' protocol.
    Robust to casing and extra space. If no 'Final:' appears, fallback:
      - take the last non-empty line as final,
      - analysis is the rest.
    """
    parts = _FINAL_SPLIT_RE.split(text)
    if len(parts) >= 2:
        # everything before the last 'Final:' we treat as analysis text
        analysis_text = "Final:".join(parts[:-1]).strip()  # rejoin any intermediate splits literally
        final_block = parts[-1].strip()
        # take the first line after Final:
        final_line = final_block.splitlines()[0].strip()
        return analysis_text, final_line

    # Fallback: try last non-empty line as "final"
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return "\n".join(lines[:-1]).strip(), lines[-1]
    return "", text.strip()

# --------------- Main ----------------

def main(cfg: Cfg, args):
    hf_logging.set_verbosity_info()
    hf_logging.enable_propagation()

    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")

    max_new_tokens = args.max_new_tokens or cfg.model.max_new_tokens
    force_final = not args.allow_thinking  # final-only if we DON'T allow thinking
    reasoning = args.reasoning

    if args.thinking_mode == "off":
        use_chat = True
        force_final
        parse_harmony = False
        parse_textual = False
    elif args.thinking_mode == "textual":
        use_chat = False
        force_final = False
        parse_harmony = False
        parse_textual = True
    else:
        use_chat = True
        force_final = False
        parse_harmony = True
        parse_textual = False

    model = GPTOSS(
        cfg.model.model_id,
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

    limit = args.limit or cfg.data.limit
    ds = load_qa(cfg.data.dataset, cfg.data.split, limit)

    preds, refs, rows = [], [], []
    total_tokens, total_time = 0, 0.0

    bar = tqdm(ds, desc="baseline", dynamic_ncols=True)
    for idx, ex in enumerate(bar):
        q, gold = ex["question"], ex["answers"]
        prompt = build_prompt(q, args.prompt_style, args.allow_thinking, args.fewshot_file)

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
            analysis_txt, ans = "", model.tok.decode(
                new_ids[0], skip_special_tokens=True).strip()

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
    ap.add_argument("--thinking_mode", choices=["off", "textual", "harmony"], default="off", help="off=final-only; textual=Analysis/Final delimiters (no Harmony); harmony=think->final with Harmony"
                    )

    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    main(cfg, args)
