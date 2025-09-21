# scripts/run_baseline.py
import argparse
import time
import torch
import os
import json
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from src.conf.config import Cfg
from src.models.gpt_oss import GPTOSS
from src.data.datasets import load_qa
from src.eval.metrics import evaluate_batch, squad_em, squad_f1

ZERO_SHOT_FMT = (
    "Answer with only the short factual span (1–5 words). No punctuation.\n"
  "Q: {q}\nA: "
)

FEWSHOT_FIXED = """You are answering trivia questions. Give only the short answer.

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


def build_prompt(prompt_style: str, q: str, fewshot_file: str = None) -> str:
    if prompt_style == "zero_shot":
        return ZERO_SHOT_FMT.format(q=q)
    elif prompt_style == "fewshot_fixed":
        return FEWSHOT_FIXED.format(EVAL_QUESTION=q)
    elif prompt_style == "fewshot_file":
        assert fewshot_file and os.path.exists(fewshot_file), \
            f"--fewshot_file missing or not found: {fewshot_file}"
        with open(fewshot_file, "r", encoding="utf-8") as f:
            template = f.read()
        assert "{EVAL_QUESTION}" in template, \
            "Template must contain {EVAL_QUESTION} placeholder."
        return template.format(EVAL_QUESTION=q)
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")


def main(cfg: Cfg, args):
    hf_logging.set_verbosity_info()
    hf_logging.enable_propagation()

    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")

    # Resolve model knobs: config with CLI overrides
    max_new_tokens = args.max_new_tokens or cfg.model.max_new_tokens
    force_final = not args.allow_thinking
    reasoning = args.reasoning

    model = GPTOSS(
        cfg.model.model_id,
        cfg.model.dtype,
        cfg.model.device_map,
        max_new_tokens=max_new_tokens,
        use_router_probs=getattr(cfg.model, "use_router_probs", True),
        cache_dir=getattr(cfg.model, "cache_dir", None),
        use_chat_template=True,
        reasoning_effort=reasoning,
        force_final_prefix=force_final,
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
        prompt = build_prompt(args.prompt_style, q, args.fewshot_file)

        t0 = time.perf_counter()
        inp, gen = model.generate_with_states(prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        start = inp["input_ids"].shape[-1]
        new_ids = gen.sequences[:, start:]
        analysis_txt, final_txt = (
            "", None) if model.force_final_prefix else model.split_channels(new_ids[0])
        ans = (final_txt or model.decode(new_ids[0])).strip()

        gen_len = int(new_ids.shape[-1])
        total_tokens += gen_len
        total_time += dt
        tokps = (gen_len / dt) if dt > 0 else float("inf")

        em_i = 1 if squad_em(ans, gold) else 0
        f1_i = squad_f1(ans, gold)
        preds.append(ans)
        refs.append(gold)

        if idx == 0:
            raw = model.tok.decode(new_ids[0], skip_special_tokens=False)
            print("[debug] head:", raw[:200].replace("\n", "\\n"))
            print("[debug] tail:", raw[-200:].replace("\n", "\\n"))

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

    print(evaluate_batch(preds, refs))

    if args.save_jsonl:
        os.makedirs(os.path.dirname(args.save_jsonl) or ".", exist_ok=True)
        with open(args.save_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[baseline] wrote per-example outputs to {args.save_jsonl}")

    if total_time > 0 and len(preds) > 0:
        print(f"[baseline] avg toks/ex: {total_tokens/len(preds):.1f} | "
              f"avg time/ex: {total_time/len(preds):.2f}s | "
              f"overall toks/s: {total_tokens/total_time:.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="override dataset size")
    ap.add_argument("--save_jsonl", type=str, default=None,
                    help="path for per-example outputs")
    ap.add_argument("--verbose", action="store_true", default=True)

    # prompt controls
    ap.add_argument(
        "--prompt_style", choices=["zero_shot", "fewshot_fixed", "fewshot_file"], default="zero_shot")
    ap.add_argument("--fewshot_file", type=str, default=None)

    # thinking vs final-only
    ap.add_argument("--allow_thinking", action="store_true",
                    help="let model emit analysis → final")
    ap.add_argument(
        "--reasoning", choices=["low", "medium", "high"], default="low")
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="override cfg.model.max_new_tokens")
    ap.add_argument("--final_allowance", type=int, default=32,
                    help="max tokens allowed inside final (think mode)")
    ap.add_argument("--analysis_cap", type=int, default=256,
                    help="cap tokens before final (think mode)")

    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    main(cfg, args)
