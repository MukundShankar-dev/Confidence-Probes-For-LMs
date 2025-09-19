import argparse, time, torch
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from src.conf.config import Cfg
from src.models.gpt_oss import GPTOSS
from src.data.datasets import load_qa
from src.eval.metrics import evaluate_batch

PROMPT_FMT = (
  "Answer with the short factual answer only. "
  "Do not include any extra words or punctuation.\n"
  "Q: {q}\nA:"
)

def main(cfg: Cfg, limit_override: int = None, save_jsonl: str = None, verbose: bool = True):
    hf_logging.set_verbosity_info()
    hf_logging.enable_propagation()

    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")

    model = GPTOSS(cfg.model.model_id, cfg.model.dtype, cfg.model.device_map, cfg.model.max_new_tokens)
    ds = load_qa(cfg.data.dataset, cfg.data.split, limit_override or cfg.data.limit)

    preds, refs = [], []
    rows = []
    total_tokens = 0
    total_time = 0.0

    bar = tqdm(ds, desc="baseline", dynamic_ncols=True)
    for idx, ex in enumerate(bar):
        q = ex["question"]
        gold = ex["answers"]

        t0 = time.perf_counter()
        inp, gen = model.generate_with_states(PROMPT_FMT.format(q=q))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = time.perf_counter() - t0

        start = inp["input_ids"].shape[-1]
        new_ids = gen.sequences[:, start:]
        ans = model.decode(new_ids[0])

        gen_len = int(new_ids.shape[-1])
        total_tokens += gen_len
        total_time += dt
        tokps = (gen_len / dt) if dt > 0 else float("inf")

        preds.append(ans)
        refs.append(gold)

        if verbose and (idx % 10 == 0):
            # quick device/mem pulse
            try:
                mem = torch.cuda.max_memory_allocated() / 1e9
                bar.set_postfix(tokens=gen_len, t=f"{dt:.2f}s", tps=f"{tokps:.1f}", vram=f"{mem:.1f}GB")
            except Exception:
                bar.set_postfix(tokens=gen_len, t=f"{dt:.2f}s", tps=f"{tokps:.1f}")
        
        rows.append({
            "idx": idx,
            "question": q,
            "prediction": ans.strip(),
            "references": gold,
            "gen_tokens": int(new_ids.shape[-1]),
            "gen_time": round(dt, 3),
        })

    print(evaluate_batch(preds, refs))

    if save_jsonl:
        import json, os
        os.makedirs(os.path.dirname(cfg.save_jsonl) or ".", exist_ok=True)
        with open(save_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[baseline] wrote per-example outputs to {save_jsonl}")

    if total_time > 0:
        print(f"[baseline] avg toks/ex: {total_tokens/len(preds):.1f} | avg time/ex: {total_time/len(preds):.2f}s | overall toks/s: {total_tokens/total_time:.1f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="override dataset size for quick debug")
    ap.add_argument("--save_jsonl", type=str, default=None, help="path to write per-example outputs")
    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    main(cfg, limit_override=args.limit, save_jsonl=args.save_jsonl)
