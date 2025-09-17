import torch
import argparse
from src.conf.config import Cfg
from src.models.gpt_oss import GPTOSS
from src.data.datasets import load_qa, exact_match
from src.eval.metrics import evaluate_batch


PROMPT_FMT = """Answer concisely.\nQ: {q}\nA:"""


def main(cfg: Cfg):
    model = GPTOSS(cfg.model.model_id, cfg.model.dtype,
                   cfg.model.device_map, cfg.model.max_new_tokens)
    ds = load_qa(cfg.data.dataset, cfg.data.split, cfg.data.limit)
    preds, refs = [], []
    for ex in ds:
        q = ex["question"]
        gold = ex["answers"]
        _, gen = model.generate_with_states(PROMPT_FMT.format(q=q))
        new_ids = gen.sequences[:, -gen.num_generated_tokens:]
        ans = model.decode(new_ids[0])
        preds.append(ans)
        refs.append(gold)
    print(evaluate_batch(preds, refs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    main(cfg)
