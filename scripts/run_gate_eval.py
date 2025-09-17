import numpy as np
import argparse
import torch
import numpy as np
from joblib import load
from src.conf.config import Cfg
from src.models.gpt_oss import GPTOSS
from src.data.datasets import load_qa
from src.data.datasets import exact_match
from src.probes.features import last_token_features, logits_entropy
from src.rag.retriever import Retriever
from src.eval.metrics import evaluate_batch


PROMPT_FMT = """Answer concisely.\nQ: {q}\nA:"""
REANSWER_FMT = """You are given evidence passages. If they contain the answer, answer using them and cite the passage title. Otherwise say: 'I am not sure.'\n\nQuestion: {q}\nEvidence:\n{ctx}\n\nAnswer:"""


def main(cfg: Cfg):
    model = GPTOSS(cfg.model.model_id, cfg.model.dtype,
                   cfg.model.device_map, cfg.model.max_new_tokens)
    retriever = Retriever(cfg.rag.backend, cfg.rag.top_k,
                          cfg.rag.max_passage_len)
    probe = load("probe.joblib")

    ds = load_qa(cfg.data.dataset, cfg.data.split, cfg.data.limit)
    preds, refs = [], []
    for ex in ds:
        q, gold = ex["question"], ex["answers"]
        inp, gen = model.generate_with_states(PROMPT_FMT.format(q=q))
        new_ids = gen.sequences[:, inp.shape[-1]:]
        ans_text = model.decode(new_ids[0])

        # features for decision
        final_step_tuple = gen.hidden_states[-1]
        feat_vec = last_token_features(
            final_step_tuple, cfg.probe.layers, cfg.probe.reducer)
        ent = logits_entropy(gen.scores[-1].unsqueeze(0)) if hasattr(
            gen, "scores") and gen.scores else torch.zeros(feat_vec.shape[0], 1)
        X = np.concatenate(
            [feat_vec.cpu().numpy(), ent.cpu().numpy()], axis=-1)
        score = probe.predict_proba(X.reshape(1, -1))[0]

        if score >= cfg.threshold:
            preds.append(ans_text)
        else:
            # retrieve + re-answer or refuse
            ctxs = retriever.fetch(q)
            ctx_block = "\n\n".join(ctxs)
            _, gen2 = model.generate_with_states(
                REANSWER_FMT.format(q=q, ctx=ctx_block))
            ans2 = model.decode(
                gen2.sequences[:, -gen2.num_generated_tokens:][0])
            preds.append(ans2)
        refs.append(gold)

    print(evaluate_batch(preds, refs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    main(cfg)
