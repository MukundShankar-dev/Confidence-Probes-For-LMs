from sklearn.metrics import roc_auc_score
import argparse
import torch
import numpy as np
from src.conf.config import Cfg
from src.models.gpt_oss import GPTOSS
from src.data.datasets import load_qa, squad_em
from src.probes.features import last_token_features, logits_entropy
from src.probes.train import Probe
from sklearn.model_selection import train_test_split
from src.probes.train import Probe


PROMPT_FMT = """Answer concisely.\nQ: {q}\nA:"""


def main(cfg: Cfg):
    model = GPTOSS(cfg.model.model_id, cfg.model.dtype,
                   cfg.model.device_map, cfg.model.max_new_tokens)
    ds = load_qa(cfg.data.dataset, cfg.data.split, cfg.data.limit)
    feats, labels = [], []
    for i, ex in enumerate(ds):
        if i >= cfg.probe.train_size:
            break
        q, gold = ex["question"], ex["answers"]
        inp, out = model.prefill(PROMPT_FMT.format(q=q))
        # generate once to get answer and logits entropy
        inp, gen = model.generate_with_states(PROMPT_FMT.format(q=q))
        start = inp["input_ids"].shape[-1]
        ans_text = model.decode(gen.sequences[:, start:][0])
        # features: last-token states from chosen layers + entropy
        # list per step; use last step tuple
        hs_last_step = [st[:, -1, :] for st in gen.hidden_states[-1]]
        # Hugging Face returns a list per step; grab final step tuple
        final_step_tuple = gen.hidden_states[-1]
        feat_vec = last_token_features(
            final_step_tuple, cfg.probe.layers, cfg.probe.reducer)
        ent = logits_entropy(gen.scores[-1].unsqueeze(0)) if hasattr(
            gen, "scores") and gen.scores else torch.zeros(feat_vec.shape[0], 1)
        fv = torch.cat([feat_vec, ent], dim=-1).cpu().numpy()
        feats.append(fv[0])
        labels.append(1 if squad_em(ans_text, gold) else 0)
    X, y = np.array(feats), np.array(labels)
    probe = Probe(cfg.probe.model)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    probe.fit(Xtr, ytr)

    p_val_raw = probe.predict_score(Xval)
    probe.fit_calibrator(p_val_raw, yval)

    probe.save("probe.joblib", "calib.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    main(cfg)
