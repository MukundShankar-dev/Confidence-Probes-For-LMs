import argparse, numpy as np, torch, time
from joblib import load
from src.conf.config import Cfg
from src.models.gpt_oss import GPTOSS
from src.data.datasets import load_qa
from src.eval.metrics import evaluate_batch
from src.probes.features import last_token_features, logits_entropy

PROMPT_FMT = (
  "Answer with the short factual answer only. "
  "Do not include any extra words or punctuation.\n"
  "Q: {q}\nA:"
)

def gen_one(model, q):
    inp, gen = model.generate_with_states(PROMPT_FMT.format(q=q))
    start = inp["input_ids"].shape[-1]
    new_ids = gen.sequences[:, start:]
    ans = model.decode(new_ids[0])
    final_step = gen.hidden_states[-1]         # tuple of layers
    feats = last_token_features(final_step, [16,24,32], "concat")
    if hasattr(gen, "scores") and gen.scores:
        ent = logits_entropy(torch.stack(gen.scores, dim=1)[:,-1:,:])
    else:
        ent = torch.zeros(feats.shape[0],1, device=feats.device)
    X = np.concatenate([feats.cpu().numpy(), ent.cpu().numpy()], axis=-1)
    return ans.strip(), X

def gen_n_best(model, q, k=3, temp=0.8, top_p=0.9):
    # replicate input and sample k candidates
    inputs = model._apply_chat_template(PROMPT_FMT.format(q=q))
    bsz = inputs["input_ids"].shape[0]
    for k_ in inputs: inputs[k_] = inputs[k_].repeat_interleave(k, dim=0)
    gen = model.model.generate(
        **inputs, max_new_tokens=model.max_new_tokens, do_sample=True,
        temperature=temp, top_p=top_p, num_return_sequences=k,
        return_dict_in_generate=True, output_scores=True, output_hidden_states=True
    )
    start = inputs["input_ids"].shape[-1]
    seqs = gen.sequences[:, start:]
    answers = [model.decode(seqs[i]).strip() for i in range(seqs.shape[0])]
    # features per candidate (use final step tuple)
    final_step = gen.hidden_states[-1]  # tuple per layer, shape [k, T, H] at last step
    feats = last_token_features(final_step, [16,24,32], "concat")
    if hasattr(gen, "scores") and gen.scores:
        ent = logits_entropy(torch.stack(gen.scores, dim=1)[:,-1:,:])
    else:
        ent = torch.zeros(feats.shape[0],1, device=feats.device)
    X = np.concatenate([feats.cpu().numpy(), ent.cpu().numpy()], axis=-1)
    return answers, X

def main(cfg: Cfg, mode: str, threshold: float, n_best: int):
    model = GPTOSS(cfg.model.model_id, cfg.model.dtype, cfg.model.device_map, cfg.model.max_new_tokens)
    ds = load_qa(cfg.data.dataset, cfg.data.split, cfg.data.limit)

    from src.data.datasets import squad_em, squad_f1
    probe = load("probe.joblib")
    try: calib = load("calib.joblib")
    except: calib = None

    preds, refs = [], []
    abstained = 0
    oracle_hit = 0   # for n-best: did any suggestion match?

    for ex in ds:
        q, gold = ex["question"], ex["answers"]
        ans, X = gen_one(model, q)
        p = probe.predict_proba(X.reshape(1,-1))[0]
        if calib is not None:
            p = float(calib.predict([p])[0])

        if p >= threshold:
            preds.append(ans)
        else:
            if mode == "abstain":
                preds.append("I am not sure about that.")
                abstained += 1
            else:
                cand_ans, Xk = gen_n_best(model, q, k=n_best)
                pk = probe.predict_proba(Xk)
                if calib is not None:
                    pk = calib.predict(pk)
                # pick highest-prob candidate for scoring, but also record oracle@k
                best_idx = int(np.argmax(pk))
                preds.append(cand_ans[best_idx])
                if any(squad_em(a, gold) for a in cand_ans):
                    oracle_hit += 1
        refs.append(gold)

    m = evaluate_batch(preds, refs)
    cov = 1 - (abstained / len(ds))
    print({"EM": m["EM"], "F1": m["F1"], "coverage": cov, "abstained": abstained,
           **({"oracle_at_k": oracle_hit/len(ds)} if mode!="abstain" else {})})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", choices=["abstain","n_best"], default="abstain")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n_best", type=int, default=3)
    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    main(cfg, args.mode, args.threshold, args.n_best)
