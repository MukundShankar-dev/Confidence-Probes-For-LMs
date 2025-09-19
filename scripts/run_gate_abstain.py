import argparse, numpy as np, torch
from joblib import load

from src.conf.config import Cfg
from src.models.gpt_oss import GPTOSS
from src.data.datasets import load_qa, squad_em
from src.eval.metrics import evaluate_batch
from src.probes.features import features_for_variant

PROMPT_FMT = (
  "Answer with the short factual answer only. "
  "Do not include any extra words or punctuation.\n"
  "Q: {q}\nA:"
)

def _int_list(csv: str):
    return [int(x) for x in csv.split(",") if x.strip() != ""]


@torch.no_grad()
def gen_one(model: GPTOSS, q: str, variant: str, base_layers, mice_layers, reducer: str,
            include_entropy: bool):
    inp, gen = model.generate_with_states(PROMPT_FMT.format(q=q))
    start = inp["input_ids"].shape[-1]
    new_ids = gen.sequences[:, start:]
    ans = model.decode(new_ids[0]).strip()

    final_step = gen.hidden_states[-1]  # tuple per layer
    W, b = model.get_unembedding()
    feat = features_for_variant(
        variant=variant,
        final_step_tuple=final_step,
        base_layers=base_layers,
        reducer=reducer,
        include_entropy=include_entropy,
        scores=gen.scores if hasattr(gen, "scores") else None,
        W=W, b=b,
        mice_layers=mice_layers if variant == "mice" else None,
    ).cpu().numpy()
    return ans, feat  # feat: [1, d]


@torch.no_grad()
def gen_n_best(model: GPTOSS, q: str, k: int, variant: str, base_layers, mice_layers, reducer: str,
               include_entropy: bool, temp=0.8, top_p=0.9):
    # replicate input and sample k candidates
    inputs = model._apply_chat_template(PROMPT_FMT.format(q=q))
    for k_ in inputs:
        inputs[k_] = inputs[k_].repeat_interleave(k, dim=0)
    gen = model.model.generate(
        **inputs, max_new_tokens=model.max_new_tokens, do_sample=True,
        temperature=temp, top_p=top_p, num_return_sequences=k,
        return_dict_in_generate=True, output_scores=True, output_hidden_states=True
    )
    start = inputs["input_ids"].shape[-1]
    seqs = gen.sequences[:, start:]
    answers = [model.decode(seqs[i]).strip() for i in range(seqs.shape[0])]

    # features per candidate (use final step tuple)
    final_step = gen.hidden_states[-1]
    W, b = model.get_unembedding()
    feat = features_for_variant(
        variant=variant,
        final_step_tuple=final_step,
        base_layers=base_layers,
        reducer=reducer,
        include_entropy=include_entropy,
        scores=gen.scores if hasattr(gen, "scores") else None,
        W=W, b=b,
        mice_layers=mice_layers if variant == "mice" else None,
    ).cpu().numpy()  # [k, d]
    return answers, feat


def main(cfg: Cfg, args: argparse.Namespace):
    model = GPTOSS(cfg.model.model_id, cfg.model.dtype,
                   cfg.model.device_map, cfg.model.max_new_tokens)
    ds = load_qa(cfg.data.dataset, cfg.data.split, cfg.data.limit)

    base_layers = _int_list(args.base_layers)
    mice_layers = _int_list(args.mice_layers)

    # Load the right probe type
    if args.probe_type == "cls":
        clf = load("probe.joblib")          # sklearn classifier (e.g., LogisticRegression/MLP)
        calib = None
        try:
            calib = load("calib.joblib")    # optional isotonic
        except Exception:
            pass
    else:
        clf = load("probe_sc.joblib")       # Ridge regressor for SC
        calib = None
        try:
            calib = load("calib_sc.joblib")
        except Exception:
            pass

    preds, refs = [], []
    abstained = 0
    oracle_hit = 0   # for n-best: did any suggestion match?

    for ex in ds:
        q, gold = ex["question"], ex["answers"]

        ans, X = gen_one(
            model, q,
            variant=args.variant, base_layers=base_layers, mice_layers=mice_layers,
            reducer=cfg.probe.reducer, include_entropy=args.include_entropy
        )  # X: [1, d]

        # raw score -> probability-like value
        if args.probe_type == "cls":
            p_raw = float(clf.predict_proba(X.reshape(1, -1))[:, 1][0]) \
                if hasattr(clf, "predict_proba") else float(clf.decision_function(X.reshape(1, -1)))
        else:
            p_raw = float(clf.predict(X.reshape(1, -1))[0])  # regression output in [0,1] (approx)

        p = float(calib.predict([p_raw])[0]) if calib is not None else p_raw

        if p >= args.threshold:
            preds.append(ans)
        else:
            if args.mode == "abstain":
                preds.append("I am not sure about that.")
                abstained += 1
            else:
                cand_ans, Xk = gen_n_best(
                    model, q, k=args.n_best,
                    variant=args.variant, base_layers=base_layers, mice_layers=mice_layers,
                    reducer=cfg.probe.reducer, include_entropy=args.include_entropy
                )
                if args.probe_type == "cls":
                    pk = clf.predict_proba(Xk)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xk)
                else:
                    pk = clf.predict(Xk)
                if calib is not None:
                    pk = calib.predict(pk)
                best_idx = int(np.argmax(pk))
                preds.append(cand_ans[best_idx])
                if any(squad_em(a, gold) for a in cand_ans):
                    oracle_hit += 1

        refs.append(gold)

    m = evaluate_batch(preds, refs)
    cov = 1 - (abstained / len(ds))
    out = {"EM": m["EM"], "F1": m["F1"], "coverage": cov, "abstained": abstained}
    if args.mode != "abstain":
        out["oracle_at_k"] = oracle_hit / len(ds)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")

    # Gate behavior
    ap.add_argument("--mode", choices=["abstain", "n_best"], default="abstain")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n_best", type=int, default=3)

    # Feature variant
    ap.add_argument("--variant", choices=["baseline", "mice"], default="baseline")
    ap.add_argument("--base_layers", type=str, default="16,24,32")
    ap.add_argument("--mice_layers", type=str, default="16,24,32")
    ap.add_argument("--include_entropy", action="store_true", default=True)

    # Which trained probe to use
    ap.add_argument("--probe_type", choices=["cls", "sc"], default="cls",
                    help="'cls' = 0/1 correctness probe; 'sc' = self-consistency regressor")

    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    main(cfg, args)
