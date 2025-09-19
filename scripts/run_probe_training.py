import argparse
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import Ridge
from joblib import dump

from src.conf.config import Cfg
from src.models.gpt_oss import GPTOSS
from src.data.datasets import load_qa, squad_em
from src.probes.features import features_for_variant
from src.probes.train import Probe  # classification-style probe with calibration helpers
from src.probes.ss_consistency import (
    sample_k_candidates, soft_self_consistency, normalize_span
)

# Closed-book, span-like prompt (helps EM/F1 labeling)
PROMPT_FMT = (
    "Answer with the short factual answer only. "
    "Do not include any extra words or punctuation.\n"
    "Q: {q}\nA:"
)


def _int_list(csv: str):
    return [int(x) for x in csv.split(",") if x.strip() != ""]


def main(cfg: Cfg, args: argparse.Namespace):
    # --- model ---
    model = GPTOSS(cfg.model.model_id, cfg.model.dtype,
                   cfg.model.device_map, cfg.model.max_new_tokens)
    W, b = model.get_unembedding()

    base_layers = _int_list(args.base_layers)
    mice_layers = _int_list(args.mice_layers)

    # --- data ---
    ds = load_qa(cfg.data.dataset, cfg.data.split, cfg.data.limit)

    if args.variant in ("baseline", "mice"):
        # -------- classification probe: 0/1 correctness labels ----------
        feats, labels = [], []

        for i, ex in enumerate(ds):
            if i >= cfg.probe.train_size:
                break

            q, gold = ex["question"], ex["answers"]

            # one closed-book generation w/ states & scores
            inp, gen = model.generate_with_states(PROMPT_FMT.format(q=q))
            start = inp["input_ids"].shape[-1]
            ans_text = model.decode(gen.sequences[:, start:][0]).strip()

            # last-step features (variant selects baseline vs mice)
            final_step_tuple = gen.hidden_states[-1]
            feat_vec = features_for_variant(
                variant=args.variant,
                final_step_tuple=final_step_tuple,
                base_layers=base_layers,
                reducer=cfg.probe.reducer,
                include_entropy=args.include_entropy,
                scores=gen.scores if hasattr(gen, "scores") else None,
                W=W, b=b,
                mice_layers=mice_layers if args.variant == "mice" else None,
            )
            feats.append(feat_vec.cpu().numpy()[0])
            labels.append(1 if squad_em(ans_text, gold) else 0)

        X = np.array(feats, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        # split once, fit, calibrate
        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        probe = Probe(cfg.probe.model)
        probe.fit(Xtr, ytr)

        # validation AUC + isotonic calibration
        try:
            p_val_raw = probe.predict_score(Xval)
            auc = roc_auc_score(yval, p_val_raw)
            print(f"[probe] variant={args.variant} | val AUC={auc:.3f} | pos_rate={yval.mean():.3f}")
            probe.fit_calibrator(p_val_raw, yval)
        except Exception as e:
            print(f"[probe] calibration skipped: {e}")

        probe.save("probe.joblib", "calib.joblib")
        # (Optional) persist training arrays if you want to inspect later:
        # np.save("X_train.npy", Xtr); np.save("y_train.npy", ytr)

    elif args.variant == "sc":
        # -------- soft self-consistency regression probe ----------
        # We regress to soft SC (share per normalized span) on a small K.
        K = args.sc_k
        feats_list, targets_list = [], []

        for i, ex in enumerate(ds):
            if i >= cfg.probe.train_size:
                break

            q = ex["question"]
            prompt = PROMPT_FMT.format(q=q)

            gen, texts = sample_k_candidates(
                model, prompt, k=K, max_new_tokens=cfg.model.max_new_tokens
            )
            final_step_tuple = gen.hidden_states[-1]  # batch size = K

            # choose which feature set to use for SC (baseline or mice)
            sc_feat_variant = "mice" if args.sc_variant == "mice" else "baseline"
            feat_batch = features_for_variant(
                variant=sc_feat_variant,
                final_step_tuple=final_step_tuple,
                base_layers=base_layers,
                reducer=cfg.probe.reducer,
                include_entropy=True,
                scores=gen.scores if hasattr(gen, "scores") else None,
                W=W, b=b,
                mice_layers=mice_layers if sc_feat_variant == "mice" else None,
            ).cpu().numpy()  # [K, d]

            # soft self-consistency targets, one per candidate
            sc_map = soft_self_consistency(texts)  # dict norm_span -> prob
            y_soft = np.array([sc_map.get(normalize_span(t), 0.0) for t in texts], dtype=np.float32)

            feats_list.append(feat_batch)
            targets_list.append(y_soft)

        X = np.concatenate(feats_list, axis=0)  # [N*K, d]
        y = np.concatenate(targets_list, axis=0)  # [N*K]

        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        reg = Ridge(alpha=1.0)
        reg.fit(Xtr, ytr)
        ypred = reg.predict(Xval)
        mse = mean_squared_error(yval, ypred)
        print(f"[probe-sc] variant={args.sc_variant} | val MSE={mse:.4f}")

        # Optional: isotonic on top of regressor outputs to tighten calibration
        # (Note: y in [0,1], good for isotonic.)
        try:
            from sklearn.isotonic import IsotonicRegression
            calib = IsotonicRegression(out_of_bounds="clip").fit(ypred, yval)
            dump(calib, "calib_sc.joblib")
            print("[probe-sc] wrote calib_sc.joblib")
        except Exception as e:
            print(f"[probe-sc] isotonic calibration skipped: {e}")

        dump(reg, "probe_sc.joblib")
        print("[probe-sc] wrote probe_sc.joblib")

    else:
        raise ValueError(f"Unknown variant: {args.variant}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")

    # Which probe are we training?
    ap.add_argument("--variant", choices=["baseline", "mice", "sc"], default="baseline")

    # Feature knobs shared by baseline/mice/sc
    ap.add_argument("--base_layers", type=str, default="16,24,32",
                    help="layers for baseline features (comma-separated indices)")
    ap.add_argument("--mice_layers", type=str, default="16,24,32",
                    help="layers for MICE features (comma-separated indices)")
    ap.add_argument("--include_entropy", action="store_true", default=True)

    # SC-specific knobs
    ap.add_argument("--sc_k", type=int, default=6,
                    help="# candidates per example for self-consistency labels (keep small)")
    ap.add_argument("--sc_variant", choices=["baseline", "mice"], default="baseline",
                    help="which features to use for SC regression")

    args = ap.parse_args()
    cfg = Cfg.load(args.config)
    main(cfg, args)
