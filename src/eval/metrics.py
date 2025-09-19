from typing import Dict
from src.data.datasets import squad_em, squad_f1


def evaluate_batch(preds, refs):
    n = max(1, len(preds))
    em = sum(squad_em(p, r) for p, r in zip(preds, refs)) / n
    f1 = sum(squad_f1(p, r) for p, r in zip(preds, refs)) / n
    return {"EM": em, "F1": f1}
