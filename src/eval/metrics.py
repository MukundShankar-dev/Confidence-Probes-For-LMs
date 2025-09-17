from typing import Dict
from .datasets import exact_match


def evaluate_batch(preds, refs):
    correct = 0
    for p, r in zip(preds, refs):
        correct += 1 if exact_match(p, r) else 0
    return {"accuracy": correct / max(1, len(preds))}
