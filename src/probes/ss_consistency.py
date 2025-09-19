# NEW: src/probes/ss_consistency.py
import torch, numpy as np
from collections import Counter
from typing import List, Tuple, Dict
from .features import features_for_variant
from ..data.datasets import _normalize  # your SQuAD normalizer

def normalize_span(s: str) -> str:
    return _normalize(s.strip())

@torch.no_grad()
def sample_k_candidates(model, prompt: str, k: int = 6, max_new_tokens: int = 24):
    inputs = model._apply_chat_template(prompt)
    # repeat inputs
    for key in inputs:
        inputs[key] = inputs[key].repeat_interleave(k, dim=0)
    gen = model.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.8, top_p=0.9,
        num_return_sequences=k,
        return_dict_in_generate=True,
        output_scores=True,
        output_hidden_states=True,
    )
    start = inputs["input_ids"].shape[-1]
    seqs = gen.sequences[:, start:]
    texts = [model.decode(seqs[i]).strip() for i in range(k)]
    return gen, texts

def soft_self_consistency(texts: List[str]) -> Dict[str, float]:
    # group by normalized span
    keys = [normalize_span(t) for t in texts]
    c = Counter(keys)
    total = sum(c.values())
    return {k: v / total for k, v in c.items()}  # probs sum to 1
