# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from datasets import load_dataset
# from sklearn.metrics import brier_score_loss

# from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# # ---------------- Qwen Wrapper ----------------
# def _enc(tok, s: str):
#     try:
#         ids = tok.encode(s, add_special_tokens=False)
#         return ids if isinstance(ids, list) else []
#     except Exception:
#         return []

# class StopOnNewline(StoppingCriteria):
#     def __init__(self, tok):
#         self.nl_ids = _enc(tok, "\n")
#         self.start_len = None
#     def set_start_len(self, start_len: int): self.start_len = int(start_len)
#     def __call__(self, input_ids, scores, **kwargs):
#         if self.start_len is None: return False
#         seq = input_ids[0].tolist()
#         return self.nl_ids and seq[-len(self.nl_ids):] == self.nl_ids

# class QwenWrapper:
#     def __init__(self, model_name="Qwen/Qwen-7B", device="auto"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map=device,
#             trust_remote_code=True,
#             output_hidden_states=True,
#         )
#         self.stopper = StopOnNewline(self.tokenizer)

#     def infer(self, question: str, max_new_tokens=64, temperature=0.0):
#         prompt = f"Question: {question}\nAnswer:"
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#         self.stopper.set_start_len(inputs.input_ids.shape[1])

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#                 do_sample=temperature > 0.0,
#                 stopping_criteria=StoppingCriteriaList([self.stopper]),
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 use_cache=False,
#             )

#         gen_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
#         answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

#         final_logits = outputs.scores[-1]
#         probs = torch.nn.functional.softmax(final_logits, dim=-1)
#         entropy = -(probs * probs.log()).sum().item()

#         return {"answer": answer, "entropy": entropy}

# # ---------------- Metrics ----------------
# def normalize_text(s):
#     import re, string
#     s = s.lower().strip()
#     return re.sub(f"[{re.escape(string.punctuation)}]", "", s)

# def exact_match(pred, golds):
#     return int(normalize_text(pred) in [normalize_text(g) for g in golds])

# def f1_score(pred, golds):
#     pred_tokens = normalize_text(pred).split()
#     best = 0
#     for g in golds:
#         gold_tokens = normalize_text(g).split()
#         common = set(pred_tokens) & set(gold_tokens)
#         if not common: continue
#         prec = len(common) / len(pred_tokens)
#         rec  = len(common) / len(gold_tokens)
#         best = max(best, 2*prec*rec/(prec+rec))
#     return best

# def expected_calibration_error(confidences, correctness, bins=10):
#     ece, boundaries = 0.0, np.linspace(0,1,bins+1)
#     for i in range(bins):
#         mask = (confidences>=boundaries[i]) & (confidences<boundaries[i+1])
#         if mask.any():
#             acc, conf = correctness[mask].mean(), confidences[mask].mean()
#             ece += mask.mean() * abs(acc-conf)
#     return ece

# # ---------------- Baseline Runner ----------------
# def run_baseline(dataset_name, config=None, split="validation", n_samples=100):
#     qwen = QwenWrapper()
#     if config:
#         ds = load_dataset(dataset_name, config)[split].select(range(n_samples))
#     else:
#         ds = load_dataset(dataset_name)[split].select(range(n_samples))

#     records = []
#     preds, refs, confs, correct = [], [], [], []

#     for ex in ds:
#         if dataset_name == "squad":
#             q = ex["question"]
#             golds = [a["text"] for a in ex["answers"] if isinstance(a, dict) and "text" in a]
#         elif dataset_name == "nq_open":
#             q, golds = ex["question"], ex["answer"]
#         elif dataset_name == "trivia_qa":
#             q, golds = ex["question"], [ex["answer"]["value"]]
#         elif dataset_name == "hotpot_qa":
#             q, golds = ex["question"], ex["answer"]
#         else:
#             continue

#         out = qwen.infer(q)
#         pred, ent = out["answer"], out["entropy"]

#         conf = float(np.exp(-ent))  # entropy → confidence
#         is_correct = exact_match(pred, golds)

#         records.append({
#             "question": q,
#             "gold": golds,
#             "prediction": pred,
#             "entropy": ent,
#             "confidence": conf,
#             "correct": is_correct,
#         })
#         preds.append(pred); refs.append(golds)
#         confs.append(conf); correct.append(is_correct)

#     preds = np.array(preds)
#     refs = np.array(refs, dtype=object)
#     confs, correct = np.array(confs), np.array(correct)

#     # Metrics
#     em = np.mean([exact_match(p,g) for p,g in zip(preds,refs)])
#     f1 = np.mean([f1_score(p,g) for p,g in zip(preds,refs)])
#     ece = expected_calibration_error(confs, correct)
#     brier = brier_score_loss(correct, confs)

#     # Save CSV
#     df = pd.DataFrame(records)
#     df.to_csv(f"{dataset_name}_baseline.csv", index=False)

#     print(f"{dataset_name.upper()} ({n_samples} samples)")
#     print(f"EM={em:.3f}, F1={f1:.3f}, ECE={ece:.3f}, Brier={brier:.3f}")

#     # Graphs
#     thresholds = np.linspace(0,1,20)
#     coverage, acc = [], []
#     for t in thresholds:
#         mask = confs >= t
#         if mask.sum()==0: continue
#         coverage.append(mask.mean())
#         acc.append(correct[mask].mean())

#     plt.figure(figsize=(10,4))
#     plt.subplot(1,2,1)
#     plt.plot(coverage, acc, marker="o")
#     plt.xlabel("Coverage")
#     plt.ylabel("Accuracy")
#     plt.title(f"Selective QA - {dataset_name}")

#     plt.subplot(1,2,2)
#     bins = np.linspace(0,1,10)
#     bin_acc, bin_conf = [], []
#     for i in range(len(bins)-1):
#         mask = (confs>=bins[i]) & (confs<bins[i+1])
#         if mask.any():
#             bin_acc.append(correct[mask].mean())
#             bin_conf.append(confs[mask].mean())
#     plt.plot([0,1],[0,1], "k--")
#     plt.plot(bin_conf, bin_acc, marker="o")
#     plt.xlabel("Confidence")
#     plt.ylabel("Accuracy")
#     plt.title(f"Calibration - {dataset_name}")
#     plt.show()

# # ---------------- Run All ----------------
# if __name__ == "__main__":
#     run_baseline("trivia_qa", config="unfiltered", n_samples=50)
#     run_baseline("nq_open", n_samples=50)
#     run_baseline("squad", n_samples=50)
#     run_baseline("hotpot_qa", n_samples=50)

# # ------------------ Test ------------------
# if __name__ == "__main__":
#     qwen = QwenWrapper()
#     q = "Who wrote Crime and Punishment?"
#     result = qwen.infer(q)
#     print(f"Q: {q}")
#     print(f"A: {result['answer']} | entropy={result['entropy']:.4f}")


# src/models/qwen7b.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from typing import Dict, Tuple


def _enc(tok, s: str):
    """Encode a literal string to token IDs without specials (safe)."""
    try:
        ids = tok.encode(s, add_special_tokens=False)
        return ids if isinstance(ids, list) else []
    except Exception:
        return []


class _StopOnNewlineOrEos(StoppingCriteria):
    """
    Conservative stopper for plain-text final-only inference:
      - Require a couple tokens beyond the prompt (avoid empty outputs)
      - Stop on EOS, or when a trailing newline appears (common short-answer format)
    """

    def __init__(self, tok, min_new_tokens_after_start: int = 2):
        self.start_len = None
        self.min_new = int(min_new_tokens_after_start)
        self.nl_ids = _enc(tok, "\n")

    def set_start_len(self, start_len: int):
        self.start_len = int(start_len)

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_len is None:
            return False
        seq = input_ids[0].tolist()
        new = len(seq) - self.start_len
        if new < self.min_new:
            return False
        if self.nl_ids and len(seq) >= len(self.nl_ids) and seq[-len(self.nl_ids):] == self.nl_ids:
            return True
        # HF will also stop on eos_token_id if configured.
        return False


class Qwen7B:
    """
    GPTOSS-compatible adapter for Qwen (no Harmony channels).
    Exposes:
      - .tok  (tokenizer)
      - generate_with_states(prompt) -> (inputs_dict, gen_output)
      - split_channels(generated_ids) -> ("", final_text)
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-7B",
        dtype: str = "float16",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        cache_dir: str = None,
    ):
        torch_dtype = getattr(torch, dtype) if hasattr(
            torch, dtype) else torch.float16
        self.tok = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, cache_dir=cache_dir, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.eos_token_id

        self.max_new_tokens = int(max_new_tokens)
        self._stopper = _StopOnNewlineOrEos(
            self.tok, min_new_tokens_after_start=2)

        # Banners (mirroring GPTOSS style)
        dm = getattr(self.model, "hf_device_map", None)
        first_dev = next(self.model.parameters()).device
        print(f"[Qwen7B] device map: {dm}")
        print(f"[Qwen7B] first param device: {first_dev}")

    # ---------- inputs ----------
    def _build_inputs_plain(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return enc

    # ---------- public APIs (GPTOSS-compatible) ----------
    @torch.no_grad()
    def generate_with_states(self, prompt: str):
        """
        Final-only textual generation (no Harmony). Returns (inputs, gen) like GPTOSS.
        """
        inputs = self._build_inputs_plain(prompt)
        start_len = int(inputs["input_ids"].shape[-1])
        self._stopper.set_start_len(start_len)

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,                 # deterministic for baseline parity
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([self._stopper]),
            min_new_tokens=2,
        )
        return inputs, gen

    def split_channels(self, generated_ids: torch.Tensor) -> Tuple[str, str]:
        """
        Keep API parity with GPTOSS.split_channels. Qwen has no channels;
        just decode as the 'final' string.
        """
        text = self.tok.decode(generated_ids, skip_special_tokens=True).strip()
        return "", text
