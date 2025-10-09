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
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        dtype: str = "float16",
        device_map: str = None,            # force full GPU (no offload)
        max_new_tokens: int = 128,
        cache_dir: str = None,
    ):
        torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16
        self.tok = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, cache_dir=cache_dir, trust_remote_code=True
        )

        # Use SDPA (built-in PyTorch attention); avoid flash-attn (GLIBC issue).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,              # None -> we'll push to CUDA below
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda").eval()

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.eos_token_id

        self.max_new_tokens = int(max_new_tokens)
        self._stopper = _StopOnNewlineOrEos(self.tok, min_new_tokens_after_start=2)

        # Banners (mirroring GPTOSS style)
        dm = getattr(self.model, "hf_device_map", None)
        first_dev = next(self.model.parameters()).device
        print(f"[Qwen7B] device map: {dm}")
        print(f"[Qwen7B] first param device: {first_dev} | dtype={torch_dtype} | attn=sdpa")

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
            # keep generation cheap:
            # (drop no_repeat_ngram_size/repetition_penalty unless truly needed)
            return_dict_in_generate=True,
            output_scores=False,
            output_hidden_states=False,
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
