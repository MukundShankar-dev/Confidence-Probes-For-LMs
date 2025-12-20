import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from typing import Dict, Tuple, List

"""
Despite being named qwen7b.py, this module is actually compatible with all qwen models.
"""

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
      - Stop on EOS, or when a trailing newline appears
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
        return False

class _StopOnStrings(StoppingCriteria):
    """
    Stop if the output ends with any of a set of string patterns (encoded to IDs).
    Useful to prevent Qwen from starting another 'Q:' block.
    """
    def __init__(self, tok, stop_strings: List[str]):
        self.stop_ids: List[List[int]] = []
        for s in stop_strings:
            ids = _enc(tok, s)
            if ids:
                self.stop_ids.append(ids)
        self.start_len = None

    def set_start_len(self, start_len: int):
        self.start_len = int(start_len)

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_len is None:
            return False
        seq = input_ids[0].tolist()
        for pat in self.stop_ids:
            L = len(pat)
            if L and len(seq) >= L and seq[-L:] == pat:
                return True
        return False

class Qwen7B:
    """
    GPTOSS-compatible adapter for Qwen (no Harmony channels).
    Uses chat template to reduce pattern continuation and adds stop strings.
    Exposes:
      - .tok
      - generate_with_states(prompt) -> (inputs_dict, gen_output)
      - split_channels(generated_ids) -> ("", final_text)
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        dtype: str = "float16",
        device_map: str = None,
        max_new_tokens: int = 128,
        cache_dir: str = None,
        use_chat_template: bool = True,
        system_prompt: str = (
            "You are answering trivia questions. "
            "Return only a single JSON object with fields 'answer' and 'confidence'. "
            "Do not include anything else."
        ),
    ):
        torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16
        self.tok = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, cache_dir=cache_dir, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda").eval()

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.eos_token_id

        self.max_new_tokens = int(max_new_tokens)
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt

        # Stoppers
        self._stopper_nl = _StopOnNewlineOrEos(self.tok, min_new_tokens_after_start=2)
        # Stop if model starts another question or a new turn marker
        self._stopper_strs = _StopOnStrings(
            self.tok,
            stop_strings=["\nQ:", " Q:", "\nUser:", "\nAssistant:"]
        )

        # Banners
        dm = getattr(self.model, "hf_device_map", None)
        first_dev = next(self.model.parameters()).device
        print(f"[Qwen7B] device map: {dm}")
        print(f"[Qwen7B] first param device: {first_dev}")

    # ---------- inputs ----------
    def _build_inputs(self, user_text: str) -> Dict[str, torch.Tensor]:
        """
        Build inputs using chat template so Qwen treats this as a single assistant turn.
        We put the entire few-shot block (with the final 'Q: ...') as the user message.
        """
        if self.use_chat_template and hasattr(self.tok, "apply_chat_template"):
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_text})
            enc = self.tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            inputs = {"input_ids": enc.to(self.model.device)}
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            return inputs
        else:
            # Fallback: plain encode WITH special tokens to avoid degenerate output
            enc = self.tok(user_text, return_tensors="pt", add_special_tokens=True)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            if "attention_mask" not in enc:
                enc["attention_mask"] = torch.ones_like(enc["input_ids"])
            return enc

    @torch.no_grad()
    def generate_with_states(self, prompt: str):
        """
        Final-only textual generation. Returns (inputs, gen) like GPTOSS.
        The `prompt` string is your few-shot-final-only block from run_baseline.
        """
        inputs = self._build_inputs(prompt)
        start_len = int(inputs["input_ids"].shape[-1])
        self._stopper_nl.set_start_len(start_len)
        self._stopper_strs.set_start_len(start_len)

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=False,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([self._stopper_nl, self._stopper_strs]),
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
