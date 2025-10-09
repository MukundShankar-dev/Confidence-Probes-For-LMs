# src/models/llama31_8b.py
import torch
from typing import Dict, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

def _enc(tok, s: str):
    try:
        ids = tok.encode(s, add_special_tokens=False)
        return ids if isinstance(ids, list) else []
    except Exception:
        return []

class _StopOnNewlineOrEos(StoppingCriteria):
    def __init__(self, tok, min_new_tokens_after_start: int = 2):
        self.start_len = None
        self.min_new = int(min_new_tokens_after_start)
        self.nl_ids = _enc(tok, "\n")

    def set_start_len(self, start_len: int): self.start_len = int(start_len)

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
    def __init__(self, tok, stop_strings: List[str]):
        self.stop_ids: List[List[int]] = []
        for s in stop_strings:
            ids = _enc(tok, s)
            if ids:
                self.stop_ids.append(ids)
        self.start_len = None

    def set_start_len(self, start_len: int): self.start_len = int(start_len)

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_len is None:
            return False
        seq = input_ids[0].tolist()
        for pat in self.stop_ids:
            L = len(pat)
            if L and len(seq) >= L and seq[-L:] == pat:
                return True
        return False

class Llama31_8B:
    """
    GPTOSS-compatible adapter for Meta-Llama-3.1-8B-Instruct (text-only).
    Exposes:
      - .tok
      - generate_with_states(prompt) -> (inputs, gen)
      - split_channels(generated_ids) -> ("", final_text)
    """
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        dtype: str = "float16",
        device_map: str = None,           # None -> push to CUDA below
        max_new_tokens: int = 128,
        cache_dir: str = None,
        use_chat_template: bool = True,
        system_prompt=(
            "You are answering trivia questions. "
            "Return only a single JSON object with keys exactly \"answer\" and \"confidence\". "
            "Do not include any other keys or text. The key must be spelled \"confidence\" (not \"conference\")."
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

        self._stopper_nl = _StopOnNewlineOrEos(self.tok, min_new_tokens_after_start=2)
        self._stopper_strs = _StopOnStrings(
            self.tok, stop_strings=["\nQ:", " Q:", "\nUser:", "\nAssistant:"]
        )

        dm = getattr(self.model, "hf_device_map", None)
        first_dev = next(self.model.parameters()).device
        print(f"[Llama31_8B] device map: {dm}")
        print(f"[Llama31_8B] first param device: {first_dev}")

    def _build_inputs(self, user_text: str) -> Dict[str, torch.Tensor]:
        if self.use_chat_template and hasattr(self.tok, "apply_chat_template"):
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_text})
            enc = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            inputs = {"input_ids": enc.to(self.model.device)}
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            return inputs
        else:
            enc = self.tok(user_text, return_tensors="pt", add_special_tokens=True)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            if "attention_mask" not in enc:
                enc["attention_mask"] = torch.ones_like(enc["input_ids"])
            return enc

    @torch.no_grad()
    def generate_with_states(self, prompt: str):
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
            output_scores=True,           # enable logits for entropy/seq_conf
            output_hidden_states=False,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([self._stopper_nl, self._stopper_strs]),
            min_new_tokens=2,
        )
        return inputs, gen

    def split_channels(self, generated_ids: torch.Tensor) -> Tuple[str, str]:
        text = self.tok.decode(generated_ids, skip_special_tokens=True).strip()
        return "", text
