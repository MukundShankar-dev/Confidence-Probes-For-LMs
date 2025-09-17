from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTOSS:
    def __init__(self, model_id: str, dtype: str = "bf16", device_map: str = "auto", max_new_tokens: int = 64, use_router_probs: bool = True):
        torch_dtype = getattr(torch, dtype) if hasattr(
            torch, dtype) else torch.float16
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.max_new_tokens = max_new_tokens
        self.use_router_probs = use_router_probs

    def _apply_chat_template(self, user_msg: str) -> torch.Tensor:
        if hasattr(self.tok, "apply_chat_template"):
            msgs = [{"role": "user", "content": user_msg}]
            return self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        else:
            return self.tok(user_msg, return_tensors="pt").to(self.model.device)

    @torch.no_grad()
    def prefill(self, prompt: str):

        inp = self._apply_chat_template(prompt)
        out = self.model(
            **inp,
            output_hidden_states=True,
            return_dict=True,
            **({"output_router_probs": True} if self.use_router_probs else {}),
        )
        return inp, out

    @torch.no_grad()
    def generate_with_states(self, prompt: str):

        inp = self._apply_chat_template(prompt)
        gen = self.model.generate(
            **inp,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        return inp, gen

    def decode(self, ids: torch.Tensor) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)
