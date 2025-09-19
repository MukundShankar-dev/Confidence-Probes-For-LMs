from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")

class GPTOSS:
    def __init__(self, model_id: str, dtype: str = "bfloat16", device_map: str = "auto",
                 max_new_tokens: int = 64, use_router_probs: bool = True, cache_dir: str = None):
        torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        
        dm = getattr(self.model, "hf_device_map", None)
        first_dev = next(self.model.parameters()).device
        print(f"[GPTOSS] device map: {dm}")
        print(f"[GPTOSS] first param device: {first_dev}")

        self.max_new_tokens = max_new_tokens
        self.use_router_probs = use_router_probs
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.eos_token_id

    def _apply_chat_template(self, user_msg: str):
        """
        Always return a mapping suitable for generate(**inputs) and include attention_mask.
        """
        if hasattr(self.tok, "apply_chat_template"):
            msgs = [{"role": "user", "content": user_msg}]
            enc = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt"
            )
            # When HF returns a bare Tensor, wrap it into a dict
            if isinstance(enc, torch.Tensor):
                enc = {"input_ids": enc}
        else:
            enc = self.tok(user_msg, return_tensors="pt")

        # move to device and normalize to dict
        if isinstance(enc, dict):
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
        else:
            enc = {"input_ids": enc.to(self.model.device)}

        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return enc

    def get_unembedding(self):
        """
        Return the unembedding (lm_head) weight and bias for logit-lens style features.
        Most models tie embeddings; bias may be None.
        """
        W = self.model.get_output_embeddings().weight  # [V, H]
        b = getattr(self.model.get_output_embeddings(), "bias", None)
        return W, b

    @torch.no_grad()
    def prefill_states(self, prompt: str):
        """
        Run a prefill-only forward to get hidden states before any new tokens are generated.
        """
        inputs = self._apply_chat_template(prompt)
        out = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        return inputs, out.hidden_states  # tuple(len = L+1)


    @torch.no_grad()
    def generate_with_states(self, prompt: str):
        inputs = self._apply_chat_template(prompt)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,          # we use this for entropy features later
            output_hidden_states=True,   # capture per-step hidden states
        )
        return inputs, gen

    def decode(self, ids: torch.Tensor) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)
