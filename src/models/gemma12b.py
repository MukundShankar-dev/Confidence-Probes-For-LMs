# src/models/gemma12b.py
import torch
from typing import Dict, Tuple
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

class Gemma12B:
    """
    GPTOSS-compatible adapter for Gemma Instruct (no Harmony).
    Uses chat template to avoid degenerate outputs.
    Exposes:
      - .tok
      - generate_with_states(prompt) -> (inputs_dict, gen_output)
      - split_channels(generated_ids) -> ("", final_text)
    """
    def __init__(
        self,
        model_id: str = "google/gemma-2-12b-it",   # use the 12B IT checkpoint you have access to
        dtype: str = "float16",
        device_map: str = None,                    # full GPU (no offload)
        max_new_tokens: int = 128,
        cache_dir: str = None,
        system_prompt: str = "Answer with only the short factual span (1–5 words). No punctuation.",
        use_chat_template: bool = True,
    ):
        torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16
        self.tok = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, cache_dir=cache_dir, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,              # None -> push to CUDA below
            cache_dir=cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda").eval()

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.eos_token_id

        self.max_new_tokens = int(max_new_tokens)
        self._stopper = _StopOnNewlineOrEos(self.tok, min_new_tokens_after_start=2)
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template

        # banner
        dm = getattr(self.model, "hf_device_map", None)
        first_dev = next(self.model.parameters()).device
        print(f"[Gemma12B] device map: {dm}")
        print(f"[Gemma12B] first param device: {first_dev} | dtype={torch_dtype} | attn=sdpa")

    # ---------- inputs ----------
    def _build_inputs(self, user_text: str) -> Dict[str, torch.Tensor]:
        """
        Build inputs using Gemma's chat template so BOS + special tokens are correct.
        We place the full few-shot block + 'Q: ...\\nA: ' as a single user message.
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
            # attention mask is inferred if missing; add for safety:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            return inputs
        else:
            # fallback: plain encode WITH special tokens
            enc = self.tok(user_text, return_tensors="pt", add_special_tokens=True)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            if "attention_mask" not in enc:
                enc["attention_mask"] = torch.ones_like(enc["input_ids"])
            return enc

    # ---------- public APIs (GPTOSS-compatible) ----------
    @torch.no_grad()
    def generate_with_states(self, prompt: str):
        """
        Final-only textual generation. Returns (inputs, gen) like GPTOSS.
        The `prompt` string is your few-shot-final-only block (from run_baseline).
        """
        inputs = self._build_inputs(prompt)
        self._stopper.set_start_len(int(inputs["input_ids"].shape[-1]))

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=False,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([self._stopper]),
            min_new_tokens=2,
        )
        return inputs, gen

    def split_channels(self, generated_ids: torch.Tensor) -> Tuple[str, str]:
        text = self.tok.decode(generated_ids, skip_special_tokens=True).strip()
        return "", text
