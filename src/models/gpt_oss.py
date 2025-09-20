from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

if torch.cuda.is_available():
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")


def _encode_or_none(tok, s: str) -> Optional[List[int]]:
    ids = tok.encode(s, add_special_tokens=False)
    return ids if isinstance(ids, list) and len(ids) > 0 else None


class _HarmonyStop(StoppingCriteria):
    """Stop when <|end|> or <|return|> token sequence appears at the end."""
    def __init__(self, tok, end_ids: List[int], ret_ids: List[int]):
        self.end_ids = end_ids
        self.ret_ids = ret_ids
        self.maxlen = max(len(end_ids), len(ret_ids))

    def _endswith(self, tail: List[int], pat: List[int]) -> bool:
        L = len(pat)
        return L > 0 and len(tail) >= L and tail[-L:] == pat

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.maxlen == 0:
            return False
        seq = input_ids[0].tolist()
        tail = seq[-(self.maxlen + 8):]
        return self._endswith(tail, self.end_ids) or self._endswith(tail, self.ret_ids)


class GPTOSS:
    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        use_router_probs: bool = True,
        cache_dir: str = None,
        use_chat_template: bool = True,         # Harmony format on
        reasoning_effort: str = "low",          # "low" | "medium" | "high"
    ):
        torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map, cache_dir=cache_dir
        )

        dm = getattr(self.model, "hf_device_map", None)
        first_dev = next(self.model.parameters()).device
        print(f"[GPTOSS] device map: {dm}")
        print(f"[GPTOSS] first param device: {first_dev}")

        self.max_new_tokens = max_new_tokens
        self.use_router_probs = use_router_probs
        self.use_chat_template = use_chat_template
        self.reasoning_effort = reasoning_effort

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.eos_token_id

        # Detect Harmony markers as TOKEN IDS (robust across templates)
        # Try several common encodings for final; pick the first that exists.
        final_candidates = [
            "<|assistant|><|final|>",
            "<|final|>",
            "<|channel|>final<|message|>",
        ]
        self.final_ids: Optional[List[int]] = None
        for cand in final_candidates:
            enc = _encode_or_none(self.tok, cand)
            if enc:
                self.final_ids = enc
                break

        # End markers
        self.end_ids  = _encode_or_none(self.tok, "<|end|>")    or []
        self.ret_ids  = _encode_or_none(self.tok, "<|return|>") or []

        # Stopping (halts as soon as model emits end/return)
        self.stopper = StoppingCriteriaList([_HarmonyStop(self.tok, self.end_ids, self.ret_ids)])

    # ---------- Harmony inputs ----------

    def _build_inputs_chat(self, user_msg: str) -> Dict[str, torch.Tensor]:
        assert hasattr(self.tok, "apply_chat_template"), "Tokenizer must provide Harmony chat template for gpt-oss."
        msgs = [
            {"role": "system", "content": f"Reasoning: {self.reasoning_effort}"},
            {"role": "user",   "content": user_msg},
        ]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if isinstance(enc, torch.Tensor):
            enc = {"input_ids": enc}
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return enc

    def _build_inputs_plain(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return enc

    def _build_inputs(self, user_msg: str) -> Dict[str, torch.Tensor]:
        return self._build_inputs_chat(user_msg) if self.use_chat_template else self._build_inputs_plain(user_msg)

    # ---------- token-level final extraction ----------

    def _slice_final_ids(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """
        Return only the token span belonging to the Harmony 'final' channel:
        [ ... <final_ids>  (content...)  (<return>|<end>|<start>) ... ]
        If no final marker found, return the original ids.
        """
        seq = generated_ids.tolist() if isinstance(generated_ids, torch.Tensor) else list(generated_ids)
        if not self.final_ids:
            return generated_ids

        # find last occurrence of final_ids
        F = self.final_ids
        n, m = len(seq), len(F)
        start_ix = -1
        for i in range(max(0, n - 3 * (m + 1)) , n - m + 1):  # small scan window near the end
            if seq[i:i+m] == F:
                start_ix = i + m
        if start_ix == -1:
            return generated_ids  # no final marker; fall back

        # find nearest boundary after start_ix
        def find_next(pat: List[int]) -> int:
            L = len(pat)
            if L == 0: return -1
            for j in range(start_ix, n - L + 1):
                if seq[j:j+L] == pat:
                    return j
            return -1

        stops = [x for x in [find_next(self.ret_ids), find_next(self.end_ids)] if x != -1]
        end_ix = min(stops) if stops else n
        if end_ix <= start_ix:
            end_ix = n

        kept = seq[start_ix:end_ix]
        return torch.tensor(kept, device=generated_ids.device, dtype=generated_ids.dtype)

    # ---------- public APIs ----------

    def get_unembedding(self):
        W = self.model.get_output_embeddings().weight
        b = getattr(self.model.get_output_embeddings(), "bias", None)
        return W, b

    @torch.no_grad()
    def prefill_states(self, prompt: str):
        inputs = self._build_inputs(prompt)
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        return inputs, out.hidden_states

    @torch.no_grad()
    def generate_with_states(self, prompt: str):
        inputs = self._build_inputs(prompt)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,   # 128–256 is usually enough at Reasoning: low
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            use_cache=True,
            stopping_criteria=self.stopper,       # stop on <|end|> / <|return|>
        )
        return inputs, gen

    def decode(self, ids: torch.Tensor) -> str:
        # slice out the final-channel content (token-level), then decode cleanly
        final_ids = self._slice_final_ids(ids)
        return self.tok.decode(final_ids, skip_special_tokens=True).strip()
