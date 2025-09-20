# src/models/gpt_oss.py
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

if torch.cuda.is_available():
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")


def _enc(tok, s: str) -> List[int]:
    ids = tok.encode(s, add_special_tokens=False)
    return ids if isinstance(ids, list) else []


class _HarmonyStop(StoppingCriteria):
    def __init__(self, tok):
        self.end_ids = _enc(tok, "<|end|>")
        self.ret_ids = _enc(tok, "<|return|>")
        self.maxlen = max(len(self.end_ids), len(self.ret_ids))

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
        cache_dir: Optional[str] = None,
        use_chat_template: bool = True,      # Harmony on
        reasoning_effort: str = "low",
        force_final_prefix: bool = True,     # <- start directly in final
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
        self.force_final_prefix = force_final_prefix

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.eos_token_id

        # Detect usable "final" marker token sequences
        final_candidates = [
            "<|channel|>final<|message|>",
            "<|assistant|><|final|>",
            "<|final|>",
        ]
        self.final_ids: Optional[List[int]] = None
        for cand in final_candidates:
            ids = _enc(self.tok, cand)
            if ids:
                self.final_ids = ids
                break

        # End/return markers & stopper
        self.end_ids = _enc(self.tok, "<|end|>")
        self.ret_ids = _enc(self.tok, "<|return|>")
        self.stopper = StoppingCriteriaList([_HarmonyStop(self.tok)])

    # -------------------- input builders --------------------

    def _build_inputs_chat(self, user_msg: str) -> Dict[str, torch.Tensor]:
        assert hasattr(self.tok, "apply_chat_template"), "Tokenizer must provide Harmony chat template."
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

    def _append_final_prefix(self, enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Append the tokenizer's 'final' marker so generation starts inside final channel."""
        if not self.force_final_prefix or not self.final_ids:
            return enc
        ii, am = enc["input_ids"], enc["attention_mask"]
        final = torch.tensor(self.final_ids, device=ii.device).unsqueeze(0).repeat(ii.size(0), 1)
        enc["input_ids"] = torch.cat([ii, final], dim=1)
        enc["attention_mask"] = torch.cat([am, torch.ones_like(final)], dim=1)
        return enc

    # -------------------- channel utilities --------------------

    def _slice_final_ids(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Return only the span belonging to the Harmony 'final' channel (token-level)."""
        seq = generated_ids.tolist() if isinstance(generated_ids, torch.Tensor) else list(generated_ids)
        if not self.final_ids:
            return generated_ids

        F = self.final_ids
        n, m = len(seq), len(F)
        start_ix = -1
        # FULL SCAN for the last occurrence of F
        i = 0
        while i <= n - m:
            if seq[i:i+m] == F:
                start_ix = i + m
                i += m
            else:
                i += 1
        if start_ix == -1:
            return generated_ids

        def find_next(pat: List[int]) -> int:
            L = len(pat)
            if L == 0: return -1
            for j in range(start_ix, n - L + 1):
                if seq[j:j+L] == pat:
                    return j
            return -1

        stops = [x for x in (find_next(self.ret_ids), find_next(self.end_ids)) if x != -1]
        end_ix = min(stops) if stops else n
        if end_ix <= start_ix:
            end_ix = n

        kept = seq[start_ix:end_ix]
        return torch.tensor(kept, device=generated_ids.device, dtype=generated_ids.dtype)

    def split_channels(self, generated_ids: torch.Tensor) -> Tuple[str, str]:
        """
        Return (analysis_text, final_text) by decoding WITH specials and splitting on Harmony markers.
        Falls back to token-level slicing for final if no text markers found.
        """
        txt = self.tok.decode(generated_ids, skip_special_tokens=False)

        M_A = "<|channel|>analysis<|message|>"
        M_Fs = ["<|channel|>final<|message|>", "<|assistant|><|final|>", "<|final|>"]
        M_ENDS = ["<|return|>", "<|end|>", "<|start|>"]

        anal, final = "", ""
        last_f = -1
        picked = None
        for mf in M_Fs:
            j = txt.rfind(mf)
            if j > last_f:
                last_f = j; picked = mf
        if last_f != -1:
            tail = txt[last_f + len(picked):]
            k = min([i for i in (tail.find(t) for t in M_ENDS) if i != -1], default=len(tail))
            final = tail[:k].strip()

            ai = txt.find(M_A)
            if ai != -1 and ai < last_f:
                anal = txt[ai + len(M_A): last_f].strip()
            return anal, final

        # fallback: token-sliced final only
        final_ids = self._slice_final_ids(generated_ids)
        return "", self.tok.decode(final_ids, skip_special_tokens=True).strip()

    # -------------------- public APIs --------------------

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
        inputs = self._append_final_prefix(inputs)  # <- start in final if enabled
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,   # 128–256 is usually enough with Reasoning: low
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            use_cache=True,
            stopping_criteria=self.stopper,
        )
        return inputs, gen

    def decode(self, ids: torch.Tensor) -> str:
        # If we forced final, ids already start inside final -> direct decode is fine.
        if self.force_final_prefix:
            return self.tok.decode(ids, skip_special_tokens=True).strip()
        # Otherwise, extract final channel
        final_ids = self._slice_final_ids(ids)
        return self.tok.decode(final_ids, skip_special_tokens=True).strip()
