# src/models/gpt_oss.py
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# --- tiny util ---


def _enc(tok, s: str) -> List[int]:
    try:
        ids = tok.encode(s, add_special_tokens=False)
        return ids if isinstance(ids, list) else []
    except Exception:
        return []

# --- GPU banner (nice to have) ---
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")


# ---------- Stoppers ----------
class _HarmonyStop(StoppingCriteria):
    """Stop when <|end|> or <|return|> appears anywhere."""
    def __init__(self, tok):
        self.end_ids = _enc(tok, "<|end|>")
        self.ret_ids = _enc(tok, "<|return|>")
        self.maxlen = max(len(self.end_ids), len(self.ret_ids))

    def _endswith(self, tail: List[int], pat: List[int]) -> bool:
        L = len(pat)
        return L > 0 and len(tail) >= L and tail[-L:] == pat

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.maxlen == 0:  # markers absent
            return False
        seq = input_ids[0].tolist()
        tail = seq[-(self.maxlen + 8):]
        return self._endswith(tail, self.end_ids) or self._endswith(tail, self.ret_ids)


class _StopOnEndOrAfterFinalN(StoppingCriteria):
    """
    Think-then-final: allow analysis until <|final|>, then permit up to N tokens of final.
    Also hard-stop on <|end|>/<|return|>, and cap runaway analysis via analysis_cap.
    """

    def __init__(self, final_ids, end_ids, ret_ids, final_allowance=32, analysis_cap=256):
        self.final_ids = final_ids or []
        self.end_ids = end_ids or []
        self.ret_ids = ret_ids or []
        self.final_allowance = int(final_allowance)
        self.analysis_cap = int(analysis_cap)
        self.final_start = None  # index where final content begins

    def _find_sub(self, seq, pat, start=0):
        n, m = len(seq), len(pat)
        if m == 0:
            return -1
        for i in range(start, n - m + 1):
            if seq[i:i+m] == pat:
                return i
        return -1

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()

        # stop if end/return appears
        for pat in (self.end_ids, self.ret_ids):
            if self._find_sub(seq, pat) != -1:
                return True

        # cap runaway analysis before final appears
        if self.final_start is None and self.analysis_cap > 0 and len(seq) >= self.analysis_cap:
            return True

        # track final start
        if self.final_start is None and self.final_ids:
            j = self._find_sub(seq, self.final_ids, 0)
            if j != -1:
                self.final_start = j + len(self.final_ids)

        # after final begins, stop once we emitted N tokens of final
        if self.final_start is not None and self.final_allowance > 0:
            if len(seq) - self.final_start >= self.final_allowance:
                return True

        return False


# ---------- Main wrapper ----------
class GPTOSS:
    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        use_router_probs: bool = True,
        cache_dir: Optional[str] = None,
        use_chat_template: bool = True,        # Harmony on
        reasoning_effort: str = "low",         # "low" | "medium" | "high"
        # final-only (fast baseline) if True
        force_final_prefix: bool = True,
        # tokens allowed in final (think mode)
        final_allowance: int = 32,
        # max tokens allowed before final (think mode)
        analysis_cap: int = 256,
    ):
        torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map, cache_dir=cache_dir
        )

        # banners
        dm = getattr(self.model, "hf_device_map", None)
        first_dev = next(self.model.parameters()).device
        print(f"[GPTOSS] device map: {dm}")
        print(f"[GPTOSS] first param device: {first_dev}")

        self.max_new_tokens = max_new_tokens
        self.use_router_probs = use_router_probs
        self.use_chat_template = use_chat_template
        self.reasoning_effort = reasoning_effort
        self.force_final_prefix = force_final_prefix
        self.final_allowance = int(final_allowance)
        self.analysis_cap = int(analysis_cap)

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.eos_token_id

        # detect Harmony markers
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

        self.end_ids = _enc(self.tok, "<|end|>")
        self.ret_ids = _enc(self.tok, "<|return|>")

        # stoppers
        self.stopper_finalonly = StoppingCriteriaList([_HarmonyStop(self.tok)])
        self.stopper_think = StoppingCriteriaList([
            _StopOnEndOrAfterFinalN(
                final_ids=self.final_ids,
                end_ids=self.end_ids,
                ret_ids=self.ret_ids,
                final_allowance=self.final_allowance,
                analysis_cap=self.analysis_cap,
            )
        ])

    # -------- inputs --------
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
        """Append 'final' marker so generation starts inside final channel."""
        if not self.force_final_prefix or not self.final_ids:
            return enc
        ii, am = enc["input_ids"], enc["attention_mask"]
        final = torch.tensor(self.final_ids, device=ii.device).unsqueeze(0).repeat(ii.size(0), 1)
        enc["input_ids"] = torch.cat([ii, final], dim=1)
        enc["attention_mask"] = torch.cat([am, torch.ones_like(final)], dim=1)
        return enc

    # -------- channels --------
    def _slice_final_ids(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Return only the token span belonging to the 'final' channel, via token-level markers."""
        seq = generated_ids.tolist() if isinstance(generated_ids, torch.Tensor) else list(generated_ids)
        if not self.final_ids:
            return generated_ids

        F = self.final_ids
        n, m = len(seq), len(F)
        start_ix = -1
        i = 0
        while i <= n - m:
            if seq[i:i+m] == F:
                start_ix = i + m
                i += m
            else:
                i += 1
        if start_ix == -1:
            return generated_ids

        def find_next(pat):
            L = len(pat)
            if L == 0: return -1
            for j in range(start_ix, n - L + 1):
                if seq[j:j+L] == pat:
                    return j
            return -1

        stops = [x for x in (find_next(self.ret_ids),
                             find_next(self.end_ids)) if x != -1]
        end_ix = min(stops) if stops else n
        if end_ix <= start_ix:
            end_ix = n
        kept = seq[start_ix:end_ix]
        return torch.tensor(kept, device=generated_ids.device, dtype=generated_ids.dtype)

    def split_channels(self, generated_ids: torch.Tensor) -> Tuple[str, str]:
        """
        Return (analysis_text, final_text) by decoding WITH specials and splitting on Harmony markers.
        Falls back to token-sliced final when no textual markers are found.
        """
        txt = self.tok.decode(generated_ids, skip_special_tokens=False)

        M_A = "<|channel|>analysis<|message|>"
        M_Fs = ["<|channel|>final<|message|>",
                "<|assistant|><|final|>", "<|final|>"]
        M_END = ["<|return|>", "<|end|>", "<|start|>"]

        anal, final = "", ""
        last_f, picked = -1, None
        for mf in M_Fs:
            j = txt.rfind(mf)
            if j > last_f:
                last_f, picked = j, mf
        if last_f != -1:
            tail = txt[last_f + len(picked):]
            stops = [i for i in (tail.find(t) for t in M_END) if i != -1]
            k = min(stops) if stops else len(tail)
            final = tail[:k].strip()

            ai = txt.find(M_A)
            if ai != -1 and ai < last_f:
                anal = txt[ai + len(M_A): last_f].strip()
            return anal, final

        # token fallback (final only)
        final_ids = self._slice_final_ids(generated_ids)
        return "", self.tok.decode(final_ids, skip_special_tokens=True).strip()

    # -------- public APIs --------
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

        if self.force_final_prefix:
            inputs = self._append_final_prefix(inputs)
            stopper = self.stopper_finalonly
        else:
            stopper = self.stopper_think

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            return_dict_in_generate=True,
            output_scores=True,           # keep on if you need entropy features
            output_hidden_states=True,    # keep on for probe features; off for speed
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            use_cache=True,
            stopping_criteria=stopper,
        )
        return inputs, gen

    def decode(self, ids: torch.Tensor) -> str:
        if self.force_final_prefix:
            return self.tok.decode(ids, skip_special_tokens=True).strip()
        # else: extract final
        final_ids = self._slice_final_ids(ids)
        return self.tok.decode(final_ids, skip_special_tokens=True).strip()
