# src/models/gpt_oss.py
from typing import Dict, List, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ------------------ small util ------------------
def _enc(tok, s: str) -> List[int]:
    """Encode a literal string to token IDs without specials (safe)."""
    try:
        ids = tok.encode(s, add_special_tokens=False)
        return ids if isinstance(ids, list) else []
    except Exception:
        return []


# ------------------ GPU banner ------------------
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")


# ------------------ Stoppers --------------------
class _HarmonyStopAfter(StoppingCriteria):
    """
    Final-only mode stopper:
      - Stop when <|end|> or <|return|> appears,
      - BUT only after we have generated at least `min_new_tokens_after_start`
        tokens beyond the prompt (prevents empty finals).
    """
    def __init__(self, tok, min_new_tokens_after_start: int = 2):
        self.end_ids = _enc(tok, "<|end|>")
        self.ret_ids = _enc(tok, "<|return|>")
        self.maxlen  = max(len(self.end_ids), len(self.ret_ids))
        self.start_len = None
        self.min_new = int(min_new_tokens_after_start)

    def set_start_len(self, start_len: int):
        self.start_len = int(start_len)

    def _endswith(self, tail: List[int], pat: List[int]) -> bool:
        L = len(pat)
        return L > 0 and len(tail) >= L and tail[-L:] == pat

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.maxlen == 0 or self.start_len is None:
            return False
        seq  = input_ids[0].tolist()
        tail = seq[-(self.maxlen + 8):]

        # Enforce at least N tokens beyond prompt length
        if len(seq) - self.start_len < self.min_new:
            return False

        return self._endswith(tail, self.end_ids) or self._endswith(tail, self.ret_ids)


class _StopOnEndOrAfterFinalN(StoppingCriteria):
    """
    Think-then-final stopper:
      • Allow analysis until we see the final marker.
      • Then allow up to `final_allowance` tokens inside final.
      • Always stop on <|end|> or <|return|>, but not *before* a small
        number of new tokens are generated (min_new_after_start).
      • Cap runaway analysis via `analysis_cap` (counted *after* prompt).
    """
    def __init__(
        self,
        final_ids,
        end_ids,
        ret_ids,
        final_allowance=32,
        analysis_cap=256,
        min_new_after_start=2,
    ):
        self.final_ids       = final_ids or []
        self.end_ids         = end_ids   or []
        self.ret_ids         = ret_ids   or []
        self.final_allowance = int(final_allowance)
        self.analysis_cap    = int(analysis_cap)
        self.min_new         = int(min_new_after_start)

        self.final_start     = None   # absolute index where final begins
        self.start_len       = None   # absolute prompt length (tokens)

    def set_start_len(self, start_len: int):
        self.start_len = int(start_len)

    def _find_sub(self, seq, pat, start=0):
        n, m = len(seq), len(pat)
        if m == 0: return -1
        for i in range(start, n - m + 1):
            if seq[i:i+m] == pat:
                return i
        return -1

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_len is None:
            return False

        seq = input_ids[0].tolist()
        new = len(seq) - self.start_len  # tokens generated so far

        # Stop on explicit end markers, but only after a tiny floor of new tokens
        if new >= self.min_new:
            for pat in (self.end_ids, self.ret_ids):
                if self._find_sub(seq, pat, self.start_len) != -1:
                    return True

        # Cap runaway analysis if final never appears (count *after* prompt)
        if self.final_start is None and self.analysis_cap > 0 and new >= self.analysis_cap:
            return True

        # Detect final start (search only *after* the prompt)
        if self.final_start is None and self.final_ids:
            j = self._find_sub(seq, self.final_ids, self.start_len)
            if j != -1:
                self.final_start = j + len(self.final_ids)

        # After final begins, stop once we emitted N tokens of final
        if self.final_start is not None and self.final_allowance > 0:
            if len(seq) - self.final_start >= self.final_allowance:
                return True

        return False


# ------------------ Wrapper ---------------------
class GPTOSS:
    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        use_router_probs: bool = True,
        cache_dir: Optional[str] = None,
        use_chat_template: bool = True,      # Harmony chat
        reasoning_effort: str = "low",       # "low" | "medium" | "high" (goes into system msg)
        force_final_prefix: bool = True,     # True: final-only (fast); False: think-then-final
        final_allowance: int = 32,           # tokens allowed inside final (think mode)
        analysis_cap: int = 256,             # max tokens allowed before final (think mode)
        append_space_after_final: bool = True,  # reduce chance of immediate <|end|>
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

        self.max_new_tokens     = max_new_tokens
        self.use_router_probs   = use_router_probs
        self.use_chat_template  = use_chat_template
        self.reasoning_effort   = reasoning_effort
        self.force_final_prefix = force_final_prefix
        self.final_allowance    = int(final_allowance)
        self.analysis_cap       = int(analysis_cap)
        self.append_space_after_final = append_space_after_final

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.eos_token_id

        # Detect Harmony markers
        analysis_candidates = [
            "<|channel|>analysis<|message|>",
            "<|assistant|><|analysis|>",
            "<|analysis|>",
        ]
        final_candidates = [
            "<|channel|>final<|message|>",
            "<|assistant|><|final|>",
            "<|final|>",
        ]
        self.analysis_ids: Optional[List[int]] = None
        for cand in analysis_candidates:
            ids = _enc(self.tok, cand)
            if ids:
                self.analysis_ids = ids
                break

        self.final_ids: Optional[List[int]] = None
        for cand in final_candidates:
            ids = _enc(self.tok, cand)
            if ids:
                self.final_ids = ids
                break

        self.end_ids = _enc(self.tok, "<|end|>")
        self.ret_ids = _enc(self.tok, "<|return|>")

        # Stoppers
        self.stopper_finalonly = _HarmonyStopAfter(self.tok, min_new_tokens_after_start=2)
        self._think_guard = _StopOnEndOrAfterFinalN(
            final_ids=self.final_ids,
            end_ids=self.end_ids,
            ret_ids=self.ret_ids,
            final_allowance=self.final_allowance,
            analysis_cap=self.analysis_cap,
            min_new_after_start=2,
        )
        self.stopper_think = StoppingCriteriaList([self._think_guard])

        # Cached single space token (neutral, language-agnostic)
        sp = _enc(self.tok, " ")
        self._space_ids = sp if sp else None

    # ------------- inputs -------------
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
        """
        Append 'final' marker so generation starts inside final channel.
        Also append a single space token to discourage immediate <|end|>.
        """
        if not self.final_ids:
            return enc
        ii, am = enc["input_ids"], enc["attention_mask"]

        toks = [self.final_ids]
        if self.append_space_after_final and self._space_ids:
            toks.append(self._space_ids)

        extra = torch.tensor(
            [t for seg in toks for t in seg],
            device=ii.device,
            dtype=ii.dtype
        ).unsqueeze(0).repeat(ii.size(0), 1)

        enc["input_ids"]      = torch.cat([ii, extra], dim=1)
        enc["attention_mask"] = torch.cat([am, torch.ones_like(extra)], dim=1)
        return enc

    def _append_analysis_prefix(self, enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Append 'analysis' marker so generation starts in the analysis channel.
        """
        if not self.analysis_ids:
            return enc
        ii, am = enc["input_ids"], enc["attention_mask"]
        extra = torch.tensor(self.analysis_ids, device=ii.device, dtype=ii.dtype).unsqueeze(0).repeat(ii.size(0), 1)
        enc["input_ids"]      = torch.cat([ii, extra], dim=1)
        enc["attention_mask"] = torch.cat([am, torch.ones_like(extra)], dim=1)
        return enc

    # ------------- channels -------------
    def _slice_final_ids(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Return only the token span belonging to the 'final' channel, via token-level markers."""
        seq = generated_ids.tolist() if isinstance(generated_ids, torch.Tensor) else list(generated_ids)
        if not self.final_ids:
            return generated_ids

        F  = self.final_ids
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

        stops  = [x for x in (find_next(self.ret_ids), find_next(self.end_ids)) if x != -1]
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

        M_A   = "<|channel|>analysis<|message|>"
        M_Fs  = ["<|channel|>final<|message|>", "<|assistant|><|final|>", "<|final|>"]
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

        # fallback: token-sliced final only
        final_ids = self._slice_final_ids(generated_ids)
        return "", self.tok.decode(final_ids, skip_special_tokens=True).strip()

    # ------------- public APIs -------------
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
        """
        Final-only (force_final_prefix=True): start directly in final channel (fast path).
        Think-then-final with Harmony (use_chat_template=True, force_final_prefix=False):
        1) Generate in analysis until <|final|> or analysis_cap.
        2) If no <|final|> appeared, append <|final|> and continue to get the final.
        Textual mode (use_chat_template=False): just a plain generate.
        """
        inputs = self._build_inputs(prompt)

        # ---------- Final-only: keep your existing fast path ----------
        if self.use_chat_template and self.force_final_prefix:
            inputs = self._append_final_prefix(inputs)
            start_len = int(inputs["input_ids"].shape[-1])
            self.stopper_finalonly.set_start_len(start_len)
            stopper = StoppingCriteriaList([self.stopper_finalonly])

            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                use_cache=True,
                stopping_criteria=stopper,
                min_new_tokens=2,
            )
            return inputs, gen

        # ---------- Think-then-final with Harmony (two-stage) ----------
        if self.use_chat_template and not self.force_final_prefix:
            # Stage-1: start in analysis channel
            inputs1 = self._append_analysis_prefix({k: v.clone() for k, v in inputs.items()})
            start_len1 = int(inputs1["input_ids"].shape[-1])
            # reset guard
            self._think_guard.final_start = None
            self._think_guard.set_start_len(start_len1)
            stopper1 = self.stopper_think

            gen1 = self.model.generate(
                **inputs1,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                use_cache=True,
                stopping_criteria=stopper1,
            )

            seq1 = gen1.sequences  # [1, L1]
            # If <|final|> appeared during stage-1, we're done (guard found it)
            if self._think_guard.final_start is not None:
                return inputs, gen1

            # Stage-2: no final seen → append <|final|> (and an optional space) to seq1 and continue
            append_ids = []
            if self.final_ids:
                append_ids.extend(self.final_ids)
            if self.append_space_after_final and self._space_ids:
                append_ids.extend(self._space_ids)
            if append_ids:
                extra = torch.tensor(append_ids, device=seq1.device, dtype=seq1.dtype).unsqueeze(0)
                input_ids2 = torch.cat([seq1, extra], dim=1)
            else:
                input_ids2 = seq1  # extremely unlikely (no known final token)

            attn2 = torch.ones_like(input_ids2)
            start_len2 = int(input_ids2.shape[-1])
            # Stop after <|end|>/<|return|> or after final_allowance tokens (whichever first)
            stopper2 = self._HarmonyStopAfter(self.tok, min_new_tokens_after_start=2) if False else None
            # The above local class reference won't exist; reuse the existing stopper:
            stopper2 = self.stopper_finalonly
            self.stopper_finalonly.set_start_len(start_len2)

            gen2 = self.model.generate(
                input_ids=input_ids2,
                attention_mask=attn2,
                max_new_tokens=max(2, self.final_allowance),
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,  # optional; stage-2 HS usually not needed for eval
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([self.stopper_finalonly]),
            )
            # Return stage-2 result (it already includes the whole prefix). `inputs` stays the original.
            return inputs, gen2

        # ---------- Plain-text mode (no Harmony) ----------
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            use_cache=True,
        )
        return inputs, gen