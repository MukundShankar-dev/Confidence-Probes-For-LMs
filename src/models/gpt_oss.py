# src/models/gpt_oss.py
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

if torch.cuda.is_available():
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    print(f"[CUDA] {props.name} | {props.total_memory/1e9:.1f} GB")

class _HarmonyStop(StoppingCriteria):
    def __init__(self, tok):
        self.end_ids = tok.encode("<|end|>", add_special_tokens=False)
        self.ret_ids = tok.encode("<|return|>", add_special_tokens=False)
        self.maxlen = max(len(self.end_ids), len(self.ret_ids))
    def __call__(self, input_ids, scores, **kwargs):
        if self.maxlen == 0: 
            return False
        seq = input_ids[0].tolist()
        tail = seq[-(self.maxlen+4):]
        def endswith(pat):
            L = len(pat)
            return L and tail[-L:] == pat
        return endswith(self.end_ids) or endswith(self.ret_ids)

class GPTOSS:
    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 64,
        use_router_probs: bool = True,
        cache_dir: str = None,
        use_chat_template: bool = True,   # IMPORTANT: Harmony format
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

        self.stopper = StoppingCriteriaList([_HarmonyStop(self.tok)])

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tok.eos_token_id

        # Harmony markers we’ll look for when parsing:
        self.MARK_FINAL = "<|channel|>final<|message|>"
        self.MARK_RETURN = "<|return|>"
        self.MARK_END = "<|end|>"
        self.MARK_START = "<|start|>"

    # ---------- input builders (Harmony) ----------

    def _build_inputs_chat(self, user_msg: str) -> Dict[str, torch.Tensor]:
        assert hasattr(self.tok, "apply_chat_template"), "gpt-oss needs the Harmony chat template"
        msgs = [
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user",   "content": user_msg},
        ]
        enc = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        )
        if isinstance(enc, torch.Tensor):
            enc = {"input_ids": enc}
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return enc
    
    def _build_inputs_plain(self, text: str) -> Dict[str, torch.Tensor]:
        # Fallback (not recommended for gpt-oss)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        return enc

    def _build_inputs(self, user_msg: str) -> Dict[str, torch.Tensor]:
        return self._build_inputs_chat(user_msg) if self.use_chat_template else self._build_inputs_plain(user_msg)

    # ---------- utils ----------

    def _extract_final_text(self, generated_ids: torch.Tensor) -> str:
        # decode WITH specials so we can see Harmony markers
        text_all = self.tok.decode(generated_ids, skip_special_tokens=False)
        # prefer the last final block if there are multiple
        ix = text_all.rfind(self.MARK_FINAL)
        if ix != -1:
            tail = text_all[ix + len(self.MARK_FINAL):]
            # cut at the first boundary
            for stop in (self.MARK_RETURN, self.MARK_END, self.MARK_START):
                j = tail.find(stop)
                if j != -1:
                    tail = tail[:j]
                    break
            return tail.strip()
        # fallback: strip specials if no markers
        return self.tok.decode(generated_ids, skip_special_tokens=True).strip()

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
        inputs = self._build_inputs_chat(prompt)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,   # try 96–128 for Harmony
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,  # keep EOS usable
            use_cache=True,
            stopping_criteria=self.stopper,
        )
        return inputs, gen

    def decode(self, ids: torch.Tensor) -> str:
        # IMPORTANT: parse out the Harmony 'final' message content
        return self._extract_final_text(ids)
