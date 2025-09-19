import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


@torch.no_grad()
def last_token_features(hidden_states: List[torch.Tensor], layers: List[int], reducer: str = "concat") -> torch.Tensor:
    # hidden_states: tuple(len=L+1) of [B, T, H]; we grab last token at chosen layers
    feats = []
    for li in layers:
        hs = hidden_states[li]  # layer index (0 is embeddings)
        feats.append(hs[:, -1, :])
    if reducer == "concat":
        return torch.cat(feats, dim=-1)
    elif reducer == "mean":
        return torch.stack(feats, dim=0).mean(0)
    else:  # last
        return feats[-1]


@torch.no_grad()
def logits_entropy(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V]; take entropy at last step
    last = logits[:, -1, :]
    probs = F.softmax(last, dim=-1)
    ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1, keepdim=True)
    return ent


@torch.no_grad()
def router_stats(output) -> Optional[torch.Tensor]:
    # Some models expose router logits/probs (MoE). If not present, return None
    r = getattr(output, "router_logits", None)
    if r is None:
        return None
    # Example: aggregate mean/max entropy across layers for last token
    # assume shape [L, B, T, E] or list per layer; handle simply
    if isinstance(r, (list, tuple)):
        stats = []
        for rl in r:
            rl_last = rl[:, -1, :]  # [B, E]
            p = F.softmax(rl_last, dim=-1)
            ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1, keepdim=True)
            stats.append(
                torch.cat([p.max(dim=-1, keepdim=True).values, ent], dim=-1))
        return torch.stack(stats, dim=0).mean(0)  # [B, 2]
    else:
        rl_last = r[:, -1, :]
        p = F.softmax(rl_last, dim=-1)
        ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1, keepdim=True)
        return torch.cat([p.max(dim=-1, keepdim=True).values, ent], dim=-1)

@torch.no_grad()
def _last_token_tuple(hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
    # HF returns a tuple: [embeds, layer1, ..., layerL]; each [B, T, H]
    # We keep last-timestep across all layers
    return [hs[:, -1, :] for hs in hidden_states]  # list len L+1 (0=embeds)

@torch.no_grad()
def unembed_logits(h: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    # h: [B, H], W: [V, H] (note: W is (V,H)), logits = h @ W^T + b
    logits = h @ W.t()
    if b is not None:
        logits = logits + b
    return logits  # [B, V]

@torch.no_grad()
def logit_lens_similarity(
    last_step_tuple: List[torch.Tensor],
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    layer_ids: List[int],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute MICE-like similarity features between the final decode (top layer)
    and each chosen intermediate layer's decode for the LAST token.
    Returns tensor [B, len(layer_ids)*2]: concat of [KL(final||layer), cosine(hL, h_l)]
    """
    # index 0 is embeddings; top hidden is last in tuple
    h_final = last_step_tuple[-1]                         # [B, H]
    p_final = F.softmax(unembed_logits(h_final, W, b) / temperature, dim=-1)  # [B, V]

    feats = []
    for li in layer_ids:
        h_l = last_step_tuple[li]                         # [B, H]
        p_l = F.softmax(unembed_logits(h_l, W, b) / temperature, dim=-1)
        # KL(final || layer)
        kl = (p_final * (p_final.clamp_min(1e-12) / p_l.clamp_min(1e-12)).log()).sum(dim=-1, keepdim=True)
        # cosine(h_final, h_l)
        cos = F.cosine_similarity(h_final, h_l, dim=-1, eps=1e-8).unsqueeze(-1)
        feats.append(torch.cat([kl, cos], dim=-1))
    return torch.cat(feats, dim=-1)  # [B, 2*len(layer_ids)]

@torch.no_grad()
def features_for_variant(
    variant: str,
    final_step_tuple: List[torch.Tensor],
    base_layers: List[int],
    reducer: str,
    include_entropy: bool,
    scores: Optional[List[torch.Tensor]],  # list of logits per step from generate() or None
    W: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    mice_layers: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Build a feature vector for the LAST generated token.
    - variant='baseline': last-token hidden states (selected layers) [+ entropy]
    - variant='mice': baseline + MICE similarities (KL & cosine) for mice_layers
    """
    # baseline core
    base = last_token_features(final_step_tuple, base_layers, reducer)  # [B, d]
    parts = [base]

    if include_entropy and scores:
        # stack per-step logits and take last step
        last_step_logits = torch.stack(scores, dim=1)[:, -1, :]  # [B, V]
        ent = logits_entropy(last_step_logits.unsqueeze(1))      # [B, 1]
        parts.append(ent)

    if variant == "mice":
        assert W is not None, "MICE features need unembedding W"
        assert mice_layers is not None and len(mice_layers) > 0
        sim = logit_lens_similarity(final_step_tuple, W, b, mice_layers)  # [B, 2*Lm]
        parts.append(sim)

    return torch.cat(parts, dim=-1)  # [B, d_total]
