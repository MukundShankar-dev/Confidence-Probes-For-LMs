import torch.nn.functional as F
import torch
from typing import List, Dict, Optional


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
