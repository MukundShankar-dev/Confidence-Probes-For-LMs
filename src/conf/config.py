from pydantic import BaseModel
from typing import List, Optional
import yaml


class ModelCfg(BaseModel):
    model_id: str = "openai/gpt-oss-20b"
    dtype: str = "bf16"
    device_map: str = "auto"
    max_new_tokens: int = 64
    use_router_probs: bool = True


class ProbeCfg(BaseModel):
    layers: List[int] = [16, 24, 32]
    reducer: str = "concat"  # concat | mean | last
    model: str = "logreg"  # logreg | mlp
    train_size: int = 8000


class RagCfg(BaseModel):
    backend: str = "wikipedia"  # wikipedia | faiss
    top_k: int = 5
    max_passage_len: int = 384
    reanswer_prompt: str = "use_evidence_or_refuse"


class DataCfg(BaseModel):
    dataset: str = "squad"
    split: str = "validation"
    limit: int = 12000
    dataset: str = "squad"
    split: str = "validation"
    limit: int = 12000


class LogCfg(BaseModel):
    wandb: bool = False


class Cfg(BaseModel):
    seed: int = 42
    model: ModelCfg = ModelCfg()
    probe: ProbeCfg = ProbeCfg()
    rag: RagCfg = RagCfg()
    data: DataCfg = DataCfg()
    logging: LogCfg = LogCfg()
    threshold: float = 0.5

    @staticmethod
    def load(path: str) -> "Cfg":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return Cfg(**raw)
