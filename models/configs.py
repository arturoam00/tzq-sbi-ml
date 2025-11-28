from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass
class _BaseConfig:
    @classmethod
    def cast(cls, conf):
        if isinstance(conf, cls):
            return conf
        if isinstance(conf, Mapping):
            return cls(**conf)
        raise NotImplementedError


@dataclass
class MLPConfig(_BaseConfig):
    k_factor: int
    activation: str
    dim_in: int = ...
    dim_out: int = ...
    n_hidden: int = 2
    bias: bool = True
    dropout_p: Optional[float] = None


@dataclass
class SAConfig(_BaseConfig):
    emb_size: int = ...
    num_heads: int = 8
    bias: bool = True
    dropout_p: Optional[float] = None
    increase_hidden_channels: int = 8
    # multi_query: bool = False
    # head_scale: bool = False

    @property
    def emb_head(self) -> int:
        assert (
            self.emb_size % self.num_heads == 0
        ), f"Embedding size {self.emb_size} not divisible by n. of heads {self.num_heads}"

        return self.emb_size // self.num_heads
