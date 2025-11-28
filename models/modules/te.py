"""Transformer encoder block"""

import torch.nn as nn

from .mlp import MLP
from .multiha import MultiHA


class TE(nn.Module):
    def __init__(
        self,
        emb_size,
        attention,
        mlp,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_size)
        self.sa = MultiHA(attention)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = MLP(mlp)

    def forward(self, x, **attn_kwds):
        # SA block
        attended = self.sa(self.norm1(x), **attn_kwds)
        x = x + attended

        # MLP block
        mlped = self.mlp(self.norm2(x))
        x = x + mlped

        return x
