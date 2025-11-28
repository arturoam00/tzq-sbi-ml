"""MLP block"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from ..configs import MLPConfig

activation_fn_map = dict(relu=nn.ReLU, gelu=nn.GELU, sigmoid=nn.Sigmoid, tanh=nn.Tanh)


class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()

        self.activation = activation_fn_map.get(config.activation)

        if self.activation is None:
            raise ValueError(f"Invalid activation: '{config.activation}'")

        layers = []
        input_dim = config.dim_in
        hidden_units = config.k_factor * config.dim_in
        for _ in range(config.n_hidden):
            layers.append(nn.Linear(input_dim, hidden_units, bias=config.bias))
            layers.append(self.activation())
            if config.dropout_p > 0.0:
                layers.append(nn.Dropout(config.dropout_p))
            input_dim = hidden_units
        layers.append(nn.Linear(input_dim, config.dim_out, bias=config.bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
