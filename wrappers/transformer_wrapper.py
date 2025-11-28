from abc import ABC, abstractmethod

import torch
from torch.nn.attention import sdpa_kernel
from torch_geometric.utils import scatter

from .base_wrapper import BaseWrapper
from .utils import att_mask, get_backends, ptr2index


class BaseTransformerWrapper(BaseWrapper, ABC):
    def __init__(self, *args, **kwds):
        kwds["key"] = "Transformer"
        super().__init__(*args, **kwds)

    @abstractmethod
    def embed(self, *args, **kwds):
        pass

    def forward(self, particles, ptr, force_math=False, embedding_kwargs={}):
        """
        particles: (num_particles, 4)
        """
        backends = get_backends(force_math)

        index = ptr2index(ptr)
        attention_mask = att_mask(index)

        tokens = self.embed(particles, **embedding_kwargs)

        with sdpa_kernel(backends):
            out = self.net(tokens, attn_mask=attention_mask)

        return scatter(src=out, index=index, dim=0, reduce="mean")


class LocalTransformerWrapper(BaseTransformerWrapper):
    def embed(self, tokens, **kwds):
        return tokens


class ParametrizedTransformerWrapper(BaseTransformerWrapper):
    def embed(
        self, particles: torch.Tensor, theta: torch.Tensor, ptr: torch.Tensor, **kwds
    ):
        """
        particles: (num_particles, 4)
        theta: (batch_size, theta_dim)
        ptr: (batch_size + 1,)

        Right now I'm just concateneting the theta vector with the particles
        fourmomenta, then embeddings are created for the concatenated vector
        """
        n, e = particles.shape
        theta_dim = theta.shape[-1]

        ptr = ptr.to(dtype=torch.long)

        theta = theta.repeat_interleave(ptr[1:] - ptr[:-1], dim=0)

        assert theta.size() == (n, theta_dim)

        tokens = torch.cat((theta, particles), dim=-1)

        assert tokens.size() == (n, e + theta_dim)

        return tokens
