from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hydra.utils import instantiate
from lgatr import extract_scalar
from torch.nn.attention import sdpa_kernel
from torch_geometric.utils import scatter

from .base_wrapper import BaseWrapper
from .decorators import filter_empty_tensor_warning
from .embed import to_multivector, to_multivector_parametrized
from .utils import att_mask, get_backends, ptr2index

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BaseLGATrWrapper(BaseWrapper, ABC):
    def __init__(self, *args, **kwds):
        kwds["key"] = "LGATr"
        super().__init__(*args, **kwds)
        self.net = self.init_net(self.net)

    @filter_empty_tensor_warning
    def init_net(self, net: DictConfig):
        return instantiate(net)

    @abstractmethod
    def output(self, multivectors, index, scalars=None):
        pass

    @abstractmethod
    def embed_mv(self, *args, **kwargs):
        pass

    def forward(
        self, particles, ptr, scalars=None, force_math=False, embedding_kwargs={}
    ):
        # If I want to compute the gradient of the ouitput w.r.t. the
        # parameters I cannot use efficient backends for self attention
        backends = get_backends(force_math)

        index = ptr2index(ptr)
        attention_mask = att_mask(index)

        mv = self.embed_mv(particles, **embedding_kwargs)

        with sdpa_kernel(backends):
            # out_mv.size() -> (batch_idx, particles, out_mv_channels, 16)
            out_mv, out_s = self.net(
                multivectors=mv, scalars=scalars, attn_mask=attention_mask
            )

        return self.output(multivectors=out_mv, scalars=out_s, index=index)


class LocalLGATrWrapper(BaseLGATrWrapper):

    def embed_mv(self, particles, theta_dim):
        return to_multivector(particles).repeat(1, 1, theta_dim, 1)

    def output(self, multivectors, index, scalars=None):
        # out.size() -> (batch_idx, batch_size, out_mv_channels, 16)
        out = scatter(multivectors, index=index, dim=1, reduce="mean")

        # I assume there are as many output multivector channels as
        # number of dimensions of the score vector to be regressed
        return extract_scalar(out)[0, :, :, 0]  # (batch_size, out_mv_channels)


class ParametrizedLGATrWrapper(BaseLGATrWrapper):
    def embed_mv(self, particles, theta, ptr, mode):
        return to_multivector_parametrized(particles, theta, ptr, mode)

    def output(self, multivectors, index, scalars=None):

        # out.size() -> (batch_idx, batch_size, out_mv_channels, 16)
        out = scatter(multivectors, index=index, dim=1, reduce="mean")

        # I assume that the first multivector channel corresponds to the
        # fourmomenta embedding, and the others to the (scalar) embeddings of
        # the parameter vector. To extract the regressed log likelihood ratio I
        # take the scalar component of the fourmomenta embedding
        return extract_scalar(out)[0, :, :1, 0]  # (batch_size, 1)
