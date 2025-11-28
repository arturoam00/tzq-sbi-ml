from abc import ABC, abstractmethod

import torch

from .base_wrapper import BaseWrapper


class MLPWrapper(BaseWrapper, ABC):
    def __init__(self, *args, **kwds):
        kwds["key"] = "MLP"
        super().__init__(*args, **kwds)

    @abstractmethod
    def embed_x(self, *args, **kwargs):
        pass

    def forward(self, x, **embedding_kargs):
        embedding = self.embed_x(x, **embedding_kargs)
        return self.net(embedding)


class LocalMLPWrapper(MLPWrapper):
    def embed_x(self, x):
        return x


class ParametrizedMLPWrapper(MLPWrapper):
    def embed_x(self, x, theta):
        return torch.cat((theta, x), dim=1)
