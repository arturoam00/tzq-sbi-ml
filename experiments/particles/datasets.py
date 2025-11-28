from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from .schemas import ParametrizedParticlesEvent, ParticlesEvent


def _reshape_particles(x: np.ndarray):
    samples, features = x.shape
    max_particles = features // 4

    # Each particle is represnted by its 4-momenta
    assert not features % 4, "invalid number of features"

    # (samples, max_particles, 4)
    return x.reshape(samples, max_particles, 4).astype(np.float32)


def _sample_lengths(x: np.ndarray):
    mask = np.abs(x).sum(axis=-1) > 0  # (samples, max_particles)
    return mask.sum(axis=-1)


# TODO: No need to store the whole (sparse) data set, or do so in a more
# mem. eff. way (e.g. scipy.sparse)
class ParticlesDataset(Dataset):
    def __init__(
        self, *, x: np.ndarray, score: Optional[np.ndarray] = None, **kwds
    ) -> None:
        # allow for unlabelled data
        score = score if score is not None else np.zeros((len(x), 1))
        assert len(x) == len(score), f"x and y differ in length"

        self._x = _reshape_particles(x)
        self._lengths = _sample_lengths(self._x)
        self._score = score  # (nsamples, ??) depends on dim(theta)

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, index) -> ParticlesEvent:
        # Return all particles (including non-available ones) with
        # together with the number of available particles
        # NOTE: I assume the first `self._lengths[index]` are
        # available
        return ParticlesEvent(
            fourmomenta=self._x[index],
            length=int(self._lengths[index]),
            score=self._score[index],
        )


class ParametrizedParticleDataset(ParticlesDataset):
    def __init__(
        self,
        *,
        x: np.ndarray,
        theta: np.ndarray,
        score: Optional[np.ndarray] = None,
        ratio: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ):
        super().__init__(x=x, score=score)
        # allow for unllabeled data
        ratio = ratio if ratio is not None else np.zeros((len(x), 1))
        labels = labels if labels is not None else np.zeros((len(x), 1))

        assert len(x) == len(theta), "x and theta differ in length"

        self._thetas = theta
        self._ratios = ratio
        self._labels = labels

    def __getitem__(self, index) -> ParametrizedParticlesEvent:
        return ParametrizedParticlesEvent(
            fourmomenta=self._x[index],
            theta=self._thetas[index],
            length=int(self._lengths[index]),
            score=self._score[index],
            ratio=self._ratios[index],
            label=self._labels[index],
        )
