from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from .logger import LOGGER as _LOGGER
from .plotting import plot_llr

if TYPE_CHECKING:
    from omegaconf import DictConfig

LOGGER = _LOGGER.getChild(__name__)


class Ensemble:
    def __init__(self, params):
        self.params = OmegaConf.create(params)

    def run_ensemble(self, cfg: DictConfig):
        assert (
            not cfg.modes.eval
            and not cfg.modes.train
            and cfg.modes.recycle
            and not cfg.modes.plot
        )

        limits_all = list()
        labels = list()
        for dir, exp, model, dataset in zip(
            self.params.dirs,
            self.params.experiments.values(),
            self.params.models.values(),
            self.params.datasets,
            strict=True,
        ):
            # Handle no model
            if model is None:
                model = ""
            else:
                exp.model = model
            LOGGER.info(
                f"Running ensemble with experiment {str(exp)} and model {str(model)}"
            )
            cfg.data.dataset, cfg.data.run = dataset, dir
            exp(cfg=cfg)
            limits_all.append(exp.checkpoints.limits)
            labels.append(f"{str(exp)} {str(model)}")

        plot_llr(
            limits_all,
            colors=self.params.plotting.colors,
            linestyles=self.params.plotting.linestyles,
            labels=labels,
            to="fig.png",
        )

    def __call__(self, *args, **kwds):
        self.run_ensemble(*args, **kwds)
