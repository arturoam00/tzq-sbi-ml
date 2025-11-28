from pathlib import Path

import numpy as np

from ..limits import AsymptoticLimitsHistos
from ..logger import LOGGER as _LOGGER
from .base_experiment_ml import BaseExperimentML
from .schemas import ModelOutput, PredictionOutput, RawData, TargetOutput

LOGGER = _LOGGER.getChild(__name__)


class BaseExperimentLocal(BaseExperimentML):
    asymptotics_cls = AsymptoticLimitsHistos

    def __init__(self, *args, **kwds) -> None:
        kwds["key"] = "local"
        super().__init__(*args, **kwds)

    def _preds(self, *args, **kwds):
        pass

    def _load_raw_data(self, source):
        source = Path(source)
        x_test = np.load(source / "x_test.npy")
        max_samples = self.cfg.train.get("clamp_samples", None)
        dummy_scores = np.zeros((len(x_test), self.cfg.dataset.theta_dim))
        return RawData(
            x_train=np.load(source / "x_train_score.npy")[:max_samples],
            score_train=np.load(source / "t_xz_train_score.npy")[:max_samples],
            x_test=x_test,
            # TODO: Fix below! (add augmented data to test data)
            score_test=dummy_scores,
        )

    def _load_dataset(self, raw: RawData, mode="train"):
        if mode == "train":
            return self.dataset_cls(x=raw.x_train, score=raw.score_train)
        elif mode == "test":
            return self.dataset_cls(x=raw.x_test, score=raw.score_test)
        raise ValueError(f"Invalid mode {mode}")

    def _eval(self, output: ModelOutput):
        return output.pred.score

    def pack_output(self, score_pred, score):
        return ModelOutput(PredictionOutput(score=score_pred), TargetOutput(score))
