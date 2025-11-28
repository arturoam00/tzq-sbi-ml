from pathlib import Path

import numpy as np
import torch
from torch.autograd import grad

from ..limits import AsymptoticLimitsRatios
from .base_experiment_ml import BaseExperimentML
from .schemas import (
    ModelOutput,
    ParametrizedPredictionOutput,
    ParametrizedRawData,
    ParametrizedTargetOutput,
)


class BaseExperimentRatios(BaseExperimentML):
    asymptotics_cls = AsymptoticLimitsRatios

    def __init__(self, *args, **kwds) -> None:
        kwds["key"] = "ratios"
        super().__init__(*args, **kwds)

    def _preds(self, *args, **kwargs):
        pass

    def _load_raw_data(self, source):
        source = Path(source)
        x_test = np.load(source / "x_test.npy")
        ratio_train = np.load(source / "r_xz_train_ratio.npy")
        score_train = np.load(source / "t_xz_train_ratio.npy")
        labels_train = np.load(source / "y_train_ratio.npy")
        max_samples = self.cfg.train.get("clamp_samples", None)
        return ParametrizedRawData(
            x_train=np.load(source / "x_train_ratio.npy")[:max_samples],
            theta_train=np.load(source / "theta0_train_ratio.npy")[:max_samples],
            ratio_train=ratio_train[:max_samples],
            score_train=score_train[:max_samples],
            labels_train=labels_train[:max_samples],
            x_test=x_test,
            theta_test=np.load(source / "theta_test.npy"),
            # TODO: Augment test data (i just put dummy data now)!
            ratio_test=np.zeros((x_test.shape[0], ratio_train.shape[1])),
            score_test=np.zeros((x_test.shape[0], score_train.shape[1])),
            labels_test=np.zeros((x_test.shape[0], labels_train.shape[1])),
        )

    def _load_dataset(self, raw: ParametrizedRawData, mode="train"):
        if mode == "train":
            return self.dataset_cls(
                x=raw.x_train,
                theta=raw.theta_train,
                score=raw.score_train,
                ratio=raw.ratio_train,
                labels=raw.labels_train,
            )
        if mode == "test":
            return self.dataset_cls(
                x=raw.x_test,
                theta=raw.theta_test,
                score=raw.score_test,
                ratio=raw.ratio_test,
                labels=raw.labels_test,
            )
        raise ValueError(f"Invalid mode {mode}")

    # will i ever need the score for evaluation ?? I don't think so
    def _eval(self, output: ModelOutput):
        return output.pred.log_ratio

    def pack_output(self, theta, log_ratio_pred, score, ratio, label):
        score_pred = None
        if self.loss_fn.REQUIRES_SCORE:
            (score_pred,) = grad(
                log_ratio_pred,
                theta,
                grad_outputs=torch.ones_like(log_ratio_pred),
                only_inputs=True,
                create_graph=True,
            )
        return ModelOutput(
            pred=ParametrizedPredictionOutput(
                score=score_pred, log_ratio=log_ratio_pred
            ),
            target=ParametrizedTargetOutput(score=score, ratio=ratio, label=label),
        )
