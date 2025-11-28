from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..logger import LOGGER as _LOGGER
from ..plotting import plot_learning_curves, plot_llr
from ..utils import device, dtype
from .base_experiment import BaseExperiment
from .schemas import Losses

if TYPE_CHECKING:

    from ..losses import Loss
    from .schemas import Limits

LOGGER = _LOGGER.getChild(__name__)


@contextmanager
def needs_grad(flag):
    ctx = torch.enable_grad() if flag else torch.no_grad()
    with ctx:
        yield


class BaseExperimentML(BaseExperiment):
    # To be defined in subclasses
    collate_fn = None
    dataset_cls = None
    asymptotics_cls = None

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.model = None
        self.loss_fn: Optional[Loss] = None
        self.train_loader = self.val_loader = None
        self.train_dataset = self.val_dataset = None
        self.test_dataset = self.test_loader = None
        self.device_kwds = {
            "device": device(self.cfg.devices.device),
            "dtype": dtype(self.cfg.devices.dtype),
            "non_blocking": self.cfg.devices.non_blocking,
        }

    @abstractmethod
    def _load_raw_data(self, *args, **kwds):
        pass

    @abstractmethod
    def _load_dataset(self, raw, mode: Literal["train", "test"] = "train"):
        pass

    @abstractmethod
    def _preds(self, batch):
        pass

    @abstractmethod
    def _eval(self, output):
        pass

    def _init(self) -> None:
        self.init_loss_function()
        self.init_datasets()
        self.init_loaders()
        self.init_model()

    def _run(self) -> None:
        if self.cfg.modes.train:
            LOGGER.info(f"Starting traing of exp. {self}")
            self.checkpoints.state_dict, self.checkpoints.losses = self.train()
        if self.cfg.modes.eval:
            LOGGER.info(f"Starting evaluation of exp. {self}")
            self.checkpoints.limits = self.eval_lims()
        if self.cfg.modes.plot:
            LOGGER.info(f"Plotting results of experiment {self}")
            self.plot()

    def init_loss_function(self) -> None:
        self.loss_fn = instantiate(self.cfg.loss)

    def init_datasets(self) -> None:
        raw = self._load_raw_data(self.cfg.dataset.path)
        dataset = self._load_dataset(raw, "train")

        split = self.cfg.train.validation_split
        assert 0.0 < split < 0.5, f"Validation split {split} seems wrong"

        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [
                int(len(dataset) * (1 - split)),
                int(len(dataset) * split),
            ],
        )
        self.test_dataset = self._load_dataset(raw, "test")

    def init_loaders(self) -> None:
        factory_kwds = {
            "batch_size": self.cfg.train.batch_size,
            "pin_memory": self.cfg.devices.pin_memory,
            "shuffle": True,
            "collate_fn": self.collate_fn,
            "num_workers": 0,  # Make dynamic?
        }
        self.train_loader = DataLoader(self.train_dataset, **factory_kwds)
        self.val_loader = DataLoader(self.val_dataset, **factory_kwds)
        # Maybe change kwds?
        self.test_loader = DataLoader(self.test_dataset, **factory_kwds)

    def init_model(self) -> None:
        self.model = instantiate(self.cfg.model)
        if self.cfg.modes.recycle and self.checkpoints.state_dict is not None:
            LOGGER.info("Starting warm!")
            self.model.load_state_dict(self.checkpoints.state_dict)
        self.model.to(**self.device_kwds)
        LOGGER.info(f"Model moved to {str(self.device_kwds["device"])}")

    def loss(self, batch):
        return self.loss_fn(self._preds(batch))

    # TODO:
    # 1. Normalize input data (and labels? theta?)
    # 2. Fix lr scheduler
    # 3. Add eraly stopping with patience
    def train(self):

        train_losses, val_losses = [], []

        # Optimizer
        # opt = torch.optim.AdamW(params=self.model.parameters(), lr=2e-4, weight_decay=1e-4)
        opt = torch.optim.Adam(params=self.model.parameters(), lr=self.cfg.train.lr)

        # LR scheduler
        # FIXME first epoch lr is 0 ;(
        # lr_sch = torch.optim.lr_scheduler.LambdaLR(
        #     opt,
        #     lambda i: min(
        #         i / (self.cfg.train.lr_warmup / self.cfg.train.batch_size), 1.0
        #     ),
        # )
        lr_sch = None

        # lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.train.lr_warmup)
        # lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1)

        # TensorBoard
        writer = SummaryWriter(log_dir="runs/model_experiment")

        # Training loop
        global_step = 0

        for e in range(self.cfg.train.epochs):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {e+1}/{self.cfg.train.epochs}",
                leave=False,
            )
            for batch in pbar:
                # Zero grads
                opt.zero_grad()

                # Move data to device and cast to dtype
                batch.to(**self.device_kwds)

                # Calculate loss on output
                loss = self.loss(batch)

                # Backward pass
                loss.backward()

                # Optimizer step
                opt.step()

                # Gradient clipping (can be `null` or not appear in config)
                max_norm = self.cfg.train.get("clip_grad_norm", float("inf"))
                max_norm = max_norm if max_norm is not None else float("inf")
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_norm, error_if_nonfinite=False
                )

                train_loss += loss.item()
                global_step += 1

                # log batch loss to TensorBoard
                writer.add_scalar("Loss/batch", loss.item(), global_step)
                writer.add_scalar("LR", opt.param_groups[0]["lr"], global_step)

                # Update progress bar
                pbar.set_postfix(
                    {"batch_loss": loss.item(), "lr": opt.param_groups[0]["lr"]}
                )

            # Scheduler step
            if lr_sch is not None:
                lr_sch.step()

            avg_train_loss = train_loss / len(self.train_loader)
            print(
                f"Epoch {e+1}/{self.cfg.train.epochs} - train loss: {avg_train_loss:.4f}"
            )

            # log epoch loss to TensorBoard
            writer.add_scalar("Loss/train_epoch", avg_train_loss, e + 1)
            train_losses.append(avg_train_loss)

            # Validation
            # TODO: Evaluation probably better in CPU?
            self.model.eval()
            val_loss = 0.0

            # If the loss function requires computing the score as the gradient
            # of the output w.r.t the input parameters, we need also a graph for
            # the evaluation
            with needs_grad(self.loss_fn.REQUIRES_SCORE):
                for batch in self.val_loader:
                    batch.to(**self.device_kwds)
                    val_loss += self.loss(batch).item()

            avg_val_loss = val_loss / len(self.val_loader)
            print(f"Epoch {e+1}/{self.cfg.train.epochs} - val loss: {avg_val_loss:.4f}")
            writer.add_scalar("Loss/val_epoch", avg_val_loss, e + 1)
            val_losses.append(avg_val_loss)

        writer.close()

        return self.model.state_dict(), Losses(train=train_losses, val=val_losses)

    # Rn I cannot calculate test metrics bc test data is not augmented :/ (yet!)
    # I can calculate limits on parameter grid though
    # TODO: What if evaluation needs to compute gradients w.r.t. theta?
    # (for llr regressors).. we couldn't be in inference mode ...
    @torch.inference_mode()
    def eval(self, loader) -> np.ndarray:
        self.model.eval()
        device_kwds = self.device_kwds.copy()
        device_kwds["device"] = device(self.cfg.devices.eval)
        self.model.to(**device_kwds)
        LOGGER.info(f"Model moved to {device_kwds["device"]}")

        preds = []

        LOGGER.info(
            f"Starting to evaluate {len(loader)} minibatches of size {loader.batch_size}"
        )
        for batch in tqdm(loader):
            batch.to(**device_kwds)
            output = self._preds(batch)
            preds.append(
                self._eval(output).detach()
            )  # i keep detach for the possible future...

        return torch.cat([p.cpu() for p in preds]).numpy()

    def eval_lims(self) -> Limits:
        events_file = self.cfg.dataset.events_file
        alims = self.asymptotics_cls(events_file)

        assert (
            len(self.cfg.limits.theta_ranges)
            == len(self.cfg.limits.resolutions)
            == self.cfg.dataset.theta_dim
        ), f"Shape mismatch for dim(theta)={self.cfg.dataset.theta_dim}"

        # Create theta grid
        theta_grid = alims.theta_grid(
            theta_ranges=self.cfg.limits.theta_ranges,
            resolutions=self.cfg.limits.resolutions,
        )

        # Load test data

        # If asimov is available, sample asimov
        if self.cfg.limits.get("asimov"):
            LOGGER.info(
                f"Starting sampling Asimov dataset from test partition of {events_file}"
            )

            assert (
                len(self.cfg.limits.asimov.theta_true) == self.cfg.dataset.theta_dim
            ), "Shape mismatch"

            x_test, weights_test = alims.asimov_data(
                self.cfg.limits.asimov.theta_true,
                self.cfg.limits.asimov.sample_only_from_closest_benchmark,
                self.cfg.limits.test_split,
                self.cfg.limits.asimov.n_asimov,
            )
            n_events = (
                self.cfg.limits.luminosity
                # NOTE: first parameter needs to be a list of arrays or list of lists
                * alims.calculate_xsecs(
                    [self.cfg.limits.asimov.theta_true], self.cfg.limits.test_split
                )[0]
            )

            LOGGER.info(f"Sampled {len(x_test)} events")

            test_loaders = self.create_lims_loaders(
                x=x_test,
                theta=theta_grid if not self.asymptotics_cls.NEEDS_HISTOS else None,
            )

        # If asimov is not available use mc toy
        else:
            weights_test = np.ones(len(self.test_dataset))
            n_events = len(self.test_dataset)
            test_loaders = [self.test_loader]

        LOGGER.info("Evaluating test data ...")
        preds = [self.eval(tl) for tl in test_loaders]

        # Histos if needed
        histos = None
        if self.asymptotics_cls.NEEDS_HISTOS:
            LOGGER.info(f"Sampling train samples from train partition of {events_file}")
            x_train, weights_train, _ = alims.weighted_events_from_partition(
                n_draws=self.cfg.limits.n_toys,
                partition="train",
                test_split=self.cfg.limits.test_split,
                thetas=theta_grid,
            )
            LOGGER.info(f"Sampled {len(x_train)} train events")
            train_dataset = self.dataset_cls(x=x_train, theta=theta_grid)
            train_loader = DataLoader(
                train_dataset, batch_size=100, collate_fn=self.collate_fn
            )
            LOGGER.info("Evaluating train data for histograms creation ...")
            preds_train = self.eval(train_loader)
            histos = alims.histos(preds_train, weights_train)

        return alims.limits(
            predictions=preds,
            n_events=n_events,
            x_weights=weights_test,
            theta_grid=theta_grid,
            luminosity=self.cfg.limits.luminosity,
            test_split=self.cfg.limits.test_split,
            histos=histos,
        )

    def create_lims_loaders(self, x, theta=None):
        factory_kwds = {"batch_size": 100, "collate_fn": self.collate_fn}
        if theta is None:
            return [DataLoader(self.dataset_cls(x=x), **factory_kwds)]
        return [
            DataLoader(
                self.dataset_cls(
                    x=x,
                    # `copy()` is needed to create a writable array, not only a view
                    theta=np.broadcast_to(t, (x.shape[0], t.shape[-1])).copy(),
                ),
                **factory_kwds,
            )
            for t in theta
        ]

    def plot_llr(self, to=None):
        return plot_llr(
            [self.checkpoints.limits], labels=[f"{self.model} {self}"], to=to
        )

    def plot_learning_curves(self, to=None):
        return plot_learning_curves(self.checkpoints.losses, to=to)

    def plot(self) -> None:
        self.plot_llr(Path(self.cfg.data.run_dir) / "llr.png")
        self.plot_learning_curves(Path(self.cfg.data.run_dir) / "learning_curve.png")
