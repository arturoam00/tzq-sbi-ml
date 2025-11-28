from __future__ import annotations

from typing import TYPE_CHECKING

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def auto_compose(cfg: DictConfig) -> DictConfig:
    GlobalHydra.instance().clear()
    model_key = cfg.model.key if cfg.model.key else "noop"
    overrides = [
        f"limits={cfg.dataset.key}",
        f"exp_model@_global_={cfg.exp.key}_{model_key}",
        f"dataset={model_key}",
        f"loss={cfg.exp.key}",
    ]
    with initialize(config_path="conf/_auto", version_base=None):
        return compose(config_name="config", overrides=overrides)


def derive_config(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, False)
    cfg.merge_with(auto_compose(cfg))
    OmegaConf.set_struct(cfg, True)
    return cfg
