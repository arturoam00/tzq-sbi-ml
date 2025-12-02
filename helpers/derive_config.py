from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def load_conf_from(path: Path, merge_on=None) -> DictConfig:
    # Find base
    base_path = path.parent / "_base.yaml"
    base = OmegaConf.load(base_path) if base_path.exists() else {}

    # Load main
    cfg = OmegaConf.merge(base, OmegaConf.load(path.with_suffix(".yaml")))
    if merge_on is not None:
        return OmegaConf.create({merge_on: cfg})
    return cfg


def derive_config(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, False)

    auto_dir = Path("conf/_auto")

    # Load limits
    cfg.merge_with(
        load_conf_from(auto_dir / "limits" / cfg.dataset.key, merge_on="limits")
    )

    # Load exp model
    model_key = cfg.model.key if cfg.model.key else "noop"
    cfg.merge_with(
        load_conf_from(auto_dir / "exp_model" / f"{cfg.exp.key}_{model_key}")
    )

    # Load dataset
    cfg.merge_with(load_conf_from(auto_dir / "dataset" / model_key, merge_on="dataset"))

    # Load (optional) loss
    loss_path = (auto_dir / "loss" / cfg.exp.key).with_suffix(".yaml")
    if loss_path.exists():
        cfg.merge_with(load_conf_from(loss_path, merge_on="loss"))

    OmegaConf.set_struct(cfg, True)

    return cfg
