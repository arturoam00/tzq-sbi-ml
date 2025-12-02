from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from helpers.derive_config import derive_config

if TYPE_CHECKING:
    from omegaconf import DictConfig


OmegaConf.register_new_resolver(
    "env",
    lambda key: {"prefix": Path(sys.executable).parent, "cwd": os.getcwd()}.get(key),
)


@hydra.main(config_name="config", config_path="conf", version_base=None)
def main(cfg: DictConfig):
    cfg = derive_config(cfg)
    instantiate(cfg.launcher)(cfg=cfg)


if __name__ == "__main__":
    main()
