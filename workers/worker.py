from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from hydra.utils import instantiate
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def run(cfg: DictConfig):
    print(
        f"exp={cfg.exp.key} model={cfg.model.key} dataset={cfg.dataset.key} run={cfg.data.run}"
    )
    instantiate(cfg.exp)(cfg=cfg)()


if __name__ == "__main__":
    run(OmegaConf.load(sys.argv[1]))
