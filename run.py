from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from derive_config import derive_config

if TYPE_CHECKING:
    from omegaconf import DictConfig


OmegaConf.register_new_resolver("sum", lambda x, y: x + y)


@hydra.main(config_name="config", config_path="conf", version_base=None)
def main(cfg: DictConfig):
    # Derive final configuration object
    cfg = derive_config(cfg)

    # Instantiate (partial) experiment, construct with cfg and call it
    instantiate(cfg.exp)(cfg=cfg)()


if __name__ == "__main__":
    main()
