from hydra.utils import instantiate


def submit(*, worker, cfg):
    """Simply pass the configuration object to the worker
    and run locally"""
    worker(cfg=cfg)
