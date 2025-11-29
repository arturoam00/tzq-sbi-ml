from dataclasses import fields

import torch

DEVICES = {"gpu": torch.device("cuda"), "cpu": torch.device("cpu")}
DTYPES = {"float32": torch.float32, "float16": torch.float16}


def device(key):
    return DEVICES[key]


def dtype(key):
    return DTYPES[key]


def to_device(*args: torch.Tensor, device: torch.device, **kwds) -> list[torch.Tensor]:
    return [elem.to(device, **kwds) for elem in args]


def to_fields(dcls, **kwargs):
    for f in fields(dcls):
        if f.type is torch.Tensor:
            setattr(dcls, f.name, getattr(dcls, f.name).to(**kwargs))
