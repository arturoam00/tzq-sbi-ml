import torch
from torch.nn.attention import SDPBackend


def ptr2index(ptr: torch.Tensor) -> torch.Tensor:
    # This is a list of repeated indices. Indices represent events
    # and they are repeated as many times as the number of particles in
    # the event
    ptr = ptr.to(dtype=torch.long)
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1]
    )


def att_mask(index: torch.Tensor) -> torch.Tensor:
    # Thanks to broadcastig the following suffices
    # The next crates a block diagonal matrix of size (num_particles, num_particles)
    # there's room for a more efficient data storage here...
    return (index.unsqueeze(0) == index.unsqueeze(1)).to(index.device)


def get_backends(force_math=False):
    return [SDPBackend.MATH] + (
        [
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
        ]
        if not force_math
        else []
    )
