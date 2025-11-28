from typing import Literal

import torch
from lgatr import embed_scalar, embed_vector

# NOTE: I'm not using the `scalars` possibility at all yet


def to_multivector(fourmomenta: torch.Tensor):
    """
    fourmomenta: (nparticles, 4)
    """
    fourmomenta = fourmomenta.unsqueeze(-2)  # (nparticles, 1, 4)
    mv = embed_vector(fourmomenta)  # (nparticles, 1, 16)
    return mv.unsqueeze(0)  # (1, nparticles, 1, 16)


def to_multivector_parametrized(
    fourmomenta: torch.Tensor,
    theta: torch.Tensor,
    ptr: torch.Tensor,
    mode: Literal["tokens", "channels"],
):
    """
    Takes as input fourmomenta, theta and ptr corresponding to a batch of
    particles. the ptr dived the batch into events

    fourmomenta: (nparticles, 4)
    theta: (batch_size, theta_dim)
    ptr: (batch_size + 1,)

    Returns:
        multivectors (torch.Tensor): to feed the LGATr
        theta_multivectors (torch.Tensor): used to create `multivectors` with concatenation
    """
    # TODO: Embed paramters either as:
    #   1. Extra multivector channels (one for each dimension of the parameter vector)
    #   this is repeated across the "particles dimension", so each particle gets an
    #   associated set of parameter multivectors (DIRTIER, EASIER)
    #   2. Extra (global) tokens (one for each dimension of the parameter vector)
    #   This are prepended as global tokens for each event (CLEANER, HARDER)
    mvs = to_multivector(fourmomenta)  # (1, nparticles, 1, 16)

    # Make sure `ptr` is of integer type
    ptr = ptr.to(dtype=torch.long)

    b, n, c, _ = mvs.shape
    if mode == "channels":
        theta_dim = theta.shape[1]
        theta = theta.repeat_interleave(
            ptr[1:] - ptr[:-1], dim=0
        )  # (nparticles, theta_dim)

        assert theta.size() == (n, theta_dim)

        theta_mvs = embed_scalar(theta.unsqueeze(-1))  # (nparticles, theta_dim, 16)
        theta_mvs = theta_mvs.unsqueeze(0)  # (1, nparticles, theta_dim, 16)
        multivectors = torch.cat((mvs, theta_mvs), dim=-2)

        return multivectors  # (batch, particles, theta_dim + 1, 16)

    elif mode == "tokens":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid mode {mode}")
