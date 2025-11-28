from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import mplhep as mh
import numpy as np
import scipy.stats as stats

from ..logger import LOGGER as _LOGGER

if TYPE_CHECKING:
    from ..base.schemas import Limits

mh.style.use("ATLAS")

LOGGER = _LOGGER.getChild(__name__)


def plot_llr(
    limits_list,
    labels,
    colors=None,
    linestyles=None,
    conf_levels=(0.68, 0.95),
    to=None,
    mode: Literal["average", "slice"] = "average",
):
    """
    Plot -2 log-likelihood ratio contours for multiple Limits objects,
    each with multiple confidence regions (e.g. 68%, 95%).

    Parameters
    ----------
    limits_list : list
        List of Limits objects.
    labels : list of str
        List of labels (same length as limits_list).
    colors : list of str, optional
        Colors for each method.
    linestyles : list of str, optional
        Linestyles for each method.
    conf_levels : tuple of floats, optional
        Confidence levels to plot (as cumulative probabilities of χ²).
    to : Path or str, optional
        Path to save figure.
    mode : {"average", "slice"}, default "average"
        Projection mode for extra dimensions.
    """
    assert len(limits_list) == len(labels)
    N = limits_list[0].grid.shape[1]
    if N < 2:
        return _plot_llr_1d(
            limits_list=limits_list,
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            to=to,
        )
    resolutions = limits_list[0].resolutions
    assert all(a == resolutions[0] for a in resolutions)
    D = resolutions[0]

    pairs = list(combinations(range(N), 2))
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # Default styles
    if colors is None:
        colors = plt.cm.tab10.colors[: len(limits_list)]
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"] * ((len(limits_list) // 4) + 1)

    # Precompute χ² levels for each confidence probability (2D case)
    chi2_levels = [stats.chi2.ppf(p, 2) for p in conf_levels]

    # Build legend handles for all methods
    handles = [
        mlines.Line2D([], [], color=c, linestyle=ls, linewidth=2, label=lab)
        for c, ls, lab in zip(colors, linestyles, labels)
    ]

    for ax, (i, j) in zip(axes, pairs):
        for limits, color, ls in zip(limits_list, colors, linestyles):
            assert np.all(limits_list[0].grid == limits.grid)
            values_nd = -2 * limits.llr.reshape(limits.resolutions)
            others = [k for k in range(N) if k not in (i, j)]

            if mode == "average":
                data_2d = values_nd.mean(axis=tuple(others))
            elif mode == "slice":
                fixed_index = D // 2
                slicer = [slice(None)] * N
                for k in others:
                    slicer[k] = fixed_index
                data_2d = values_nd[tuple(slicer)]
            # TODO add mode 'mle'
            else:
                raise ValueError("mode must be 'average' or 'slice'")

            data_2d -= data_2d.min()
            xi = np.unique(limits.grid[:, i])
            yj = np.unique(limits.grid[:, j])
            X, Y = np.meshgrid(xi, yj, indexing="ij")

            # Plot multiple contours per method
            for lvl, alpha in zip(chi2_levels, np.linspace(1.0, 0.4, len(chi2_levels))):
                ax.contour(
                    X,
                    Y,
                    data_2d,
                    levels=[lvl],
                    colors=[color],
                    linestyles=[ls],
                    linewidths=1.75,
                    alpha=alpha,
                )

            imle, jmle = np.unravel_index(np.argmin(data_2d), data_2d.shape)
            ax.plot(X[imle, jmle], Y[imle, jmle], marker="*", color=color)

        ax.set_xlabel(limits.param_names[i])
        ax.set_ylabel(limits.param_names[j])
        ax.set_title(f"({limits.param_names[i]}, {limits.param_names[j]}) projection")

    # Shared legend on top
    fig.legend(handles=handles, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if to is not None:
        fig.savefig(
            Path(to).with_stem(f"{Path(to).stem}_" + "_".join(limits.param_names))
        )
    plt.close(fig)


def _plot_llr_1d(
    limits_list: List[Limits],
    labels,
    colors=None,
    linestyles=None,
    to=None,
    levels=(0.68, 0.95),
    ylim=(0, 30),
):
    """
    Plot -2 log-likelihood ratio curves for multiple 1D Limits objects.

    Parameters
    ----------
    limits_list : list
        List of 1D Limits objects.
    labels : list of str
        List of labels (same length as limits_list).
    colors : list of str, optional
        List of colors for each line.
    linestyles : list of str, optional
        List of linestyles for each line.
    to : Path or str, optional
        File path to save figure.
    levels: tuple, optional
        Asymptotic confidence intervals
    ylim : tuple, optional
        y-axis range (default = (0, 30)).
    """
    assert len(limits_list) == len(labels)

    # Styling defaults
    if colors is None:
        colors = plt.cm.tab10.colors[: len(limits_list)]
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":"] * ((len(limits_list) // 4) + 1)

    # All Limits should have one parameter
    param = limits_list[0].param_names[-1]
    x_range = limits_list[0].ranges[-1]

    fig, ax = plt.subplots()

    for limits, label, color, ls in zip(limits_list, labels, colors, linestyles):
        x = limits.grid[:, 0]
        y = -2 * limits.llr
        ax.plot(x, y, color=color, linestyle=ls, linewidth=2, label=label)

    # values = [stats.chi2.ppf(level, 1) for level in levels]
    for level in levels:
        value = stats.chi2.ppf(level, 1)
        ax.hlines(
            value,
            *x_range,
            colors="grey",
            linestyles="--",
            alpha=0.5,
        )
        ax.text(
            x=x_range[-1] - 0.25,
            y=value + 0.3,
            s=f"{int(level * 100)}% CI",
            color="grey",
        )
    ax.set_ylim(*ylim)
    ax.set_xlim(x_range)
    ax.set_xlabel(param)
    ax.set_ylabel(r"$-2\log\Lambda$")
    ax.legend(frameon=False)
    fig.tight_layout()

    if to is not None:
        fig.savefig(Path(to).with_stem(f"{Path(to).stem}_{param}"))
    plt.close(fig)

    return fig, ax


def plot_learning_curves(losses, to=None):
    epochs = np.arange(len(losses.train))
    fig, ax = plt.subplots()
    ax.plot(epochs, losses.train, label="train")
    ax.plot(epochs, losses.val, label="val")
    ax.legend()
    fig.tight_layout()

    if to is not None:
        fig.savefig(to)

    return fig, ax
