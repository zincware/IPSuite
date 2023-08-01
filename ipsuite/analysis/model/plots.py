import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interpn


def density_scatter(ax, x, y, bins, **kwargs) -> None:
    """Create a scatter plot colored by 2d histogram density.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x: np.ndarray
    y: np.ndarray
    bins: int
    kwargs
        any kwargs passed to 'ax.scatter'

    Returns
    -------

    References
    ----------
    Adapted from https://stackoverflow.com/a/53865762/10504481

    """
    # convert e.g. DataFrame to numpy array values
    x = np.array(x)
    y = np.array(y)

    if "cmap" not in kwargs:
        kwargs["cmap"] = "viridis"

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    points = (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1]))
    xi = np.vstack([x, y]).T
    z = interpn(points, data, xi, method="splinef2d", bounds_error=False)
    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, **kwargs)


def get_figure(
    true, prediction, datalabel: str, xlabel: str, ylabel: str, figsize: tuple = (10, 7)
) -> plt.Figure:
    """Create a correlation plot for true, prediction values.

    Parameters
    ----------
    true: the true values
    prediction: the predicted values
    datalabel: str, the label for the prediction, e.g. 'MAE: 0.123 meV'
    xlabel: str, the xlabel
    ylabel: str, the xlabel
    figsize: tuple, size of the figure

    Returns
    -------
    plt.Figure

    """
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(true, true, color="grey", zorder=0)  # plot the diagonal in the background
    bins = 500 if (len(true) / 10) > 500 else int(len(true) * 0.1)
    if bins < 20:
        # don't use density for very small datasets
        ax.scatter(true, prediction, marker="x", s=20.0, label=datalabel)
    else:
        density_scatter(
            ax, true, prediction, bins=bins, marker="x", s=20.0, label=datalabel
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig


def get_hist(data, label, xlabel, ylabel) -> typing.Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()

    sns.histplot(
        data,
        ax=ax,
        stat="percent",
        label=label,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return fig, ax


def get_histogram_figure(
    bin_edges,
    counts,
    datalabel: str,
    xlabel: str,
    ylabel: str,
    x_lim: typing.Tuple[float, float] = None,
    y_lim: typing.Tuple[float, float] = None,
    logy_scale=True,
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """Creates a Matplotlib figure based on precomputed bin edges and counts.

    Parameters
    ----------
    bin_edges: np.array
        Edges of the histogram bins.
    counts: np.array
        Number of occurrences in each bin.
    datalabel: str
        Labels for the figure legend.
    xlabel: str
        X-axis label.
    ylabel: str
        Y-axis label.
    x_lim: tuple
        X-axis limits.
    y_lim: tuple
        Y-axis limits.
    figsize: tuple
        Size of the Matplotlib figure
    """
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)

    ax.stairs(counts, bin_edges, label=datalabel, fill=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if logy_scale:
        ax.set_yscale("log")
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return fig
