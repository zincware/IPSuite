import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interpn
from scipy.optimize import curve_fit
from scipy.stats import foldnorm, gaussian_kde


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
    true,
    prediction,
    datalabel: str,
    xlabel: str,
    ylabel: str,
    ymax: typing.Optional[float] = None,
    figsize: tuple = (10, 7),
    density=True,
) -> plt.Figure:
    """Create a correlation plot for true, prediction values.

    Parameters
    ----------
    true: the true values
    prediction: the predicted values
    datalabel: str, the label for the prediction, e.g. 'MAE: 0.123 eV'
    xlabel: str, the xlabel
    ylabel: str, the xlabel
    figsize: tuple, size of the figure

    Returns
    -------
    plt.Figure

    """
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(true, np.zeros_like(true), color="grey", zorder=0)
    bins = 25
    if true.shape[0] < 20 or not density:
        # don't use density for very small datasets
        ax.scatter(true, prediction, marker="x", s=20.0, label=datalabel)
    else:
        density_scatter(
            ax, true, prediction, bins=bins, marker="x", s=20.0, label=datalabel
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim([-ymax, ymax])
    ax.legend()
    return fig


def get_calibration_figure(
    error,
    std,
    markersize: float = 3.0,
    datalabel=None,
    forces=False,
    figsize: tuple = (10, 7),
):
    """Log-log plot of errors vs predicted standard deviations with quantiles
    for a linearly increasing noise level.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)

    x = np.linspace(1e-6, 5e3, 5)
    noise_level_2 = x

    quantiles_lower_01 = [foldnorm.ppf(0.15, 0.0, 0.0, i) for i in noise_level_2]
    quantiles_upper_01 = [foldnorm.ppf(0.85, 0.0, 0.0, i) for i in noise_level_2]
    quantiles_lower_05 = [foldnorm.ppf(0.05, 0.0, 0.0, i) for i in noise_level_2]
    quantiles_upper_05 = [foldnorm.ppf(0.95, 0.0, 0.0, i) for i in noise_level_2]
    quantiles_lower_005 = [foldnorm.ppf(0.005, 0.0, 0.0, i) for i in noise_level_2]
    quantiles_upper_005 = [foldnorm.ppf(0.995, 0.0, 0.0, i) for i in noise_level_2]

    ax.scatter(
        std,
        error,
        s=markersize,
        alpha=0.3,
        color="tab:blue",
        rasterized=True,
        linewidth=0.0,
        label=datalabel,
    )
    ax.loglog()
    ax.plot(x, quantiles_upper_05, color="gray", alpha=0.5)
    ax.plot(x, quantiles_lower_05, color="gray", alpha=0.5)
    ax.plot(x, quantiles_upper_01, color="gray", alpha=0.5)
    ax.plot(x, quantiles_lower_01, color="gray", alpha=0.5)
    ax.plot(x, quantiles_upper_005, color="gray", alpha=0.5)
    ax.plot(x, quantiles_lower_005, color="gray", alpha=0.5)

    ax.plot(
        np.logspace(-10, 100.0), np.logspace(-10, 100.0), linestyle="--", color="grey"
    )
    ax.set_xlim(np.min(std) / 1.5, np.max(std) * 1.5)
    ax.set_ylim(np.min(error) / 1.5, np.max(error) * 1.5)

    if forces:
        xlabel = r"$\sigma_{f_{i\alpha}}(A)$ [meV/$\AA$] "
        ylabel = r"$|\Delta f_{i\alpha}(A)|$ [meV/$\AA$] "
    else:
        xlabel = r"$\sigma_{E_{i}}(A)$ [meV/atom] "
        ylabel = r"$|\Delta E_{i}(A)|$ [meV/atom] "

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if datalabel:
        ax.legend()
    return fig


def gauss(x, *p):
    m, s = p
    return np.exp(-(((x - m) / s) ** 2) * 0.5) / np.sqrt(2 * np.pi * s**2)


def slice_ensemble_uncertainty(true, pred_ens, slice_start, slice_end):
    pred_mean = np.mean(pred_ens, axis=1)
    pred_std = np.std(pred_ens, axis=1)

    isel = np.where((slice_start < pred_std) & (pred_std < slice_end))[0]

    error_true = np.reshape(true[isel] - pred_mean[isel], -1)
    error_pred = np.reshape(pred_ens[isel, :] - pred_mean[isel, np.newaxis], -1)
    return error_true, error_pred


def slice_uncertainty(true, pred_mean, pred_std, slice_start, slice_end):
    isel = np.where((slice_start < pred_std) & (pred_std < slice_end))[0]

    error_true = np.reshape(true[isel] - pred_mean[isel], -1)
    error_pred = pred_std[isel]
    return error_true, error_pred


def get_gaussianicity_figure(error_true, error_pred, forces=True):
    """Plots empirical and predicted error distributions.
    If possible, it also tries to fit a gaussian to the empirical distribution.
    """
    true_kde_sel = gaussian_kde(error_true)
    ens_kde_sel = gaussian_kde(error_pred)

    bounds = 1.5 * max(np.max(np.abs(error_true)), np.max(np.abs(error_pred)))

    fig, ax = plt.subplots()

    xgrid = np.linspace(-bounds, bounds, 400)
    ax.set_xlim([-bounds, bounds])

    ens_sel = ens_kde_sel(xgrid)
    true_sel = true_kde_sel(xgrid)

    try:
        guess = [0.0, 100]
        coeff, _ = curve_fit(gauss, xgrid, true_sel, p0=guess)
        std = coeff[1]
        ax.semilogy(xgrid, gauss(xgrid, 0, std), "k--", label="Gaussian")

    except RuntimeError:
        print("Curve fit failed, only plotting distributions")

    ax.semilogy(xgrid, true_sel, "r-", label="empirical")
    ax.semilogy(xgrid, ens_sel, "b-", label="predicted")
    ymax = 5 * max(np.max(true_sel), np.max(ens_sel))
    ax.set_ylim(1e-6, ymax)
    ax.set_yscale("log")

    if forces:
        xlabel = r"$\Delta (S)$ / meV/Ang"
        ylabel = r"$p(\Delta | S)$"
    else:
        xlabel = r"$\Delta (S)$ / meV/atom"
        ylabel = r"$p(\Delta | S)$"

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
