import matplotlib.pyplot as plt
import numpy as np


def plot_with_uncertainty(value, ylabel: str, xlabel: str, x=None, **kwargs) -> dict:
    """Parameters
    ----------
    value: data of shape (n, m) where n is the number of ensembles.
    x: optional x values of shape (m,)

    Returns
    -------

    """
    if isinstance(value, dict):
        data = value
    else:
        data = {
            "mean": np.mean(value, axis=0),
            "std": np.std(value, axis=0),
            "max": np.max(value, axis=0),
            "min": np.min(value, axis=0),
        }

    fig, ax = plt.subplots(**kwargs)
    if x is None:
        x = np.arange(len(data["mean"]))
    ax.fill_between(
        x,
        data["mean"] + data["std"],
        data["mean"] - data["std"],
        facecolor="lightblue",
    )
    if "max" in data:
        ax.plot(x, data["max"], linestyle="--", color="darkcyan")
    if "min" in data:
        ax.plot(x, data["min"], linestyle="--", color="darkcyan")
    ax.plot(x, data["mean"], color="black")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    return fig, ax, data
