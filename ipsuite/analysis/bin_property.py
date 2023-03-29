import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import zntrack

from ipsuite import base
from ipsuite.utils.helpers import get_deps_if_node


def get_histogram_figure(
    bin_edges,
    counts,
    datalabel: str,
    xlabel: str,
    ylabel: str,
    logy_scale=True,
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """Creates a Matplotlib figure based on precomputed bin edges and counts.

    Parameters
    ----------
    bin_edges: np.array
        Edges of the histogram bins.
    counts: np.array
        Number of occurences in each bin.
    datalabel: str
        Labels for the figure legend.
    xlabel: str
        X-axis label.
    ylabel: str
        Y-axis label.
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
    return fig


class LabelHistogram(base.AnalyseAtoms):
    """Base class for creating histogram of a dataset.

    Parameters
    ----------
    data: list
        List of Atoms objects.
    bins: int
        Number of bins in the histogram.
    """

    bins: int = zntrack.zn.params(None)
    plots_dir: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "plots")
    labels_df: pd.DataFrame = zntrack.zn.plots()
    datalabel: str = None
    xlabel: str = None
    ylabel: str = "Occurences"
    logy_scale: bool = True

    def _post_init_(self):
        """Load metrics - if available."""
        self.data = get_deps_if_node(self.data, "atoms")

    def get_labels(self):
        raise NotImplementedError

    def get_hist(self):
        """Create a pandas dataframe from the given data."""
        labels = self.get_labels()
        if self.bins is None:
            self.bins = int(np.ceil(len(labels) / 100))
        counts, bin_edges = np.histogram(labels, self.bins)
        return counts, bin_edges

    def get_plots(self, counts, bin_edges):
        """Create figures for all available data."""
        self.plots_dir.mkdir()

        label_hist = get_histogram_figure(
            bin_edges,
            counts,
            datalabel=self.datalabel,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            logy_scale=self.logy_scale,
        )
        label_hist.savefig(self.plots_dir / "hist.png")

    def run(self):
        counts, bin_edges = self.get_hist()
        self.get_plots(counts, bin_edges)

        self.labels_df = pd.DataFrame({"bin_edges": bin_edges[1:], "counts": counts})


class EnergyHistogram(LabelHistogram):
    """Creates a histogram of all energy labels contained in a dataset."""

    datalabel = "energy"
    xlabel = r"$E$ / eV"

    def get_labels(self):
        return [x.get_potential_energy() for x in self.data]


class ForcesHistogram(LabelHistogram):
    """Creates a histogram of all force labels contained in a dataset."""

    datalabel = "energy"
    xlabel = r"$F$ / eV/Ang"

    def get_labels(self):
        labels = np.concatenate([x.get_forces() for x in self.data], axis=0)
        # compute magnitude of vector labels. Histogram works element wise for N-D Arrays
        labels = np.linalg.norm(labels, ord=2, axis=1)
        return labels


class DipoleHistogram(LabelHistogram):
    """Creates a histogram of all dipole labels contained in a dataset."""

    datalabel = "dipole"
    xlabel = r"$\mu$ / eV Ang"

    def get_labels(self):
        labels = np.array([x.calc.results["dipole"] for x in self.data])
        # compute magnitude of vector labels. Histogram works element wise for N-D Arrays
        labels = np.linalg.norm(labels, ord=2, axis=1)
        return labels
