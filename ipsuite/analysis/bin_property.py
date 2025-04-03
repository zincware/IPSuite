import pathlib

import numpy as np
import pandas as pd
import zntrack

from ipsuite import base
from ipsuite.analysis.model.math import decompose_stress_tensor
from ipsuite.analysis.model.plots import get_histogram_figure


class LabelHistogram(base.AnalyseAtoms):
    """Base class for creating histogram of a dataset.

    Parameters
    ----------
    data: list
        List of Atoms objects.
    bins: int
        Number of bins in the histogram.
    """

    bins: int = zntrack.params(None)
    x_lim: tuple = zntrack.params(None)
    y_lim: tuple = zntrack.params(None)
    plots_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plots")
    labels_df: pd.DataFrame = zntrack.plots()
    logy_scale: bool = zntrack.params(True)

    metrics: float = zntrack.metrics()

    def get_labels(self):
        raise NotImplementedError

    def get_hist(self):
        """Create a pandas dataframe from the given data."""
        labels = self.get_labels()

        self.metrics = {
            "mean": np.mean(labels),
            "std": np.std(labels),
            "max": np.max(labels),
            "min": np.min(labels),
        }

        if self.bins is None:
            self.bins = int(np.ceil(len(labels) / 100))
        counts, bin_edges = np.histogram(labels, self.bins)
        return counts, bin_edges

    def get_plots(self, counts, bin_edges):
        """Create figures for all available data."""
        self.plots_dir.mkdir(exist_ok=True)
        ylabel = "Occurrences"

        label_hist = get_histogram_figure(
            bin_edges,
            counts,
            datalabel=self.datalabel,
            xlabel=self.xlabel,
            ylabel=ylabel,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            logy_scale=self.logy_scale,
        )
        label_hist.savefig(self.plots_dir / "hist.png")

    def run(self):
        counts, bin_edges = self.get_hist()
        self.get_plots(counts, bin_edges)

        self.labels_df = pd.DataFrame({"bin_edges": bin_edges[1:], "counts": counts})


class EnergyHistogram(LabelHistogram):
    """Creates a histogram of all energy labels contained in a dataset."""

    datalabel: str = zntrack.params("energy")
    xlabel: str = zntrack.params(r"$E$ / eV")

    def get_labels(self):
        return [x.get_potential_energy() for x in self.data]


class ForcesHistogram(LabelHistogram):
    """Creates a histogram of all force labels contained in a dataset."""

    datalabel: str = zntrack.params("forces")
    xlabel: str = zntrack.params(r"$F$ / eV/Ang")

    def get_labels(self):
        labels = np.concatenate([x.get_forces() for x in self.data], axis=0)
        # compute magnitude of vector labels. Histogram works element wise for N-D Arrays
        labels = np.linalg.norm(labels, ord=2, axis=1)
        return labels


class ForcesUncertaintyHistogram(LabelHistogram):
    """Creates a histogram of all force uncertainties in a prediction."""

    datalabel: str = zntrack.params("forces-uncertainty")
    xlabel: str = zntrack.params(r"$\sigma(F)$ / eV/Ang")

    def get_labels(self):
        labels = np.concatenate(
            [x.calc.results["forces_uncertainty"] for x in self.data], axis=0
        )
        labels = np.linalg.norm(labels, ord=2, axis=1)
        return labels


class EnergyUncertaintyHistogram(LabelHistogram):
    """Creates a histogram of all energy uncertainties in a prediction."""

    datalabel: str = zntrack.params("energy-uncertainty")
    xlabel: str = zntrack.params(r"$\sigma(E)$ / eV")

    def get_labels(self):
        return np.reshape([x.calc.results["energy_uncertainty"] for x in self.data], (-1))


class DipoleHistogram(LabelHistogram):
    """Creates a histogram of all dipole labels contained in a dataset."""

    datalabel: str = zntrack.params("dipole")
    xlabel: str = zntrack.params(r"$\mu$ / eV Ang")

    def get_labels(self):
        labels = np.array([x.calc.results["dipole"] for x in self.data])
        # compute magnitude of vector labels. Histogram works element wise for N-D Arrays
        labels = np.linalg.norm(labels, ord=2, axis=1)
        return labels


class StressHistogram(base.AnalyseAtoms):
    """Creates histograms for the hydrostatic and
    deviatoric components of the stress tensor.

    Parameters
    ----------
    data: list
        List of Atoms objects.
    bins: int
        Number of bins in the histogram.
    """

    bins: int = zntrack.params(None)
    plots_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plots")
    labels_df: pd.DataFrame = zntrack.plots()
    logy_scale: bool = zntrack.params(True)

    def get_labels(self):
        labels = np.array([x.get_stress(voigt=False) for x in self.data])
        return labels

    def get_hist(self):
        """Create a pandas dataframe from the given data."""
        labels = self.get_labels()
        hydrostatic_stresses, deviatoric_stresses = decompose_stress_tensor(labels)

        if self.bins is None:
            self.bins = int(np.ceil(len(labels) / 100))
        hydro_counts, hydro_bin_edges = np.histogram(hydrostatic_stresses, self.bins)
        devia_counts, devia_bin_edges = np.histogram(deviatoric_stresses, self.bins)
        counts = (hydro_counts, devia_counts)
        bin_edges = (hydro_bin_edges, devia_bin_edges)
        return counts, bin_edges

    def get_plots(self, counts, bin_edges, hydrostatic=True):
        """Create figures for all available data."""
        if hydrostatic:
            xlabel = r"$\pi$ / eV / Ang$^3$"
            datalabel = "hydrostatic stress"
            fname = "hydrostatic_hist.png"
        else:
            xlabel = r"$\sigma_{ij}$ / eV / Ang$^3$"
            datalabel = "deviatoric stress components"
            fname = "deviatoric_hist.png"

        label_hist = get_histogram_figure(
            bin_edges,
            counts,
            datalabel=datalabel,
            xlabel=xlabel,
            ylabel="Occurrences",
            logy_scale=self.logy_scale,
        )
        label_hist.savefig(self.plots_dir / fname)

    def run(self):
        counts, bin_edges = self.get_hist()
        self.plots_dir.mkdir()
        self.get_plots(counts[0], bin_edges[0], hydrostatic=True)
        self.get_plots(counts[1], bin_edges[1], hydrostatic=False)

        self.labels_df = pd.DataFrame(
            {
                "hydro_bin_edges": bin_edges[0][1:],
                "hydro_counts": counts[0],
                "deviatoric_bin_edges": bin_edges[1][1:],
                "deviatoric_counts": counts[1],
            }
        )
