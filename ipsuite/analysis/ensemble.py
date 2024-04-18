import typing

import ase.calculators.singlepoint
import matplotlib.pyplot as plt
import numpy as np
import zntrack

from ipsuite import base, utils


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


class ModelEnsembleAnalysis(base.AnalyseAtoms):
    """Attributes
    ----------
        models: list of models to ensemble
        data: list of ASE Atoms objects to evaluate against.
    """

    models: list = zntrack.deps()

    normal_plot_path = zntrack.dvc.outs(zntrack.nwd / "normal_plot.png")
    sorted_plot_path = zntrack.dvc.outs(zntrack.nwd / "sorted_plot.png")
    histogram = zntrack.dvc.outs(zntrack.nwd / "histogram.png")

    prediction_list = zntrack.zn.outs()
    predictions: typing.List[ase.Atoms] = zntrack.zn.outs()

    bins: int = zntrack.zn.params(100)

    def run(self):
        # TODO axis labels
        # TODO save indices
        # TODo rewrite this Node based on MLModel and then add a Analysis Node
        self.prediction_list = [model.predict(self.data[:]) for model in self.models]

        self.predictions = []
        for idx, atom in enumerate(self.data):
            atom.calc = ase.calculators.singlepoint.SinglePointCalculator(
                atoms=atom,
                energy=np.mean(
                    [p[idx].get_potential_energy() for p in self.prediction_list]
                ),
                forces=np.mean(
                    [p[idx].get_forces() for p in self.prediction_list], axis=0
                ),
            )
            self.predictions.append(atom)

        figures = self.get_plots()
        figures[0][0].savefig(self.normal_plot_path)
        figures[1][0].savefig(self.sorted_plot_path)
        figures[2].savefig(self.histogram)

    def get_plots(self):
        energy = np.stack(
            [
                np.stack([x.get_potential_energy() for x in p])
                for p in self.prediction_list
            ]
        )

        figures = []
        # Plot the energy
        figures.append(
            plot_with_uncertainty(
                energy, figsize=(10, 5), xlabel="data point", ylabel="Energy / a.u."
            )
        )
        figures.append(
            plot_with_uncertainty(
                energy[:, np.argsort(np.std(energy, axis=0))[::-1]],
                figsize=(10, 5),
                xlabel="sorted data point",
                ylabel="Energy / a.u.",
            )
        )

        # Plot the histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(np.std(energy, axis=0), bins=self.bins)
        ax.set_xlabel("Energy standard deviation histogram")
        figures.append(fig)
        return figures
