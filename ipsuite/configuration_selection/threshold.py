"""Selecting atoms with a given step between them."""
import typing

import ase
import matplotlib.pyplot as plt
import numpy as np
import zntrack

from ipsuite.analysis.ensemble import plot_with_uncertainty
from ipsuite.configuration_selection import ConfigurationSelection


def mean_reduction(values, axis):
    return np.mean(values, axis=axis)


def max_reduction(values, axis):
    return np.max(values, axis=axis)


def check_dimension(values):
    if values.ndim > 1:
        raise ValueError(
            f"Value dimension is {values.ndim} != 1. "
            "Reduce the dimension by defining dim_reduction, "
            "use mean or max to get (n_structures,) shape."
        )


REDUCTIONS = {
    "mean": mean_reduction,
    "max": max_reduction,
}


class ThresholdSelection(ConfigurationSelection):
    """Select atoms based on a given threshold.

    Select atoms above a given threshold or the n_configurations with the
    highest / lowest value. Typically useful for uncertainty based selection.

    Attributes
    ----------
    key: str
        The key in 'calc.results' to select from
    threshold: float, optional
        All values above (or below if negative) this threshold will be selected.
        If n_configurations is given, 'self.threshold' will be prioritized,
        but a maximum of n_configurations will be selected.
    reference: str, optional
        For visualizing the selection a reference value can be given.
        For 'energy_uncertainty' this would typically be 'energy'.
    n_configurations: int, optional
        Number of configurations to select.
    min_distance: int, optional
        Minimum distance between selected configurations.
    dim_reduction: str, optional
        Reduces the dimensionality of the chosen uncertainty along the specified axis
        by calculating either the maximum or mean value.

        Choose from ["max", "mean"]
    reduction_axis: tuple(int), optional
        Specifies the axis along which the reduction occurs.
    """

    key = zntrack.params("energy_uncertainty")
    reference = zntrack.params("energy")
    threshold = zntrack.params(None)
    n_configurations = zntrack.params(None)
    min_distance: int = zntrack.params(1)
    dim_reduction: str = zntrack.params(None)
    reduction_axis = zntrack.params((1, 2))

    def _post_init_(self):
        if self.threshold is None and self.n_configurations is None:
            raise ValueError("Either 'threshold' or 'n_configurations' must not be None.")

        return super()._post_init_()

    def select_atoms(
        self, atoms_lst: typing.List[ase.Atoms], save_fig: bool = True
    ) -> typing.List[int]:
        """Take every nth (step) object of a given atoms list.

        Parameters
        ----------
        atoms_lst: typing.List[ase.Atoms]
            list of atoms objects to arange

        Returns
        -------
        typing.List[int]:
            list containing the taken indices
        """

        self.reduction_axis = tuple(self.reduction_axis)
        values = np.array([atoms.calc.results[self.key] for atoms in atoms_lst])

        if self.dim_reduction is not None:
            reduction_fn = REDUCTIONS[self.dim_reduction]
            values = reduction_fn(values, self.reduction_axis)

        check_dimension(values)

        if self.threshold is not None:
            if self.threshold < 0:
                indices = np.where(values < self.threshold)[0]
                if self.n_configurations is not None:
                    indices = np.argsort(values)[indices]
            else:
                indices = np.where(values > self.threshold)[0]
                if self.n_configurations is not None:
                    indices = np.argsort(values)[::-1][indices]
        else:
            if np.mean(values) > 0:
                indices = np.argsort(values)[::-1]
            else:
                indices = np.argsort(values)

        selection = self.get_selection(indices)

        return selection

    def get_selection(self, indices):
        selected = []
        for val in indices:
            # If the value is close to any of the already selected values, skip it.
            if not any(np.abs(val - np.array(selected)) < self.min_distance):
                selected.append(val)
            if len(selected) == self.n_configurations:
                break

        return selected

    def _get_plot(self, atoms_lst: typing.List[ase.Atoms], indices: typing.List[int]):
        indices = np.array(indices)

        values = np.array([atoms.calc.results[self.key] for atoms in atoms_lst])
        if self.reference is not None:
            reference = np.array(
                [atoms.calc.results[self.reference] for atoms in atoms_lst]
            )
            if reference.ndim > 1:
                reference = np.max(reference, axis=self.reduction_axis)

            fig, ax, _ = plot_with_uncertainty(
                {"std": values, "mean": reference},
                ylabel=self.key,
                xlabel="configuration",
            )
            ax.plot(indices, reference[indices], "x", color="red")
        else:
            fig, ax = plt.subplots()
            ax.plot(values, label=self.key)
            ax.plot(indices, values[indices], "x", color="red")
            ax.set_ylabel(self.key)
            ax.set_xlabel("configuration")

        fig.savefig(self.img_selection, bbox_inches="tight")
