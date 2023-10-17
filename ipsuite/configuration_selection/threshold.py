"""Selecting atoms with a given step between them."""
import typing

import ase
import matplotlib.pyplot as plt
import numpy as np
import zntrack

from ipsuite.analysis.ensemble import plot_with_uncertainty
from ipsuite.configuration_selection import ConfigurationSelection


class ThresholdSelection(ConfigurationSelection):
    """Select atoms based on a given threshold.

    Select atoms above a given threshold or the n_configurations with the
    highest / lowest value. Typically useful for uncertainty based selection.

    Attributes
    ----------
    key: str
        the key in 'calc.results' to select from
    threshold: float, optional
        All values above (or below if negative) this threshold will be selected.
        If n_configurations is given, 'self.threshold' will be prioritized,
        but a maximum of n_configurations will be selected.
    reference: str, optional
        For visualizing the selection a reference value can be given.
        For 'energy_uncertainty' this would typically be 'energy'.
    n_configurations: int, optional
        number of configurations to select.
    min_distance: int, optional
        minimum distance between selected configurations.
    dim_reduction: dict
        Reduces the dimensionality of the chosen uncertainty along the specified axis
        by calculating either the maximum or mean value.

        Example for maximum force from all atoms along all Cartesian coordinates:
        {"max": (1, 2)}
    """

    key = zntrack.params("energy_uncertainty")
    reference = zntrack.params("energy")
    threshold = zntrack.params(None)
    n_configurations = zntrack.params(None)
    min_distance: int = zntrack.params(1)
    dim_reduction = zntrack.params(None)
    img_selection = zntrack.outs_path(zntrack.nwd / "selection.png")
    save_fig = zntrack.params(True)

    def _post_init_(self):
        if self.threshold is None and self.n_configurations is None:
            raise ValueError("Either 'threshold' or 'n_configurations' must not be None.")

        return super()._post_init_()

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
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
        values = np.array([atoms.calc.results[self.key] for atoms in atoms_lst])

        if self.dim_reduction is None and values.ndim > 1:
            raise ValueError(
                f"{self.key} dimension is {values.ndim} != 1. "
                "Reduce the dimension by defining dim_reduction, "
                "use mean or max to get (n_structures,) shape."
            )
        if self.dim_reduction is not None:
            if "max" in self.dim_reduction:
                self.reduction_axis = self.dim_reduction["max"]
                values = np.max(values, axis=self.reduction_axis)
            elif "mean" in self.dim_reduction:
                self.reduction_axis = self.dim_reduction["mean"]
                values = np.mean(values, axis=self.reduction_axis)

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

        selected = []
        for val in indices:
            # If the value is close to any of the already selected values, skip it.
            if not any(np.abs(val - np.array(selected)) < self.min_distance):
                selected.append(val)
            if len(selected) == self.n_configurations:
                break

        if self.save_fig:
            self._get_plot(values, atoms_lst, np.array(selected))

        return selected

    def _get_plot(
        self,
        values: np.ndarray,
        atoms_lst: typing.List[ase.Atoms],
        indices: typing.List[int],
    ):
        if self.reference is not None:
            reference = np.array(
                [atoms.calc.results[self.reference] for atoms in atoms_lst]
            )
            if self.dim_reduction is not None:
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
