import typing

import ase
import matplotlib.pyplot as plt
import numpy as np
import zntrack

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

class PropertyFilter(ConfigurationSelection):

    reference = zntrack.params("energy")
    cutoffs: typing.Union[typing.List[float]] = zntrack.params()
    direction: typing.Literal["above", "below", "both"] = zntrack.params("both")
    n_configurations = zntrack.params(None)
    min_distance: int = zntrack.params(1)
    dim_reduction: str = zntrack.params(None)
    reduction_axis = zntrack.params((1, 2))

    def _post_init_(self):
        if self.direction not in ["above", "below", "both"]:
            raise ValueError("'direction' should be set to 'above', 'below', or 'both'.")

        return super()._post_init_()
    
    def select_atoms(
            self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        self.reduction_axis = tuple(self.reduction_axis)
        values = np.array([atoms.calc.results[self.reference] for atoms in atoms_lst])

        if self.dim_reduction is not None:
            reduction_fn = REDUCTIONS[self.dim_reduction]
            values = reduction_fn(values, self.reduction_axis)

        check_dimension(values)

        lower_limit, upper_limit = self.cutoffs[0], self.cutoffs[1]

        if self.direction == "above":
            pre_selection = np.array([i for i, x in enumerate(values) if x > upper_limit])
            sorting_idx = np.argsort(values[pre_selection])[::-1]
        elif self.direction == "below":
            pre_selection = np.array([i for i, x in enumerate(values) if x < lower_limit])
            sorting_idx = np.argsort(values[pre_selection])
        else:
            pre_selection = np.array([
                i for i, x in enumerate(values) if x < lower_limit or x > upper_limit
            ])
            mean = (lower_limit+upper_limit)/2
            dist_to_mean = abs(values[pre_selection]-mean)
            sorting_idx = np.argsort(dist_to_mean)[::-1]

        selection = self.get_selection(pre_selection[sorting_idx])

        return selection

    def get_selection(self, indices):
        selection = []
        for idx in indices:
            # If the value is close to any of the already selected values, skip it.
            if not selection:
                selection.append(idx)
            if not any(np.abs(idx - np.array(selection)) < self.min_distance):
                selection.append(idx)
            if len(selection) == self.n_configurations:
                break

        return selection

    def _get_plot(self, atoms_lst: typing.List[ase.Atoms], indices: typing.List[int]):
        indices = np.array(indices)
        values = np.array([atoms.calc.results[self.reference] for atoms in atoms_lst])

        if self.dim_reduction is not None:
            reduction_fn = REDUCTIONS[self.dim_reduction]
            values = reduction_fn(values, self.reduction_axis)

        fig, ax = plt.subplots()
        ax.plot(values, label=self.reference)
        ax.plot(indices, values[indices], "x", color="red")
        ax.fill_between(
                np.arange(len(values)),
                self.cutoffs[0],
                self.cutoffs[1],
                color="black",
                alpha=0.2,
                label=f"{self.reference} +- std",
            )
        ax.set_ylabel(self.reference)
        ax.set_xlabel("configuration")

        fig.savefig(self.img_selection, bbox_inches="tight")
