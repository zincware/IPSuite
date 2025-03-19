import typing as t

import ase
import matplotlib.pyplot as plt
import numpy as np
import zntrack

from ipsuite import base

class FilterOutlier(base.IPSNode):
    """Remove outliers from the data based on a given property.

    Attributes
    ----------
    key : str, default="energy"
        The property to filter on.
    threshold : float, default=3
        The threshold for filtering in units of standard deviations.
    direction : {"above", "below", "both"}, default="both"
        The direction to filter in.
    """

    data: list[ase.Atoms] = zntrack.deps()
    key: str = zntrack.params("energy")
    threshold: float = zntrack.params(3)
    direction: t.Literal["above", "below", "both"] = zntrack.params("both")

    filtered_indices: list = zntrack.outs()
    histogram: str = zntrack.outs_path(zntrack.nwd / "histogram.png")

    def run(self):
        values = [x.calc.results[self.key] for x in self.data]
        mean = np.mean(values)
        std = np.std(values)

        if self.direction == "above":
            self.filtered_indices = [
                i for i, x in enumerate(values) if x > mean + self.threshold * std
            ]
        elif self.direction == "below":
            self.filtered_indices = [
                i for i, x in enumerate(values) if x < mean - self.threshold * std
            ]
        else:
            self.filtered_indices = [
                i
                for i, x in enumerate(values)
                if x > mean + self.threshold * std or x < mean - self.threshold * std
            ]

        fig, ax = plt.subplots(3, figsize=(10, 10))
        ax[0].hist(values, bins=100)
        ax[0].set_title("All")
        ax[1].hist(
            [values[i] for i in range(len(values)) if i not in self.filtered_indices],
            bins=100,
        )
        ax[1].set_title("Filtered")
        ax[2].hist([values[i] for i in self.filtered_indices], bins=100)
        ax[2].set_title("Excluded")
        fig.savefig(self.histogram, bbox_inches="tight")

    @property
    def frames(self) -> list[ase.Atoms]:
        return [
            self.data[i] for i in range(len(self.data)) if i not in self.filtered_indices
        ]

    @property
    def excluded_frames(self):
        return [self.data[i] for i in self.filtered_indices]


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


class PropertyFilter(base.IPSNode):
    data: list[ase.Atoms] = zntrack.deps()
    reference: str = zntrack.params("energy")
    cutoffs: t.Union[t.List[float]] = zntrack.params()
    direction: t.Literal["above", "below", "both"] = zntrack.params("both")
    n_configurations: int = zntrack.params(None)
    min_distance: int = zntrack.params(1)
    dim_reduction: str = zntrack.params(None)
    reduction_axis: t.List[int] = zntrack.params((1, 2))

    filtered_indices: list = zntrack.outs()
    selection_plot: str = zntrack.outs_path(zntrack.nwd / "slecection.png")
    
    def _post_init_(self):
        if self.direction not in ["above", "below", "both"]:
            raise ValueError("'direction' should be set to 'above', 'below', or 'both'.")

        return super()._post_init_()

    def pad_list(self, inputs):
        max_rows = max(val.shape[0] for val in inputs)
        
        # Ensure all arrays are (N, 3)
        padded_list = []
        for arr in inputs:
            pad_rows = max_rows - arr.shape[0]
            if pad_rows > 0:
                # Pad with the given value along axis 0
                padding = np.full((pad_rows, 3), 0)
                padded_arr = np.vstack([arr, padding])
            else:
                padded_arr = arr  # Already the max size
            padded_list.append(padded_arr)
        return np.array(padded_list)
    
    def run(self) -> t.List[int]:
        self.reduction_axis = tuple(self.reduction_axis)
        values = [atoms.calc.results[self.reference] for atoms in self.data]
        values = self.pad_list(values)
        print(values.shape)
        if self.dim_reduction is not None:
            reduction_fn = REDUCTIONS[self.dim_reduction]
            values = reduction_fn(values, self.reduction_axis)

        check_dimension(values)

        lower_limit, upper_limit = self.cutoffs[0], self.cutoffs[1]
        self.outlier = True

        if self.direction == "above":
            pre_selection = np.array([i for i, x in enumerate(values) if x > upper_limit])
            sorting_idx = np.argsort(values[pre_selection])[::-1]
        elif self.direction == "below":
            pre_selection = np.array([i for i, x in enumerate(values) if x < lower_limit])
            sorting_idx = np.argsort(values[pre_selection])
        else:
            pre_selection = [i for i, x in enumerate(values) if x < lower_limit or x > upper_limit]
            if pre_selection:
                pre_selection = np.array(pre_selection)
                mean = (lower_limit + upper_limit) / 2
                dist_to_mean = abs(values[pre_selection] - mean)
                sorting_idx = np.argsort(dist_to_mean)[::-1]
            else:
                self.outlier = False
                print('no outlier')
        
        if self.outlier:
            self.filtered_indices = self.get_selection(pre_selection[sorting_idx])
            selection_idx = np.array(self.filtered_indices)

            values = [atoms.calc.results[self.reference] for atoms in self.data]
            values = self.pad_list(values)

            if self.dim_reduction is not None:
                reduction_fn = REDUCTIONS[self.dim_reduction]
                values = reduction_fn(values, self.reduction_axis)

            fig, ax = plt.subplots()
            ax.plot(values, label=self.reference)
            ax.plot(selection_idx, values[selection_idx], "x", color="red")
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

            fig.savefig(self.selection_plot, bbox_inches="tight")
        else:
            self.filtered_indices = [len(self.data)+1]
            fig, ax = plt.subplots()
            ax.plot(1, 1, label=self.reference)
            fig.savefig(self.selection_plot, bbox_inches="tight")
            
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
            
        for id, val in enumerate(selection):
            selection[id] = int(val)
        return selection

    @property
    def frames(self) -> list[ase.Atoms]:
        return [
                self.data[i] for i in range(len(self.data)) if i not in self.filtered_indices
        ]

    @property
    def excluded_frames(self):
        if self.outlier:
            return [self.data[i] for i in self.filtered_indices]
        else:
            return []