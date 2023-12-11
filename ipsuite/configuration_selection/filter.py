import typing as t

import matplotlib.pyplot as plt
import numpy as np
import zntrack

from ipsuite import base


def direct_cutoff(values, threshold, cutoffs):
    # Filtering the direct cutoff values
    if cutoffs is None:
        raise ValueError("cutoffs not specified.")
    return (cutoffs[0], cutoffs[1])

def cutoff_around_mean(values, threshold, cutoffs):
    # Filtering in multiples of the standard deviation around the mean.
    mean = np.mean(values)
    std = np.std(values)

    upper_limit = mean + threshold * std
    lower_limit = mean - threshold * std
    return (upper_limit, lower_limit)

CUTOFF = {
    "direct": direct_cutoff,
    "around_mean": cutoff_around_mean
}


class FilterOutliers(base.ProcessAtoms):
    """Remove outliers from the data based on a given property.

    Attributes
    ----------
    key : str, default="energy"
        The property to filter on.
    cutoff_type : {"direct", "around_mean"}, default="around_mean"
        Defines the cutoff type.
    direction : {"above", "below", "both"}, default="both"
        The direction to filter in.
    threshold : float, default=3
        The threshold for filtering in units of standard deviations.
    cutoffs : list(float), default=None
        Upper and lower cutoff.
    """

    key: str = zntrack.params("energy")
    cutoff_type: t.Literal["direct", "around_mean"] = zntrack.params("around_mean")
    direction: t.Literal["above", "below", "both"] = zntrack.params("both")
    threshold: float = zntrack.params(3)
    cutoffs: list(float) = zntrack.params(None)


    filtered_indices: list = zntrack.outs()
    histogram: str = zntrack.outs_path(zntrack.nwd / "histogram.png")

    def run(self):         
        values = [x.calc.results[self.key] for x in self.data]

        if len(values[0][0]) == 3:
            # calculates the maximal magnetude of cartesian values
            values = [np.max(np.linalg.norm(value, axis=1), axis=0) for value in values]

        upper_limit, lower_limit = CUTOFF(self.cutoff_type)(
            values,
            self.threshold,
            self.cutoffs,
        )

        if self.direction == "above":
            self.filtered_indices = [
                i for i, x in enumerate(values) if x > upper_limit
            ]
        elif self.direction == "below":
            self.filtered_indices = [
                i for i, x in enumerate(values) if x < lower_limit
            ]
        else:
            self.filtered_indices = [
                i
                for i, x in enumerate(values)
                if x > upper_limit or x < lower_limit
            ]

        plot_hist(values, self.filtered_indices, self.histogram)

    @property
    def atoms(self):
        return [
            self.data[i] for i in range(len(self.data)) if i not in self.filtered_indices
        ]

    @property
    def excluded_atoms(self):
        return [self.data[i] for i in self.filtered_indices]
    

def plot_hist(values, filtered_indices, histogram):
    fig, ax = plt.subplots(3, figsize=(10, 10))
    ax[0].hist(values, bins=100)
    ax[0].set_title("All")
    ax[1].hist(
        [values[i] for i in range(len(values)) if i not in filtered_indices],
        bins=100,
    )
    ax[1].set_title("Filtered")
    ax[2].hist([values[i] for i in filtered_indices], bins=100)
    ax[2].set_title("Excluded")
    fig.savefig(histogram, bbox_inches="tight")