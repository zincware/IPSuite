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
    def frames(self):
        return [
            self.data[i] for i in range(len(self.data)) if i not in self.filtered_indices
        ]

    @property
    def excluded_frames(self):
        return [self.data[i] for i in self.filtered_indices]
