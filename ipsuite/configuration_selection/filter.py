import typing as t

import ase
import matplotlib.pyplot as plt
import numpy as np
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


def direct_cutoff(values, threshold, cutoffs):
    # Filtering the direct cutoff values
    if cutoffs is None:
        raise ValueError("cutoffs have to be specified for using the direct cutoff filter.")
    return (cutoffs[0], cutoffs[1])

def cutoff_around_mean(values, threshold, cutoffs):
    # Filtering in multiples of the standard deviation around the mean.
    mean = np.mean(values)
    std = np.std(values)

    upper_cutoff = mean + threshold * std
    lower_cutoff = mean - threshold * std
    return (lower_cutoff, upper_cutoff)

CUTOFF = {
    "direct": direct_cutoff,
    "around_mean": cutoff_around_mean
}


class FilterOutlier(ConfigurationSelection):
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
        Lower and upper cutoff.
    """

    key: str = zntrack.params("energy")
    cutoff_type: t.Literal["direct", "around_mean"] = zntrack.params("around_mean")
    direction: t.Literal["above", "below", "both"] = zntrack.params("both")
    threshold: float = zntrack.params(3)
    cutoffs: t.Union[t.List[float], None] = zntrack.params(None)
    
    def select_atoms(self, atoms_lst: t.List[ase.Atoms]) -> t.List[int]:         
        values = [atoms.calc.results[self.key] for atoms in atoms_lst]

        # get maximal atomic value per struckture
        if np.array(values).ndim == 3:
            # calculates the maximal magnetude of cartesian values
            values = [np.max(np.linalg.norm(value, axis=1), axis=0) for value in values]
        elif np.array(values).ndim == 2:
            # calculates the maximal atomic values
            values = [np.max(value, axis=0) for value in values]
            
        lower_limit, upper_limit = CUTOFF[self.cutoff_type](
            values,
            self.threshold,
            self.cutoffs,
        )

        if self.direction == "above":
            selection = [
                i for i, x in enumerate(values) if x < upper_limit
            ]
        elif self.direction == "below":
            selection = [
                i for i, x in enumerate(values) if x > lower_limit
            ]
        else:
            selection = [
                i
                for i, x in enumerate(values)
                if x > lower_limit and x < upper_limit
            ]

        return selection


    def _get_plot(self, atoms_lst: t.List[ase.Atoms], indices: t.List[int]):
        values = [atoms.calc.results[self.key] for atoms in atoms_lst]

        # check if property is in cartesian basis
        if np.array(values).ndim == 3:
            # calculates the maximal magnetude of cartesian values
            values = [np.max(np.linalg.norm(value, axis=1), axis=0) for value in values]

        fig, ax = plt.subplots(3, figsize=(10, 10))
        ax[0].hist(values, bins=100)
        ax[0].set_title("All")
        ax[1].hist(
            [values[i] for i in range(len(values)) if i not in indices],
            bins=100,
        )
        ax[1].set_title("Filtered")
        ax[2].hist([values[i] for i in indices], bins=100)
        ax[2].set_title("Excluded")
        fig.savefig(self.img_selection, bbox_inches="tight")