"""Select configurations by item, e.g. slice or list of indices."""

import typing

import ase
import zntrack

from ipsuite.configuration_selection import ConfigurationSelection


class IndexSelection(ConfigurationSelection):
    """Select configurations by explicit indices or slice parameters.

    Parameters
    ----------
    data : list[ase.Atoms]
        The atomic configurations to select from.
    indices : list[int], optional
        Explicit list of indices to select. Cannot be used with slice parameters.
    start : int, optional
        Start index for slice selection.
    stop : int, optional
        Stop index for slice selection.
    step : int, optional
        Step size for slice selection.

    Attributes
    ----------
    selected_ids : list[int]
        Indices of selected configurations.
    frames : list[ase.Atoms]
        The selected atomic configurations.
    excluded_frames : list[ase.Atoms]
        The atomic configurations that were not selected.

    Examples
    --------
    >>> with project:
    ...     data = ips.AddData(file="ethanol.xyz")
    ...     selector = ips.IndexSelection(data=data.frames, indices=[0, 5, 10, 15])
    >>> project.repro()
    >>> print(f"Selected {len(selector.selected_ids)} configurations with IDs: "
    ...       f"{selector.selected_ids}")
    Selected 4 configurations with IDs: [0, 5, 10, 15]
    """

    indices: list[int] | None = zntrack.params(None)
    start: int | None = zntrack.params(None)
    stop: int | None = zntrack.params(None)
    step: int | None = zntrack.params(None)

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Select Atoms randomly."""
        if self.indices:
            if isinstance(self.indices, typing.Iterable):
                return self.indices
            else:
                raise ValueError("indices must be an iterable of integers")
        else:
            idx_slice = slice(self.start, self.stop, self.step)
            return list(range(len(atoms_lst)))[idx_slice]
