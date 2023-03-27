"""Helpers to work with inputs from multiple nodes."""

import dataclasses
import typing

import numpy as np


@dataclasses.dataclass
class ExcludeIds:
    """Remove entries from a dataset."""

    data: list
    ids: list

    def get_clean_data(self):
        """Remove the 'ids' from the 'data'."""
        return [x for i, x in enumerate(self.data) if i not in self.ids]

    def get_original_ids(self, ids: list):
        """Shift the 'ids' such that they are valid for the initial data."""
        ids = np.array(ids).astype(int)
        ids = np.sort(ids)
        for removed_id in self.ids:
            ids[ids >= removed_id] += 1
        return ids.tolist()


def get_flat_data_from_dict(data: dict, silent_ignore: bool = False) -> list:
    """Flatten a dictionary of lists into a single list.

    Parameters
    ----------
    data : dict
        Dictionary of lists.
    silent_ignore : bool, optional
        If True, the function will return the input if it is not a
        dictionary. If False, it will raise a TypeError.

    Example
    -------
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> get_flat_data_from_dict(data)
        >>> # [1, 2, 3, 4, 5, 6]
    """
    if not isinstance(data, dict):
        if silent_ignore:
            return data
        else:
            raise TypeError(f"data must be a dictionary and not {type(data)}")

    flat_data = []
    for x in data.values():
        flat_data.extend(x)
    return flat_data


def get_ids_per_key(
    data: dict, ids: list, silent_ignore: bool = False
) -> typing.Dict[str, list]:
    """Get the ids per key from a dictionary of lists.

    Parameters
    ----------
    data : dict
        Dictionary of lists.
    ids : list
        List of ids. The ids are assumed to be taken from the flattened
        'get_flat_data_from_dict(data)' data. If the ids aren't sorted,
        they will be sorted.
    silent_ignore : bool, optional
        If True, the function will return the input if it is not a
        dictionary. If False, it will raise a TypeError.

    Example
    -------
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> get_ids_per_key(data, [0, 1, 3, 5])
        >>> # {'a': [0, 1], 'b': [0, 2]}
    """
    if not isinstance(data, dict):
        if silent_ignore:
            return ids
        else:
            raise TypeError(f"data must be a dictionary and not {type(data)}")

    ids_per_key = {}
    ids = np.array(ids).astype(int)
    ids = np.sort(ids)
    start = 0

    for key, val in data.items():
        condition = ids - start
        condition = np.logical_and(condition < len(val), condition >= 0)

        ids_per_key[key] = (ids[condition] - start).tolist()
        start += len(val)

    return ids_per_key
