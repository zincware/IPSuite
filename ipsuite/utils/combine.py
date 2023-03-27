"""Helpers to work with inputs from multiple nodes."""

import typing

import numpy as np


def get_flat_data_from_dict(data: dict) -> list:
    """Flatten a dictionary of lists into a single list.

    Example
    -------
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> get_flat_data_from_dict(data)
        >>> # [1, 2, 3, 4, 5, 6]
    """
    flat_data = []
    for x in data.values():
        flat_data.extend(x)
    return flat_data


def get_ids_per_key(data: dict, ids: list) -> typing.Dict[str, list]:
    """Get the ids per key from a dictionary of lists.

    Parameters
    ----------
    data : dict
        Dictionary of lists.
    ids : list
        List of ids. The ids are assumed to be in the same order as
        'get_flat_data_from_dict(data)'.

    Example
    -------
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> get_ids_per_key(data, [0, 1, 3, 5])
        >>> # {'a': [0, 1], 'b': [0, 2]}
    """
    ids_per_key = {}
    ids = np.array(ids).astype(int)
    start = 0

    for key, val in data.items():
        condition = ids - start
        condition = np.logical_and(condition < len(val), condition >= 0)

        ids_per_key[key] = (ids[condition] - start).tolist()
        start += len(val)

    return ids_per_key
