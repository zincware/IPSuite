"""Helpers to work with inputs from multiple nodes."""

import dataclasses
import typing

import numpy as np


@dataclasses.dataclass
class ExcludeIds:
    """Remove entries from a dataset."""

    data: typing.Union[list, dict]
    ids: typing.Union[list, dict]

    def _post_init_(self):
        if isinstance(self.ids, list):
            self.ids = np.sort(self.ids).astype(int)
        else:
            for key, ids in self.ids.items():
                self.ids[key] = np.sort(ids).astype(int)

    def get_clean_data(self, flatten: bool = False) -> list:
        """Remove the 'ids' from the 'data'."""
        # TODO do we need a dict return here or could we just return a flat list?
        if self.ids is None:
            return self.data
        if isinstance(self.data, list) and isinstance(self.ids, list):
            return [x for i, x in enumerate(self.data) if i not in self.ids]
        elif isinstance(self.data, dict) and isinstance(self.ids, dict):
            clean_data = {}
            for key, data in self.data.items():
                if key in self.ids:
                    clean_data[key] = [
                        x for i, x in enumerate(data) if i not in self.ids[key]
                    ]
                else:
                    clean_data[key] = data
            if flatten:
                return get_flat_data_from_dict(clean_data)
            return clean_data
        else:
            raise TypeError(
                "ids and data must be of the same type. "
                f"ids is {type(self.ids)} and data is {type(self.data)}"
            )

    def get_original_ids(self, ids: list, per_key: bool = False) -> list:
        """Shift the 'ids' such that they are valid for the initial data."""
        ids = np.array(ids).astype(int)
        ids = np.sort(ids)

        if isinstance(self.ids, list):
            for removed_id in self.ids:
                ids[ids >= removed_id] += 1
        elif isinstance(self.ids, dict):
            for removed_id in self.ids_as_list:
                ids[ids >= removed_id] += 1
        if per_key:
            return get_ids_per_key(self.data, ids, silent_ignore=True)
        return ids.tolist()

    @property
    def ids_as_list(self) -> list:
        # {a: [1, 2], b: [1, 3]}
        # {a: list(10), b:list(10)}
        # [1, 2, 1+10, 3+10]
        ids = []
        size = 0
        for key in self.data:
            # we iterate through data, not ids, because ids must not contain all keys
            if key in self.ids:
                ids.append(np.array(self.ids[key]) + size)
            size += len(self.data[key])
        if len(ids):
            ids = np.concatenate(ids)
            ids = np.sort(ids)
            return ids.astype(int).tolist()
        return []


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
