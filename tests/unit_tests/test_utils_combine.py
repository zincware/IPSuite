from ipsuite.utils import combine
import numpy.testing as npt
import pytest


def test_get_flat_data_from_dict():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    npt.assert_array_equal([1, 2, 3, 4, 5, 6], combine.get_flat_data_from_dict(data))

    assert combine.get_flat_data_from_dict([1, 2, 3], silent_ignore=True) == [1, 2, 3]

    with pytest.raises(TypeError):
        combine.get_flat_data_from_dict([1, 2, 3])


def test_get_ids_per_key():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    assert {"a": [0, 1], "b": [0, 2]} == combine.get_ids_per_key(data, [0, 1, 3, 5])

    assert [0, 1, 3, 5] == combine.get_ids_per_key(
        [0, 1, 3, 5], [0, 1, 3, 5], silent_ignore=True
    )

    with pytest.raises(TypeError):
        combine.get_ids_per_key([0, 1, 3, 5], [0, 1, 3, 5])
