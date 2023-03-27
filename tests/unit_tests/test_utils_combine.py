from ipsuite.utils import combine
import numpy.testing as npt


def test_get_flat_data_from_dict():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    npt.assert_array_equal([1, 2, 3, 4, 5, 6], combine.get_flat_data_from_dict(data))


def test_get_ids_per_key():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    assert {"a": [0, 1], "b": [0, 2]} == combine.get_ids_per_key(data, [0, 1, 3, 5])
