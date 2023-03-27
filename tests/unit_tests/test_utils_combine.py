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
    assert {"a": [0, 1], "b": [0, 2]} == combine.get_ids_per_key(data, [3, 1, 0, 5])

    assert [0, 1, 3, 5] == combine.get_ids_per_key(
        [0, 1, 3, 5], [0, 1, 3, 5], silent_ignore=True
    )

    with pytest.raises(TypeError):
        combine.get_ids_per_key([0, 1, 3, 5], [0, 1, 3, 5])


def test_ExcludeIds():
    data = list(range(10))
    exclude = combine.ExcludeIds(data, ids=[0])
    assert [1, 2, 3, 4, 5, 6, 7, 8, 9] == exclude.get_clean_data()

    # we have excluded [0] so everything after [0] is shifted by 1
    assert exclude.get_original_ids([0, 1, 2]) == [1, 2, 3]

    # we have excluded [0, 5] so everything after [0] is shifted by 1
    #  and everything after [5] is shifted by 1+1 = 2
    exclude = combine.ExcludeIds(data, ids=[0, 5])
    assert exclude.get_clean_data() == [1, 2, 3, 4, 6, 7, 8, 9]
    assert exclude.get_original_ids([0, 1, 5, 6]) == [1, 2, 7, 8]
