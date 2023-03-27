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


def test_ExcludeIds_list():
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


def test_ExcludeIds_dict():
    data = {"a": list(range(10))}
    exclude = combine.ExcludeIds(data, ids={"a": [0]})
    assert {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9]} == exclude.get_clean_data()

    # # we have excluded [0] so everything after [0] is shifted by 1
    assert exclude.get_original_ids([0, 1, 2]) == [1, 2, 3]

    # we have excluded [0, 5] so everything after [0] is shifted by 1
    #  and everything after [5] is shifted by 1+1 = 2
    exclude = combine.ExcludeIds(data, ids={"a": [0, 5]})
    assert exclude.get_clean_data() == {"a": [1, 2, 3, 4, 6, 7, 8, 9]}
    assert exclude.get_original_ids([0, 1, 5, 6]) == [1, 2, 7, 8]

    exclude = combine.ExcludeIds(data, ids={})
    assert exclude.get_clean_data() == {"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    assert exclude.get_original_ids([0, 1, 5, 6]) == [0, 1, 5, 6]


def test_ids_as_list():
    data = {"a": list(range(8)), "b": list(range(10)), "c": list(range(10))}

    exclude = combine.ExcludeIds(data, ids={"a": [0, 5], "b": [3, 7], "c": [1]})
    assert exclude.ids_as_list == [0, 5, 8 + 3, 8 + 7, 8 + 10 + 1]

    exclude = combine.ExcludeIds(data, ids={"a": [0, 5], "c": [1]})
    assert exclude.ids_as_list == [0, 5, 8 + 10 + 1]


@pytest.mark.parametrize(
    "ids",
    [
        {"a": [0, 5], "b": [3, 7], "c": [1]},
        [{"a": [0, 5], "b": [3]}, {"a": [], "b": [7], "c": [1]}],
    ],
)
def test_ExcludeIds_dict_multi(ids):
    data = {"a": list(range(8)), "b": list(range(10)), "c": list(range(10))}
    exclude = combine.ExcludeIds(data, ids=ids)
    assert exclude.ids_as_list == [0, 5, 8 + 3, 8 + 7, 8 + 10 + 1]
    assert exclude.get_clean_data() == {
        "a": [1, 2, 3, 4, 6, 7],
        "b": [0, 1, 2, 4, 5, 6, 8, 9],
        "c": [0, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    assert exclude.get_clean_data(flatten=True) == combine.get_flat_data_from_dict(
        exclude.get_clean_data()
    )
    assert exclude.get_original_ids([0, 1, 2, 3, 4, 5]) == [1, 2, 3, 4, 6, 7]
    assert exclude.get_original_ids([8 + 0]) == [2 + 8]
    assert exclude.get_original_ids([8 + 0, 8 + 1, 8 + 2, 8 + 3, 8 + 4, 8 + 5]) == [
        2 + 8 + 0,
        3 + 8 + 1,
        3 + 8 + 2,
        3 + 8 + 3,
        4 + 8 + 4,
        4 + 8 + 5,
    ]  # [missing + size of a, idx]

    for x in exclude.get_original_ids(
        [8 + 0, 8 + 1, 8 + 2, 8 + 3, 8 + 4, 8 + 5]
    ):  # the ids must not be in the excluded ids.
        assert x not in exclude.ids_as_list

    assert exclude.get_original_ids([8 + 10 + 0, 8 + 10 + 1]) == [
        5 + 8 + 10 + 0,
        5 + 8 + 10 + 1,
    ]  # [missing + size of a + size of b, idx]
    for x in exclude.get_original_ids([0, 1, 2, 3, 8 + 10 + 0, 8 + 10 + 1]):
        assert x not in exclude.ids_as_list

    assert exclude.get_original_ids(
        [8 + 10 + 0, 8 + 10 + 1], per_key=True
    ) == combine.get_ids_per_key(data, exclude.get_original_ids([8 + 10 + 0, 8 + 10 + 1]))
