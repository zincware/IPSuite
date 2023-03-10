import numpy as np
import pytest

from ipsuite.utils import metrics


@pytest.fixture()
def data_a():
    return np.linspace(0, 1, 100) ** 2


@pytest.fixture()
def data_b():
    return np.linspace(0.5, 1.5, 100) ** 2


def test_lp_norm(data_a, data_b):
    assert metrics.calculate_l_p_norm(data_a, data_b, p=1) == 0.75
    assert metrics.calculate_l_p_norm(data_a, data_b, p=2) == 0.08046843076740312
    assert metrics.calculate_l_p_norm(data_a, data_b, p=4) == 0.028019232241236994


def test_mae(data_a, data_b):
    assert metrics.mean_absolute_error(data_a, data_b) == 0.75


def test_rmse(data_a, data_b):
    assert metrics.root_mean_squared_error(data_a, data_b) == 0.8046843076740314


def test_maximum_error(data_a, data_b):
    assert metrics.maximum_error(data_a, data_b) == 1.25


def test_get_u_vecs():
    unit_vectors = metrics.get_u_vecs(np.random.random(size=(50, 3)))
    norm = np.linalg.norm(unit_vectors, axis=-1)
    np.testing.assert_array_almost_equal(norm, np.ones_like(norm))


def test_get_angles():
    vector_a = np.array([[1, 0, 0], [10, 0, 0], [13, 13, 0], [1, 1, 0]])
    vector_b = np.array([[0, 1, 0], [0, 1, 0], [0.001, 0, 0], [0, 0, 1]])

    np.testing.assert_array_almost_equal(
        metrics.get_angles(vector_a, vector_b), [90, 90, 45, 90]
    )
