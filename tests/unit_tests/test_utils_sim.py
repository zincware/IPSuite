import ase.build
import numpy.testing as npt

from ipsuite.utils.ase_sim import get_box_from_density
from ipsuite.utils.helpers import lower_dict


def test_get_box():
    water = [ase.build.molecule("H2O")]

    box = get_box_from_density([water], [10], 997)
    npt.assert_almost_equal(box, [6.6946735, 6.6946735, 6.6946735])


def test_lower_dict():
    d = {"A": 1, "B": 2, "C": 3}
    lower = lower_dict(d)
    assert lower == {"a": 1, "b": 2, "c": 3}

    d = {"A": 1, "C": {"D": 4}}
    lower = lower_dict(d)
    assert lower == {"a": 1, "c": {"d": 4}}

    d = {"A": [1, 2], "B": [3, 4]}
    lower = lower_dict(d)
    assert lower == {"a": [1, 2], "b": [3, 4]}
