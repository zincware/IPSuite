import ase.build
import numpy.testing as npt

from ipsuite.utils.ase_sim import get_box_from_density


def test_get_box():
    water = [ase.build.molecule("H2O")]

    box = get_box_from_density([water], [10], 997)
    npt.assert_almost_equal(box, [6.6946735, 6.6946735, 6.6946735])
