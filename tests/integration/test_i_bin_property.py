import pytest
import numpy as np
from ase import Atoms
import ipsuite as ips


@pytest.mark.parametrize("eager", [True, False])
def test_ips_bin_property(data_repo, eager):
    """Test the BarycenterMapping class."""
    data = ips.AddData.from_rev(name="BMIM_BF4_363_15K")

    with ips.Project() as project:
        e_hist = ips.analysis.EnergyHistogram(data=data.atoms)
        f_hist = ips.analysis.ForcesHistogram(data=data.atoms)

    project.run(eager=eager)
    if not eager:
        e_hist.load()
        f_hist.load()

    assert e_hist.labels_df["bin_edges"].shape == (1,)
