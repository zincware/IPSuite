import pytest

import ipsuite as ips


def test_ips_bin_property(data_repo):
    """Test the BarycenterMapping class."""
    data = ips.AddData.from_rev(name="BMIM_BF4_363_15K")

    with ips.Project() as project:
        e_hist = ips.analysis.EnergyHistogram(data=data.atoms)
        f_hist = ips.analysis.ForcesHistogram(data=data.atoms, bins=100)

    project.repro()

    assert e_hist.labels_df["bin_edges"].shape == (1,)
    assert f_hist.labels_df["bin_edges"].shape == (100,)
