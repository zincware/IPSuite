import pytest

import ipsuite as ips

@pytest.mark.skip(reason="dagshub is no longer public")
def test_ips_bin_property(data_repo, traj_file):
    """Test the BarycenterMapping class."""
    # data = ips.AddData.from_rev(name="BMIM_BF4_363_15K")

    with ips.Project() as project:
        data = ips.AddData(file=traj_file)
        e_hist = ips.EnergyHistogram(data=data.frames)
        f_hist = ips.ForcesHistogram(data=data.frames, bins=100)

    project.repro()

    assert e_hist.labels_df["bin_edges"].shape == (1,)
    assert f_hist.labels_df["bin_edges"].shape == (100,)
