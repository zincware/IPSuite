"""Use index selection to test the selection base."""

import ipsuite as ips
import pytest


@pytest.mark.parametrize("proj_w_data", [1], indirect=True)
def test_direct_selection(proj_w_data):
    proj, data = proj_w_data
    with proj:
        selection = ips.configuration_selection.IndexSelection(
            data=data[0], indices=[0, 1, 2]
        )

    proj.run()
    selection.load()
    data[0].load()
    assert selection.selected_configurations == {"data_0": [0, 1, 2]}

    assert selection.atoms == data[0].atoms[:3]
