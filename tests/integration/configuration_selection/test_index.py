"""Use index selection to test the selection base."""

import ipsuite as ips
import pytest


@pytest.mark.parametrize("data_style", ["lst", "single", "attr", "attr_lst", "dct"])
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("proj_w_data", [1], indirect=True)
def test_direct_selection(proj_w_data, eager, data_style):
    proj, data = proj_w_data

    if eager:
        for node in data:
            node.load()
    with proj:
        if data_style == "lst":
            _data = data
        elif data_style == "single":
            _data = data[0]
        elif data_style == "attr":
            _data = data[0].atoms
        elif data_style == "attr_lst":
            _data = [d.atoms for d in data]
        elif data_style == "dct":
            _data = {"data_0": data[0].atoms}
        else:
            raise ValueError(data_style)

        selection = ips.configuration_selection.IndexSelection(
            data=_data, indices=[0, 1, 2]
        )
        selection_w_exclusion = ips.configuration_selection.IndexSelection(
            data=_data,
            indices=[0, 1, 2],
            exclude_configurations=selection.selected_configurations,
            name="selection2",
        )

    proj.run(eager=eager)
    if not eager:
        selection.load()
        data[0].load()
        selection_w_exclusion.load()
    assert selection.selected_configurations == {"data_0": [0, 1, 2]}
    assert selection_w_exclusion.selected_configurations == {"data_0": [3, 4, 5]}

    assert selection.atoms == data[0].atoms[:3]
    assert selection_w_exclusion.atoms == data[0].atoms[3:6]
