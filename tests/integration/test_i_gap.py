import ase
import numpy as np
import xmltodict

import ipsuite


def test_model_training(proj_path, traj_file):
    with ipsuite.Project() as project:
        data = ipsuite.AddData(file=traj_file, name="data_1")
        train_selection = ipsuite.configuration_selection.UniformEnergeticSelection(
            data=data.atoms, n_configurations=10, name="train_data"
        )

        validation_selection = ipsuite.configuration_selection.UniformEnergeticSelection(
            data=train_selection.excluded_frames, n_configurations=8, name="val_data"
        )

        model = ipsuite.models.GAP(soap={"cutoff": 0.7}, data=train_selection.frames)

    project.repro()

    prediction = model.predict(data.atoms)
    assert isinstance(prediction, list)
    assert isinstance(prediction[0], ase.Atoms)

    data.atoms[0].calc = model.get_calculator()
    with open(model.model_directory.resolve() / "model.xml", "r") as file:
        second_line = file.readlines()[1]
    content_as_dict = xmltodict.parse(second_line)
    gap_xml_label = f"{content_as_dict['Potential']['@label']}"

    assert isinstance(data.atoms[0].get_potential_energy(), float)
    assert data.atoms[0].get_potential_energy() != 0.0

    assert isinstance(data.atoms[0].get_forces(), np.ndarray)
    assert data.atoms[0].get_forces()[0, 0] != 0.0

    assert model.lammps_pair_style == "quip"
    assert (
        model.lammps_pair_coeff
        == f'* * {model.model_directory.resolve()}/model.xml "Potential'
        f' xml_label={gap_xml_label}" 6 8'
    )
