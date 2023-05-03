import os
import pathlib
import shutil

import ipsuite as ips
from ipsuite.calculators import ApaxJaxMD
from ipsuite.configuration_selection.uniform_energetic import UniformEnergeticSelection
from ipsuite.models import Apax

TEST_PATH = pathlib.Path(__file__).parent.resolve()


def test_model_training(proj_path, traj_file):
    minmal_path = pathlib.Path(os.path.dirname(TEST_PATH))

    model_minimal = shutil.copy(
        minmal_path / "apax_minimal.yaml", proj_path / "apax_minimal.yaml"
    )
    md_minimal = shutil.copy(
        TEST_PATH / "apax_md_minimal.yaml", proj_path / "apax_md_minimal.yaml"
    )

    with ips.Project() as project:
        raw_data = ips.AddData(file=traj_file, name="raw_data")
        train_selection = UniformEnergeticSelection(
            data=raw_data.atoms, n_configurations=10, name="train_data"
        )

        val_selection = UniformEnergeticSelection(
            data=train_selection.excluded_atoms, n_configurations=8, name="val_data"
        )

        model = Apax(
            parameter_file=model_minimal,
            data=train_selection.atoms,
            validation_data=val_selection.atoms,
        )

        md = ApaxJaxMD(
            model=model,
            data=raw_data,
            md_parameter_file=md_minimal,
        )
    project.run()

    md.load()

    assert len(md.atoms) == 6
