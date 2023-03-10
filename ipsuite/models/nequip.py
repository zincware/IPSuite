"""The Nequip and allegro model module."""
import logging
import pathlib
import shutil
import subprocess
import typing

import ase.io
import ase.symbols
import numpy as np
import pandas as pd
import torch
import yaml
import zntrack
import zntrack.utils
from tqdm import tqdm

from ipsuite import utils
from ipsuite.models import MLModel, Prediction
from ipsuite.utils.helpers import check_duplicate_keys

log = logging.getLogger(__name__)


def _write_xyz_input_files(
    file: typing.Union[str, pathlib.Path], data: typing.List[ase.Atoms]
) -> typing.Tuple[int, list]:
    """Write the xyz input files."""
    ase.io.write(file, data)
    n_train = len(data)

    # sort the chemical numbers
    chemical_numbers = sorted(set(data[0].get_atomic_numbers()))
    chemical_symbols = list(ase.symbols.Symbols(chemical_numbers))

    return n_train, chemical_symbols


class Nequip(MLModel):
    """The Nequip and allegro model."""

    parameter: str = zntrack.dvc.params()
    validation_data = zntrack.zn.deps()

    train_data_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "train.extxyz")
    validation_data_file: pathlib.Path = zntrack.dvc.outs(
        zntrack.nwd / "validation.extxyz"
    )

    deployed_model: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "deployed_model.pth")
    model_directory: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "model")

    metrics_batch_train = zntrack.dvc.plots(
        zntrack.nwd / "metrics_batch_train.csv"  # , y=" loss", y_label="loss"
    )
    metrics_batch_val = zntrack.dvc.plots(
        zntrack.nwd / "metrics_batch_val.csv"  # , y=" loss", y_label="loss"
    )
    metrics_epoch = zntrack.dvc.plots(
        zntrack.nwd / "metrics_epoch.csv",
        # x="epoch",
        # x_label="epochs",
        # y="validation_loss",
        # y_label="validation loss",
    )

    metrics = zntrack.zn.metrics()

    device: str = zntrack.meta.Text("cuda" if torch.cuda.is_available() else "cpu")
    remove_processed_dataset = True

    def _post_init_(self):
        """Post init hook."""
        if not self.state.loaded:
            if self.parameter is None:
                raise ValueError("Can not train nequip model without a parameter file")
            else:
                log.info(
                    "Please keep track of the parameter file with git, just like the"
                    f" params.yaml. Use 'git add {self.parameter}'."
                )

        self.data = utils.helpers.get_deps_if_node(self.data, "atoms")
        self.validation_data = utils.helpers.get_deps_if_node(
            self.validation_data, "atoms"
        )

    def _handle_parameter_file(self, n_train: int, n_val: int, chemical_symbols: list):
        """Update and rewrite the nequip parameter file."""
        parameter = pathlib.Path(self.parameter).read_text()
        parameter = yaml.safe_load(parameter)

        custom_parameters = {
            "root": self.model_directory.as_posix(),
            "run_name": "dvc-run",
            "chemical_symbols": chemical_symbols,
            # training dataset
            "dataset": "ase",
            "dataset_file_name": self.train_data_file.as_posix(),
            "n_train": n_train,
            # validation dataset
            "validation_dataset": "ase",
            "validation_dataset_file_name": self.validation_data_file.as_posix(),
            "n_val": n_val,
        }

        check_duplicate_keys(custom_parameters, parameter, log)
        parameter.update(custom_parameters)

        params_file = self.model_directory / "config.yaml"
        params_file.write_text(yaml.safe_dump(parameter))

    def train_model(self):
        """Train the model using nequip-train."""
        subprocess.check_call(
            ["nequip-train", (self.model_directory / "config.yaml").as_posix()]
        )

    def move_metrics(self):
        """Move the metrics to the correct directories for DVC."""
        shutil.move(
            self.model_directory / "dvc-run" / self.metrics_batch_train.name,
            self.metrics_batch_train,
        )
        shutil.move(
            self.model_directory / "dvc-run" / self.metrics_batch_val.name,
            self.metrics_batch_val,
        )
        shutil.move(
            self.model_directory / "dvc-run" / self.metrics_epoch.name,
            self.metrics_epoch,
        )

    def get_metrics_from_plots(self):
        """In addition to the plots write a model metric."""
        metrics_df = pd.read_csv(self.metrics_epoch)
        self.metrics = metrics_df.iloc[-1].to_dict()

    def deploy_model(self):
        """Deploy the model using nequip-deploy."""
        subprocess.check_call(
            [
                "nequip-deploy",
                "build",
                "--train-dir",
                (self.model_directory / "dvc-run").as_posix(),
                self.deployed_model.as_posix(),
            ]
        )

    def run(self):
        """Primary method to run which executes all steps of the model training."""
        self.model_directory.mkdir(exist_ok=False)

        if self.remove_processed_dataset:
            for dataset in self.model_directory.glob("processed_dataset*"):
                shutil.rmtree(dataset)

        n_train, chemical_symbols = _write_xyz_input_files(
            self.train_data_file, self.data
        )
        n_val, _ = _write_xyz_input_files(self.validation_data_file, self.validation_data)

        self._handle_parameter_file(n_train, n_val, chemical_symbols)
        self.train_model()
        self.move_metrics()
        self.get_metrics_from_plots()
        self.deploy_model()

    @property
    def calc(self):
        """Get a nequip ase calculator."""
        from nequip.ase.nequip_calculator import NequIPCalculator

        return NequIPCalculator.from_deployed_model(
            model_path=self.deployed_model.as_posix(),
            device=self.device,
        )

    def predict(self, atoms: typing.List[ase.Atoms]) -> Prediction:
        """Use the nequip model to run a prediction."""
        validation_energy = []
        validation_forces = []
        for configuration in tqdm(atoms, ncols=70):
            configuration.set_calculator(self.calc)
            if self.use_energy:
                validation_energy.append(configuration.get_potential_energy())
            if self.use_forces:
                validation_forces.append(configuration.get_forces())

        return Prediction(
            energy=np.array(validation_energy),
            forces=np.array(validation_forces),
            n_atoms=len(atoms[0]),
        )

    @property
    def lammps_pair_style(self) -> str:
        """Get the lammps pair style for the nequip model."""
        with open(self.model_directory / "config.yaml", "r") as file:
            parameter = yaml.safe_load(file)
        if "allegro.model.Allegro" in parameter.get("model_builders", []):
            return "allegro"
        return "nequip"

    @property
    def lammps_pair_coeff(self) -> typing.List[str]:
        """Get the lammps pair coefficient for the nequip model."""
        with open(self.model_directory / "config.yaml", "r") as file:
            parameter = yaml.safe_load(file)

        type_names = " ".join(parameter["chemical_symbols"])

        return [f"* * {self.deployed_model.as_posix()} {type_names}"]
