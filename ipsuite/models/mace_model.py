"""MACE model module."""

import functools
import json
import logging
import pathlib
import subprocess
import typing

import ase
import numpy as np
import pandas as pd
import torch
import tqdm
import zntrack
from mace.calculators import MACECalculator

from ipsuite import utils
from ipsuite.models import MLModel, Prediction
from ipsuite.static_data import STATIC_PATH

log = logging.getLogger(__name__)


def execute(cmd, **kwargs):
    """Execute a command and yield the output line by line.
    
    Adapted from https://stackoverflow.com/a/4417735/10504481
    """
    popen = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, universal_newlines=True, **kwargs
    )
    yield from iter(popen.stdout.readline, "")
    popen.stdout.close()
    if return_code := popen.wait():  # finally a use for walrus operator
        raise subprocess.CalledProcessError(return_code, cmd)


class MACE(MLModel):
    """MACE model."""

    train_data_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "train-data.extxyz")

    test_data = zntrack.zn.deps()
    test_data_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "test-data.extxyz")
    model_dir: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "model")
    training: pathlib.Path = zntrack.dvc.plots(
        zntrack.nwd / "training.csv",
        template=STATIC_PATH / "y_log.json",
        x="epoch",
        y="loss",
    )

    seed = zntrack.zn.params(42)

    hidden_irreps: str = zntrack.zn.params("16x0e + 16x1o")
    r_max: float = zntrack.zn.params(5.0)
    batch_size: int = zntrack.zn.params(10)
    max_num_epochs: int = zntrack.zn.params(1000)
    device: str = zntrack.meta.Text("cuda" if torch.cuda.is_available() else "cpu")

    swa: bool = zntrack.zn.params(True)
    start_swa: int = zntrack.zn.params(1200)

    ema: bool = zntrack.zn.params(True)
    ema_decay: int = zntrack.zn.params(0.99)

    amsgrad: bool = zntrack.zn.params(True)

    num_radial_basis: int = zntrack.zn.params(8)
    num_cutoff_basis: int = zntrack.zn.params(5)

    num_interactions: int = zntrack.zn.params(2)

    def _post_init_(self):
        self.data = utils.helpers.get_deps_if_node(self.data, "atoms")
        self.test_data = utils.helpers.get_deps_if_node(self.test_data, "atoms")

    def run(self):
        """Train a MACE model."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        cmd = """curl -sSL https://raw.githubusercontent.com/ACEsuit/mace/main/scripts/run_train.py | python - """  # noqa E501
        cmd += '--name="MACE_model" '
        cmd += f'--train_file="{self.train_data_file.resolve().as_posix()}" '
        cmd += "--valid_fraction=0.05 "
        cmd += f'--test_file="{self.test_data_file.resolve().as_posix()}" '
        # cmd += """--config_type_weights='{"Default":1.0}' """
        cmd += """--E0s='average' """
        cmd += """--model="MACE" """
        cmd += f"""--seed={self.seed} """
        cmd += f"""--hidden_irreps='{self.hidden_irreps}' """
        cmd += f"--r_max={self.r_max} "
        cmd += f"--batch_size={self.batch_size} "
        cmd += f"--max_num_epochs={self.max_num_epochs} "
        cmd += f"--device={self.device} "
        cmd += f"--num_radial_basis={self.num_radial_basis} "
        cmd += f"--num_cutoff_basis={self.num_cutoff_basis} "
        cmd += f"--num_interactions={self.num_interactions} "

        if self.swa and self.start_swa < self.max_num_epochs:
            cmd += f"--swa --start_swa={self.start_swa} "

        if self.ema:
            cmd += f"--ema --ema_decay={self.ema_decay} "

        if self.amsgrad:
            cmd += "--amsgrad "

        self.write_data_to_file(file=self.train_data_file, atoms_list=self.data)
        self.write_data_to_file(file=self.test_data_file, atoms_list=self.test_data)

        log.debug(f"Running: {cmd}")

        for path in execute(cmd, shell=True, cwd=self.model_dir):
            print(path, end="")
            file = list((self.model_dir / "results").glob("*.*"))
            if len(file) == 1:
                data = []

                with file[0].open() as f:
                    for line in f.readlines():
                        value = json.loads(line)
                        if value["mode"] == "eval":
                            data.append(value)

                pd.DataFrame(data).set_index("epoch").to_csv(self.training)

    @functools.cached_property
    def calc(self):
        """Return the ASE calculator."""
        return MACECalculator(
            model_path=self.model_dir / "MACE_model.model", device=self.device
        )

    def predict(self, atoms: typing.List[ase.Atoms]) -> Prediction:
        """Predict energy, forces and stresses.

        based on what was used to train on with trained GAP potential.

        Parameters
        ----------
        atoms: List[Atoms]
            contains atoms objects to validate

        Returns
        -------
        Prediction
            dataclass contains predicted energy, force and stresses
        """
        potential = self.calc
        validation_energy = []
        validation_forces = []
        validation_stresses = []
        for configuration in tqdm.tqdm(atoms, ncols=70):
            configuration.calc = potential
            if self.use_energy:
                validation_energy.append(configuration.get_potential_energy())
            if self.use_forces:
                validation_forces.append(configuration.get_forces())
            if self.use_stresses:
                validation_stresses.append(configuration.get_stress())

        return Prediction(
            energy=np.array(validation_energy),
            forces=np.array(validation_forces),
            # quippy potential output needs minus sign here
            stresses=np.array(validation_stresses),
            n_atoms=len(atoms[0]),
        )
