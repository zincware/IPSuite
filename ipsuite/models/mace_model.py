"""MACE model module."""

import json
import logging
import pathlib
import subprocess

import pandas as pd
import torch
import yaml
import zntrack
from mace.calculators import MACECalculator

from ipsuite import utils
from ipsuite.models import MLModel
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

    config: str = zntrack.dvc.deps("mace.yaml")
    config_kwargs: dict = zntrack.zn.params(None)
    device: str = zntrack.meta.Text(None)

    training: pathlib.Path = zntrack.dvc.plots(
        zntrack.nwd / "training.csv",
        template=STATIC_PATH / "y_log.json",
        x="epoch",
        y=["loss", "rmse_e_per_atom", "rmse_f"],
    )

    def _post_init_(self):
        self.data = utils.helpers.get_deps_if_node(self.data, "atoms")
        self.test_data = utils.helpers.get_deps_if_node(self.test_data, "atoms")

    def _post_load_(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def generate_config_file(self, file: str = "mace.yaml"):
        example = {
            "amsgrad": True,
            "batch_size": 5,
            "ema": True,
            "ema_decay": 0.99,
            "hidden_irreps": "128x0e + 128x1o",
            "max_num_epochs": 1000,
            "num_cutoff_basis": 5,
            "num_interactions": 2,
            "num_radial_basis": 8,
            "r_max": 5.0,
            "seed": 42,
            "start_swa": 1200,
            "swa": True,
            "E0s": "average",
        }
        pathlib.Path(file).write_text(yaml.safe_dump(example))

    def run(self):
        """Train a MACE model."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        cmd = """curl -sSL https://raw.githubusercontent.com/ACEsuit/mace/main/scripts/run_train.py | python - """  # noqa E501
        cmd += '--name="MACE_model" '
        cmd += f'--train_file="{self.train_data_file.resolve().as_posix()}" '
        cmd += "--valid_fraction=0.05 "
        cmd += f'--test_file="{self.test_data_file.resolve().as_posix()}" '
        cmd += f"--device={self.device} "

        config = yaml.safe_load(pathlib.Path(self.config).read_text())
        if self.config_kwargs:
            log.warning(
                f"Overwriting '{self.config}' with values from 'params.yaml':"
                f" {self.config_kwargs}"
            )
            config.update(self.config_kwargs)

        for key, val in config.items():
            if val is True:
                cmd += f"--{key} "
            elif val is False:
                pass
            else:
                cmd += f'--{key}="{val}" '

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

    def get_calculator(self, device=None, **kwargs):
        """Return the ASE calculator."""
        device = device or self.device
        return MACECalculator(
            model_path=self.model_dir / "MACE_model.model", device=self.device
        )
