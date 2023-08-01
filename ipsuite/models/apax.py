import logging
import pathlib
import shutil
import typing
from uuid import uuid4

from apax.md import ASECalculator
from apax.train.run import run as apax_run
import ase.io
import pandas as pd
import yaml
import zntrack.utils
from jax.config import config
from tqdm import tqdm
from zntrack import dvc, zn

from ipsuite import base, utils
from ipsuite.models.base import MLModel
from ipsuite.static_data import STATIC_PATH
from ipsuite.utils.ase_sim import freeze_copy_atoms
from ipsuite.utils.helpers import check_duplicate_keys

log = logging.getLogger(__name__)


class Apax(MLModel):
    """Class for the implementation of the apax model

    Attributes
    ----------
    config: str
        path to the apax config file
    validation_data: ase.Atoms
        atoms object with the validation data set
    model_directory: pathlib.Path
        model directory
    train_log_file: pathlib.Path
        log file directory
    train_data_file: pathlib.Path
        path to the training data
    validation_data_file: pathlib.Path
        path to the valdidation data
    """

    config: str = dvc.params("apax.yaml")
    validation_data = zn.deps()

    model_directory: pathlib.Path = dvc.outs(zntrack.nwd / "apax_model")
    train_log_file: pathlib.Path = dvc.outs(zntrack.nwd / "train.log")

    train_data_file: pathlib.Path = dvc.outs(zntrack.nwd / "train_atoms.extxyz")
    validation_data_file: pathlib.Path = dvc.outs(zntrack.nwd / "val_atoms.extxyz")

    jax_enable_x64: bool = zn.params(True)

    metrics_epoch = dvc.plots(
        zntrack.nwd / "log.csv",
        template=STATIC_PATH / "y_log.json",
        x="epoch",
        x_label="epochs",
        y="val_loss",
        y_label="validation loss",
    )
    metrics = zn.metrics()

    _parameter: dict = None

    def _post_init_(self):
        self.data = utils.helpers.get_deps_if_node(self.data, "atoms")
        self.validation_data = utils.helpers.get_deps_if_node(
            self.validation_data, "atoms"
        )

    def _handle_parameter_file(self):
        self._parameter = yaml.safe_load(pathlib.Path(self.config).read_text())

        custom_parameters = {
            "model_path": self.model_directory.as_posix(),
            "model_name": "",
            "train_data_path": self.train_data_file.as_posix(),
            "val_data_path": self.validation_data_file.as_posix(),
        }

        check_duplicate_keys(custom_parameters, self._parameter["data"], log)
        self._parameter["data"].update(custom_parameters)

    def train_model(self):
        """Train the model using `apax.train.run`"""
        apax_run(self._parameter, log_file=self.train_log_file)

    def move_metrics(self):
        """Move the metrics to the correct directories for DVC"""
        shutil.move(self.model_directory / self.metrics_epoch.name, self.metrics_epoch)

    def get_metrics_from_plots(self):
        """In addition to the plots write a model metric"""
        metrics_df = pd.read_csv(self.metrics_epoch)
        self.metrics = metrics_df.iloc[-1].to_dict()

    def run(self):
        """Primary method to run which executes all steps of the model training"""
        config.update("jax_enable_x64", self.jax_enable_x64)

        ase.io.write(self.train_data_file, self.data)
        ase.io.write(self.validation_data_file, self.validation_data)
        self._handle_parameter_file()

        self.train_model()
        self.move_metrics()
        self.get_metrics_from_plots()

        with pathlib.Path(self.train_log_file).open("a") as f:
            f.write("Training completed\n")

    def get_calculator(self, **kwargs):
        """Get a apax ase calculator"""

        self._handle_parameter_file()
        return ASECalculator(model_dir=self.model_directory)


class ApaxEnsemble(base.IPSNode):
    models: typing.List[Apax] = zntrack.zn.deps()

    def run(self) -> None:
        pass

    def get_calculator(self, **kwargs) -> ase.calculators.calculator.Calculator:
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """
        from apax.md import ASECalculator

        param_files = [m._parameter["data"]["model_path"] for m in self.models]

        calc = ASECalculator(param_files)
        return calc

