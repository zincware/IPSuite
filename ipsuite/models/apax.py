import logging
import pathlib
import shutil

import ase.io
import pandas as pd
import yaml
import zntrack.utils
from jax.config import config
from zntrack import dvc, zn

from ipsuite import utils
from ipsuite.models.base import MLModel
from ipsuite.utils.helpers import check_duplicate_keys

log = logging.getLogger(__name__)


class Apax(MLModel):
    """Class for the implementation of the apax model

    Attributes
    ----------
    parameter : dict
        dict of the model parameter
    parameter_file: str
        path to the model parameter file
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

    parameter: dict = zn.params(None)
    parameter_file: str = dvc.params(None)
    validation_data = zn.deps()

    model_directory: pathlib.Path = dvc.outs(zntrack.nwd / "apax_model")
    train_log_file: pathlib.Path = dvc.outs(zntrack.nwd / "train.log")

    train_data_file: pathlib.Path = dvc.outs(zntrack.nwd / "train_atoms.extxyz")
    validation_data_file: pathlib.Path = dvc.outs(zntrack.nwd / "val_atoms.extxyz")

    jax_enable_x64: bool = zn.params(True)

    metrics_epoch = dvc.plots(
        zntrack.nwd / "log.csv",
        # x="epoch",
        # x_label="epochs",
        # y="val_loss",
        # y_label="validation loss",
    )
    metrics = zn.metrics()

    def _post_init_(self):
        if not self.state.loaded:
            dict_supplied = self.parameter is None
            file_supplied = self.parameter_file is None

            if dict_supplied and file_supplied:
                raise TypeError("Please specify either an input dict or file")
            elif not dict_supplied and not file_supplied:
                raise TypeError(
                    "Can not train apax model without a parameter dict or file"
                )
            else:
                log.info(
                    "Please keep track of the parameter file with git, just like the"
                    " params.yaml"
                )

        self.data = utils.helpers.get_deps_if_node(self.data, "atoms")
        self.validation_data = utils.helpers.get_deps_if_node(
            self.validation_data, "atoms"
        )

    def _handle_parameter_file(self):
        if self.parameter_file:
            parameter_file_content = pathlib.Path(self.parameter_file).read_text()
            self.parameter = yaml.safe_load(parameter_file_content)

        custom_parameters = {
            "model_path": self.model_directory.as_posix(),
            "model_name": "",
            "train_data_path": self.train_data_file.as_posix(),
            "val_data_path": self.validation_data_file.as_posix(),
        }

        check_duplicate_keys(custom_parameters, self.parameter["data"], log)
        self.parameter["data"].update(custom_parameters)

    def train_model(self):
        """Train the model using `apax.train.run`"""
        from apax.train.run import run as apax_run

        apax_run(self.parameter, log_file=self.train_log_file)

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

    def get_calculator(self):
        """Get a apax ase calculator"""
        from apax.md import ASECalculator

        self._handle_parameter_file()
        return ASECalculator(model_dir=self.model_directory)