import logging
import pathlib
import shutil
import typing
from typing import Optional

import ase.io
import numpy as np
import pandas as pd
import yaml
import zntrack.utils
from apax.bal import kernel_selection
from apax.md import ASECalculator
from apax.md.function_transformations import available_transformations
from apax.train.run import run as apax_run
from jax.config import config
from matplotlib import pyplot as plt
from zntrack import dvc, zn

from ipsuite import base, utils
from ipsuite.analysis.ensemble import plot_with_uncertainty
from ipsuite.configuration_selection.base import BatchConfigurationSelection
from ipsuite.models.base import MLModel
from ipsuite.static_data import STATIC_PATH
from ipsuite.utils.combine import get_flat_data_from_dict
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
    validation_data = zntrack.deps()
    model: Optional[MLModel] = zntrack.deps(None)

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
        self._handle_parameter_file()

    def _post_load_(self) -> None:
        self._handle_parameter_file()

    def _handle_parameter_file(self):
        self._parameter = yaml.safe_load(pathlib.Path(self.config).read_text())

        custom_parameters = {
            "directory": self.model_directory.as_posix(),
            "experiment": "",
            "train_data_path": self.train_data_file.as_posix(),
            "val_data_path": self.validation_data_file.as_posix(),
        }

        if self.model is not None:
            param_files = self.model._parameter["data"]["model_path"]
            base_path = {"base_model_checkpoint": param_files + "/best"}

            self._parameter["checkpoints"].update(base_path)

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

        self.train_model()
        self.move_metrics()
        self.get_metrics_from_plots()

        with pathlib.Path(self.train_log_file).open("a") as f:
            f.write("Training completed\n")

    def get_calculator(self, **kwargs):
        """Get an apax ase calculator"""

        return ASECalculator(model_dir=self.model_directory)


class ApaxEnsemble(base.IPSNode):
    """Parallel apax model ensemble in ASE.

    Attributes
    ----------
    models: list
        List of `ApaxModel` nodes to ensemble.
    nl_skin: float
        Neighborlist skin.
    transformations: dict
        Key-parameter dict with function transformations applied
        to the model function within the ASE calculator.
        See the apax documentation for available methods.
    """

    models: typing.List[Apax] = zntrack.deps()
    nl_skin: float = zntrack.params(0.5)
    transformations: typing.Dict[str, dict] = zntrack.params(None)

    def run(self) -> None:
        pass

    def get_calculator(self, **kwargs) -> ase.calculators.calculator.Calculator:
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """

        param_files = [m._parameter["data"]["directory"] for m in self.models]

        transformations = []
        if self.transformations:
            for transform, params in self.transformations.items():
                transformations.append(available_transformations[transform](**params))

        calc = ASECalculator(
            param_files,
            dr=self.nl_skin,
            transformations=transformations,
        )
        return calc


class BatchKernelSelection(BatchConfigurationSelection):
    """Interface to the batch active learning methods implemented in apax.
    Check the apax documentation for a list and explanation of implemented properties.

    Attributes
    ----------
    models: Union[Apax, List[Apax]]
        One or more Apax models to construct a feature map from.
    base_feature_map: dict
        Name and parameters for the feature map transformation.
    selection_method: str
        Name of the selection method to be used. Choose from:
        ["max_dist", ]
    n_configurations: int
        Number of samples to be selected.
    processing_batch_size: int
        Number of samples to be processed in parallel.
        Does not affect the result, just the speed of computing features.
    """

    models: typing.List[Apax] = zntrack.deps()
    base_feature_map: dict = zntrack.params({"name": "ll_grad", "layer_name": "dense_2"})
    selection_method: str = zntrack.params("max_dist")
    n_configurations: str = zntrack.params()
    processing_batch_size: str = zntrack.meta.Text(64)
    img_selection = zntrack.outs_path(zntrack.nwd / "selection.png")

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        if isinstance(self.models, list):
            param_files = [m._parameter["data"]["directory"] for m in self.models]
        else:
            param_files = self.models._parameter["data"]["directory"]

        if isinstance(self.train_data, dict):
            self.train_data = get_flat_data_from_dict(self.train_data)

        selected = kernel_selection(
            param_files,
            self.train_data,
            atoms_lst,
            self.base_feature_map,
            self.selection_method,
            selection_batch_size=self.n_configurations,
            processing_batch_size=self.processing_batch_size,
        )
        self._get_plot(atoms_lst, selected)

        return list(selected)

    def _get_plot(self, atoms_lst: typing.List[ase.Atoms], indices: typing.List[int]):
        energies = np.array([atoms.calc.results["energy"] for atoms in atoms_lst])

        if "energy_uncertainty" in atoms_lst[0].calc.results.keys():
            uncertainty = np.array(
                [atoms.calc.results["energy_uncertainty"] for atoms in atoms_lst]
            )
            fig, ax, _ = plot_with_uncertainty(
                {"mean": energies, "std": uncertainty},
                ylabel="energy",
                xlabel="configuration",
            )
        else:
            fig, ax = plt.subplots()
            ax.plot(energies, label="energy")
            ax.set_ylabel("energy")
            ax.set_xlabel("configuration")

        ax.plot(indices, energies[indices], "x", color="red")

        fig.savefig(self.img_selection, bbox_inches="tight")
