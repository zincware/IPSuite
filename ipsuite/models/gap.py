"""Description.

GAP module to perform training using the Gaussian process
regression from Gabor Csanyi.
"""
import dataclasses
import importlib.resources as pkg_resources
import logging
import os
import subprocess as sp
import typing
from collections import OrderedDict
from pathlib import Path
from typing import List

import ase
import ase.io
import numpy as np
import quippy.potential
import xmltodict
import znflow
import zntrack
from ase import Atoms
from jinja2 import Template
from tqdm import tqdm
from znjson import ConverterBase, config

from ipsuite import static_data, utils
from ipsuite.models import MLModel, Prediction

log = logging.getLogger(__name__)


@dataclasses.dataclass()
class SOAP:
    """Dataclass to store SOAP parameters used by gap model.

    Attributes
    ----------
    n_max: int
        number of radial basis functions used for SOAP descriptor
    l_max: int
        number of angular basis functions used for SOAP descriptor
    cutoff: float
        distance cutoff within kernel, in Angstrom
    sigma: float
        Gaussian smearing width of atom density for SOAP, in Angstrom
    zeta: float
        power kernel is raised to - together with dot_product gives a polynomial kernel
    cutoff_transition_width: float
        distance across which kernel is smoothly taken to zero, in Angstrom
    delta: float
        scaling of kernel, per descriptor, here for SOAP it is per atom, in eV
    covariance_type: str
        form of kernel
    n_sparse: int
        number of representative points
    sparse_method: str
        choice of representative points, default is CUR decomposition of descriptor matrix
    add_species: bool
        if True: take the species into account, so it will generate more GAPs
        automatically
    """

    n_max: int = 7
    l_max: int = 7
    cutoff: float = 6.0
    sigma: float = 0.5
    zeta: float = 4.0
    cutoff_transition_width: float = 0.5
    delta: float = 1.0
    covariance_type: str = "dot_product"
    n_sparse: int = 50
    sparse_method: str = "CUR_POINTS"
    add_species: bool = True


@dataclasses.dataclass()
class DistanceNb:
    """Dataclass to store DistanceNb parameter used by gap model.

    Attributes
    ----------
    order: int
        order of descriptor, default 2 -> descriptor is 2-body i.e. pair-potential
    cutoff: float
        distance cutoff in the kernel, in Angstrom
    covariance_type: str
        form of kernel, default: squared exponential (Gaussian)
    delta: float
        scaling of kernel, per descriptor, here it is per atom pair, in eV
    sparse_method: str
        method to choose representative points
    add_species: bool
        if True: take the species into account, so it will generate more GAPs
        automatically
    n_sparse: int
        number of representative points
    theta_uniform: float
        length scale in Gaussian kernel in Angstrom
    """

    order: int = 2
    cutoff: float = 6.0
    covariance_type: str = "ard_se"
    delta: float = 0.1
    sparse_method: str = "CUR_POINTS"
    add_species: bool = True
    n_sparse: int = 50
    theta_uniform: float = 1.0


@dataclasses.dataclass()
class GapParameter:
    """Dataclass to store general gap parameter for the gap model.

    Attributes
    ----------
    default_sigma: str
        default regularisation corresponding to energy, force, virial, hessian
    sparse_jitter: float
        extra diagonal regulariser
    """

    e0_method: str = "average"
    default_sigma: str = "{0.0001 0.01 0.01 0.01}"
    sparse_jitter: float = 1.0e-10


class SOAPConverter(ConverterBase):
    """Converter for SOAP dataclass."""

    level = 100
    representation = "soap_dataclass"
    instance = SOAP

    def encode(self, obj: SOAP) -> dict:
        """Encode SOAP dataclass."""
        return dataclasses.asdict(obj)

    def decode(self, value: dict) -> SOAP:
        """Decode SOAP dataclass."""
        return SOAP(**value)


class DistanceNbConverter(ConverterBase):
    """Converter for DistanceNb dataclass."""

    level = 100
    representation = "distanceNb_dataclass"
    instance = DistanceNb

    def encode(self, obj: DistanceNb) -> dict:
        """Encode DistanceNb dataclass."""
        return dataclasses.asdict(obj)

    def decode(self, value: dict) -> DistanceNb:
        """Decode DistanceNb dataclass."""
        return DistanceNb(**value)


class GapParameterConverter(ConverterBase):
    """Converter for GapParameter dataclass."""

    level = 100
    representation = "GapParameter_dataclass"
    instance = GapParameter

    def encode(self, obj: GapParameter) -> dict:
        """Encode GapParameter dataclass."""
        return dataclasses.asdict(obj)

    def decode(self, value: dict) -> GapParameter:
        """Decode GapParameter dataclass."""
        return GapParameter(**value)


config.register([SOAPConverter, DistanceNbConverter, GapParameterConverter])


class GAP(MLModel):
    """Class for the implementation of the GAP module.

    Attributes
    ----------
    soap: typing.Union[dict, SOAP]
        dictionary/dataclass containing soap parameter GAP will use
    distance_nb: typing.Union[dict, DistanceNb]
        dictionary/dataclass containing DistanceNb parameter GAP will use
    gap: typing.Union[dict, GapParameter]
        dictionary/dataclass containing GAP internal parameters
    use_energy: bool
        boolean parameter to set train on energy on or off
    use_forces: bool
        boolean parameter to set train on forces on or off
    use_stresses: bool
        boolean parameter to set train on virials on or off
    model_outputs: Path
        Path to save model files
    train_data: List[ase.Atoms]
        atoms object to train the model on
    """

    SOAP
    soap: typing.Union[dict, SOAP] = zntrack.zn.params(SOAP())
    # DistanceNb
    distance_nb: typing.Union[dict, DistanceNb] = zntrack.zn.params(DistanceNb())
    # GAP
    gap: typing.Union[dict, GapParameter] = zntrack.zn.params(GapParameter())
    # #
    model_directory: Path = zntrack.dvc.outs(zntrack.nwd / "model")
    train_data_file: Path = zntrack.dvc.outs(zntrack.nwd / "train_atoms.extxyz")
    gap_input_script: Path = zntrack.dvc.outs(zntrack.nwd / "gap.input")

    #
    openblas_num_threads = zntrack.meta.Text(None)

    #
    def _post_init_(self):
        if self.openblas_num_threads is not None:
            os.environ["OPENBLAS_NUM_THREADS"] = self.openblas_num_threads

        if not self.state.loaded:
            if self.soap is None:
                self.soap = {}
            if self.gap is None:
                self.gap = {}
            if self.distance_nb is None:
                self.distance_nb = {}

            if isinstance(self.soap, dict):
                self.soap = SOAP(**self.soap)
            if isinstance(self.gap, dict):
                self.gap = GapParameter(**self.gap)
            if isinstance(self.distance_nb, dict):
                self.distance_nb = DistanceNb(**self.distance_nb)
            self.data = utils.helpers.get_deps_if_node(self.data, "atoms")

    def run(self):
        """Create output directory and train the model."""
        self.model_directory.mkdir(exist_ok=True, parents=True)
        self.train_model()

    @property
    def gap_input(self) -> (str, str):
        """Return the gap input string and the rendered template.

        Returns
        -------
        tuple:
            contains gap input string and rendered template
        """
        with znflow.disable_graph():
            general_params = {
                "atoms_filename": self.train_data_file.as_posix(),
                "gp_file": (self.model_directory / "model.xml").as_posix(),
                "gp_dump": (self.model_directory / "GAP_dump.txt").as_posix(),
                "use_energy": self.use_energy,
                "use_forces": self.use_forces,
                "use_stresses": self.use_stresses,
            }
            template_src = pkg_resources.read_text(static_data, "gap.jinja2")
            template = Template(template_src)
            rendered_template = template.render(
                distance_nb=self.distance_nb,
                soap=self.soap,
                gap_parameter=self.gap,
                general_params=general_params,
            )
            input_str = "".join(f" {line}" for line in rendered_template.split("\n"))
            return input_str, rendered_template, general_params["gp_file"]

    #
    def train_model(self):
        """Fit a GAP model.

        Returns
        -------
        Will save a GAP model to the output's directory.
        """
        log.info("---- Writing the GAP input file ----")
        self.write_data_to_file(file=self.train_data_file, atoms_list=self.data)
        log.info("--- Training model ---")
        self.gap_input_script.write_text(self.gap_input[0])

        sp.run([self.gap_input[0]], shell=True, check=True)

        if not Path(self.gap_input[2]).exists():
            raise RuntimeError(
                "gp_fit could not fit the given data. Check the GAP log files for more"
                " information"
            )

    #
    def predict(self, atoms: List[Atoms]) -> Prediction:
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
        for configuration in tqdm(atoms, ncols=70):
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

    #
    @property
    def calc(self):
        """Get the calculator object.

        Returns
        -------
        Calculator object to use e.g. for ase.
        """
        with znflow.disable_graph():
            file = Path(self.model_directory.resolve(), "model.xml")
            if file.exists():
                return quippy.potential.Potential(
                    "IP GAP",
                    param_filename=file.as_posix(),
                )
            else:
                raise FileNotFoundError(f"Could not load GAP potential from file {file}")

    #
    @property
    def lammps_pair_style(self) -> str:
        """Return the lammps pair style string."""
        return "quip"

    @property
    def lammps_pair_coeff(self) -> typing.List[str]:
        """Return the lammps pair coeff string."""
        with znflow.disable_graph():
            atomic_numbers = list(
                OrderedDict.fromkeys(ase.io.read(self.train_data_file).numbers)
            )
            with (self.model_directory.resolve() / "model.xml").open("r") as file:
                second_line = file.readlines()[1]
            content_as_dict = xmltodict.parse(second_line)
            gap_xml_label = f"{content_as_dict['Potential']['@label']}"
            coeff = (
                f'* * {self.model_directory.resolve() / "model.xml"} "Potential'
                f' xml_label={gap_xml_label}" {" ".join(map(str, atomic_numbers))}'
            )
            return coeff
