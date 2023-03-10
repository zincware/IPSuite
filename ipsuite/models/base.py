"""Base class for all MLModel Implementations."""
import contextlib
import dataclasses
import pathlib
import typing

import ase.calculators.calculator
import ase.io
import numpy as np
import znjson
import zntrack

from ipsuite import base


@dataclasses.dataclass
class Prediction:
    """Dataclass to store prediction values of the model.

    Attributes
    ----------
    energy: np.ndarray
        energy prediction with shape (b,)
    forces: np.ndarray
        force prediction with shape (b, atoms, 3)
    virials: np.ndarray
        virials prediction with shape (b, 6)
    n_atoms: int
        number of atoms
    """

    energy: np.ndarray = None
    forces: np.ndarray = None
    stresses: np.ndarray = None
    n_atoms: int = None

    def update_n_atoms(self):
        """Update the number of atoms if they are not provided.

        n_atoms has a higher priority than the shape of the forces
        """
        if self.n_atoms is None:
            with contextlib.suppress(AttributeError):
                self.n_atoms = self.forces.shape[1]

    @property
    def asdict(self) -> dict:
        """Convert dataclass to dict."""
        return dataclasses.asdict(self)


class PredictionConverter(znjson.ConverterBase):
    """Converter for Prediction dataclass."""

    instance = Prediction
    representation = "ipsuite.models.Prediction"

    def encode(self, value: Prediction):
        return value.asdict

    def decode(self, value: str):
        return Prediction(**value)


znjson.config.register(PredictionConverter)


class MLModel(base.AnalyseAtoms):
    """Parent class for all MLModel Implementations."""

    _name_ = "MLModel"

    use_energy: bool = zntrack.zn.params(True)
    use_forces: bool = zntrack.zn.params(True)
    use_stresses: bool = zntrack.zn.params(False)

    @property
    def calc(self) -> ase.calculators.calculator.Calculator:
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """
        raise NotImplementedError

    def predict(self, atoms: typing.List[ase.Atoms]) -> Prediction:
        """Predict energy, forces and stresses.

        based on what was used to train for given atoms objects.

        Parameters
        ----------
        atoms: typing.List[ase.Atoms]
            list of atoms objects to predict on

        Returns
        -------
        Prediction: Prediction
            dataclass which contains the predicted energy, forces and stresses
        """
        raise NotImplementedError

    @property
    def lammps_pair_style(self) -> str:
        """Get the lammps pair_style command attribute.

        See https://docs.lammps.org/pair_style.html
        Returns
        -------
        This can be e.g. 'quip' or 'allegro'
        """
        raise NotImplementedError

    @property
    def lammps_pair_coeff(self) -> typing.List[str]:
        """Get the lammps pair_coeff command attribute.

        See https://docs.lammps.org/pair_coeff.html

        Returns
        -------
        a list of pair_coeff attributes.
        E.g. [' * * model/deployed_model.pth B C F H N']

        """
        raise NotImplementedError

    @staticmethod
    def write_data_to_file(file, atoms_list: typing.List[ase.Atoms]):
        """Save e.g. train / test data to a file.

        Parameters
        ----------
        file: str|Path
            path to save to.
        atoms_list: list[Atoms]
            atoms that should be saved.
        """
        pathlib.Path(file).parent.mkdir(parents=True, exist_ok=True)
        for atom in atoms_list:
            atom.wrap()
        ase.io.write(file, images=atoms_list)
