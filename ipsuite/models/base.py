"""Base class for all MLModel Implementations."""
import pathlib
import typing

import ase.calculators.calculator
import ase.io
import tqdm
import zntrack

from ipsuite import base
from ipsuite.utils.ase_sim import freeze_copy_atoms


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

    def predict(self, atoms_list: typing.List[ase.Atoms]) -> typing.List[ase.Atoms]:
        """Predict energy, forces and stresses.

        based on what was used to train for given atoms objects.

        Parameters
        ----------
        atoms_list: typing.List[ase.Atoms]
            list of atoms objects to predict on

        Returns
        -------
        Prediction: typing.List[ase.Atoms]
            Atoms with updated calculators
        """
        calc = self.calc
        results = []
        for atoms in tqdm.tqdm(atoms_list, ncols=120):
            atoms.calc = calc
            atoms.get_potential_energy()
            results.append(freeze_copy_atoms(atoms))
        return results

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
