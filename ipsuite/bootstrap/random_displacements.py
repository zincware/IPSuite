import logging

import ase
import numpy as np
import tqdm
import zntrack
from numpy.random import default_rng
from scipy.spatial.transform import Rotation

import ipsuite as ips
from ipsuite import base
from ipsuite.utils.ase_sim import freeze_copy_atoms

log = logging.getLogger(__name__)


class Bootstrap(base.ProcessSingleAtom):
    """Baseclass for dataset bootstrapping.
    Derived classes need to implement a `bootstrap_config` method.

    Attributes
    ----------
    n_configurations: int
        Number of displaced configurations.
    maximum: float
        Bounds for uniform distribution from which displacements are drawn.
    include_original: bool
        Whether or not to include the original configuration in `self.atoms`.
    seed: int
        Random seed.
    model: IPSNode
        Any IPSNode that provides a `get_calculator` method to
        label the bootstrapped configurations.
    """

    n_configurations: int = zntrack.zn.params()
    maximum: float = zntrack.zn.params()
    include_original: bool = zntrack.zn.params(True)
    seed: int = zntrack.zn.params(0)
    model: base.IPSNode = zntrack.deps(None)
    model_outs = zntrack.dvc.outs(zntrack.nwd / "model_outs")

    def run(self) -> None:
        atoms = self.get_data()
        rng = default_rng(self.seed)
        atoms_list = self.bootstrap_configs(
            atoms,
            rng,
        )

        self.atoms = atoms_list

        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        if self.model is not None:
            calculator = self.model.get_calculator(directory=self.model_outs)
            for atoms in tqdm.tqdm(self.atoms, ncols=120, desc="Applying model"):
                atoms.calc = calculator
                atoms.get_potential_energy()
            self.atoms = freeze_copy_atoms(self.atoms)

    def bootstrap_configs(sefl, atoms: ase.Atoms, rng):
        raise NotImplementedError


class RattleAtoms(Bootstrap):
    """Create randomly displaced versions of a particular atomic configuration.
    Useful for learning on the fly applications.
    `maximum` specifies the maximal atomic displacement.
    """

    def bootstrap_configs(self, atoms, rng):
        if self.include_original:
            atoms_list = [atoms]
        else:
            atoms_list = []

        for _ in range(self.n_configurations):
            new_atoms = atoms.copy()
            displacement = rng.uniform(
                -self.maximum, self.maximum, size=new_atoms.positions.shape
            )
            new_atoms.positions += displacement
            atoms_list.append(new_atoms)
        return atoms_list


class TranslateMolecules(Bootstrap):
    """Create versions of a particular atomic configuration with
    randomly displaced molecular units.
    Only applicable if there are covalent units present in the system.
    Useful for learning on the fly applications.
    `maximum` specifies the maximal molecular displacement.
    """

    def bootstrap_configs(self, atoms, rng):
        if self.include_original:
            atoms_list = [atoms]
        else:
            atoms_list = []

        mapping = ips.geometry.BarycenterMapping(data=None)

        _, molecules = mapping.forward_mapping(atoms)
        for _ in range(self.n_configurations):
            molecule_lst = []
            for molecule in molecules:
                mol = molecule.copy()

                displacement = rng.uniform(-self.maximum, self.maximum, size=(3,))
                mol.positions += displacement

                molecule_lst.append(mol)

            new_atoms = molecule_lst[0]
            for i in range(1, len(molecule_lst)):
                new_atoms += molecule_lst[i]
            atoms_list.append(new_atoms)

        return atoms_list


class RotateMolecules(Bootstrap):
    """Create versions of a particular atomic configuration with
    randomly rotated molecular units.
    Only applicable if there are covalent units present in the system.
    Useful for learning on the fly applications.
    `maximum` specifies the maximal molecular rotation in degrees.
    """

    def bootstrap_configs(self, atoms, rng):
        if self.include_original:
            atoms_list = [atoms]
        else:
            atoms_list = []

        if self.maximum > 2 * np.pi:
            log.warning("Setting maximum to 2 Pi.")

        mapping = ips.geometry.BarycenterMapping(data=None)

        _, molecules = mapping.forward_mapping(atoms)
        for _ in range(self.n_configurations):
            molecule_lst = []
            for molecule in molecules:
                mol = molecule.copy()

                euler_angles = rng.uniform(0, self.maximum, size=(3,))
                rotate = Rotation.from_euler("zyx", euler_angles, degrees=False)
                pos = mol.positions
                barycenter = np.mean(pos, axis=0)
                pos -= barycenter
                pos_rotated = rotate.apply(pos)
                mol.positions = barycenter + pos_rotated

                molecule_lst.append(mol)

            new_atoms = molecule_lst[0]
            for i in range(1, len(molecule_lst)):
                new_atoms += molecule_lst[i]
            atoms_list.append(new_atoms)

        return atoms_list
