import logging
from pathlib import Path

import ase
import h5py
import numpy as np
import znh5md
import zntrack
from numpy.random import default_rng
from scipy.spatial.transform import Rotation

import ipsuite as ips
from ipsuite import base

log = logging.getLogger(__name__)


class Bootstrap(base.IPSNode):
    """Base class for dataset bootstrapping with structural modifications.

    Parameters
    ----------
    data : list[ase.Atoms]
        Input atomic configurations to bootstrap from.
    data_id : int, default=-1
        Index of the configuration to use from the data list.
    n_configurations : int
        Number of new configurations to generate.
    maximum : float
        Maximum displacement/rotation/translation magnitude.
    include_original : bool, default=True
        Whether to include the original configuration in output.
    seed : int, default=0
        Random seed for reproducible generation.

    Attributes
    ----------
    frames : list[ase.Atoms]
        Generated atomic configurations after bootstrapping.
    frames_path : Path
        Path to the HDF5 file storing the generated configurations.
    """

    data: list[ase.Atoms] = zntrack.deps()
    data_id: int = zntrack.params(-1)
    n_configurations: int = zntrack.params()
    maximum: float = zntrack.params()
    include_original: bool = zntrack.params(True)
    seed: int = zntrack.params(0)

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def run(self) -> None:
        atoms = self.data[self.data_id]
        rng = default_rng(self.seed)
        atoms_list = self.bootstrap_configs(
            atoms,
            rng,
        )
        # Store frames in HDF5 file
        db = znh5md.IO(self.frames_path)
        db.extend(atoms_list)

    def bootstrap_configs(self, atoms: ase.Atoms, rng):
        raise NotImplementedError

    @property
    def frames(self) -> list[ase.Atoms]:
        """Load and return the generated atomic configurations."""
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]


class RattleAtoms(Bootstrap):
    """Generate configurations with randomly displaced atomic positions.

    Creates new configurations by applying random displacements to each atom's
    position.

    Parameters
    ----------
    data : list[ase.Atoms]
        Input atomic configurations to modify.
    data_id : int, default=-1
        Index of the configuration to use from the data list.
    n_configurations : int
        Number of rattled configurations to generate.
    maximum : float
        Maximum displacement magnitude (Ångström) for each atom.
    include_original : bool, default=True
        Whether to include the original configuration in output.
    seed : int, default=0
        Random seed for reproducible displacement generation.

    Attributes
    ----------
    frames : list[ase.Atoms]
        Generated configurations with rattled atomic positions.
    frames_path : Path
        Path to the HDF5 file storing the generated configurations.

    Examples
    --------
    >>> with project:
    ...     data = ips.AddData(file="ethanol.xyz")
    ...     rattled = ips.RattleAtoms(data=data.frames, n_configurations=5, maximum=0.1)
    >>> project.repro()
    >>> print(f"Generated {len(rattled.frames)} rattled configurations")
    Generated 6 rattled configurations
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
    """Generate configurations with randomly translated molecular units.

    Creates new configurations by applying random translations to individual
    molecular units while preserving their internal structure. Requires the
    presence of distinct molecular entities in the system.

    Parameters
    ----------
    data : list[ase.Atoms]
        Input atomic configurations containing molecular units.
    data_id : int, default=-1
        Index of the configuration to use from the data list.
    n_configurations : int
        Number of configurations with translated molecules to generate.
    maximum : float
        Maximum translation distance (Ångström) for each molecule.
    include_original : bool, default=True
        Whether to include the original configuration in output.
    seed : int, default=0
        Random seed for reproducible translation generation.

    Attributes
    ----------
    frames : list[ase.Atoms]
        Generated configurations with translated molecular units.
    frames_path : Path
        Path to the HDF5 file storing the generated configurations.

    Examples
    --------
    >>> with project:
    ...     data = ips.AddData(file="ethanol.xyz")
    ...     translated = ips.TranslateMolecules(data=data.frames, n_configurations=5,
    ...                                        maximum=0.5)
    >>> project.repro()
    >>> print(f"Generated {len(translated.frames)} configurations with translated "
    ...       f"molecules")
    Generated 6 configurations with translated molecules
    """

    def bootstrap_configs(self, atoms, rng):
        if self.include_original:
            atoms_list = [atoms]
        else:
            atoms_list = []

        mapping = ips.BarycenterMapping()

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
    """Generate configurations with randomly rotated molecular units.

    Creates new configurations by applying random rotations to individual
    molecular units around their barycenter while preserving internal bond
    structures. Requires distinct molecular entities in the system.

    Parameters
    ----------
    data : list[ase.Atoms]
        Input atomic configurations containing molecular units.
    data_id : int, default=-1
        Index of the configuration to use from the data list.
    n_configurations : int
        Number of configurations with rotated molecules to generate.
    maximum : float
        Maximum rotation angle (radians) for each molecule.
    include_original : bool, default=True
        Whether to include the original configuration in output.
    seed : int, default=0
        Random seed for reproducible rotation generation.

    Attributes
    ----------
    frames : list[ase.Atoms]
        Generated configurations with rotated molecular units.
    frames_path : Path
        Path to the HDF5 file storing the generated configurations.

    Examples
    --------
    >>> with project:
    ...     data = ips.AddData(file="ethanol.xyz")
    ...     rotated = ips.RotateMolecules(data=data.frames, n_configurations=5,
    ...                                  maximum=3.14159)
    >>> project.repro()
    >>> print(f"Generated {len(rotated.frames)} configurations with rotated molecules")
    Generated 6 configurations with rotated molecules
    """

    def bootstrap_configs(self, atoms, rng):
        if self.include_original:
            atoms_list = [atoms]
        else:
            atoms_list = []

        if self.maximum > 2 * np.pi:
            log.warning("Setting maximum to 2 Pi.")

        mapping = ips.BarycenterMapping()

        _, molecules = mapping.forward_mapping(atoms.copy())
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
