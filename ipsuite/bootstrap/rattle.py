import ase
import zntrack
from numpy.random import default_rng

from ipsuite import base


def create_initial_configurations(
    atoms: ase.Atoms, displacement_range: float, n_configs: int, rng
):
    atoms_list = [atoms]
    for _ in range(n_configs):
        new_atoms = atoms.copy()
        displacement = rng.uniform(
            -displacement_range, displacement_range, size=new_atoms.positions.shape
        )
        new_atoms.positions += displacement
        atoms_list.append(new_atoms)
    return atoms_list


class RattleAtoms(base.ProcessSingleAtom):
    """Create randomly displaced versions of a particular atomic configuration.
    Useful for learning on the fly applications.
    Only the displaced structures are returned.

    Attributes
    ----------
    n_configs: int
        Number of displaced configurations.
    displacement_range: float
        Bounds for uniform distribution from which displacments are drawn.
    seed: int
        Random seed.

    """

    n_configs: int = zntrack.zn.params()
    displacement_range: float = zntrack.zn.params(0.1)
    seed: int = zntrack.zn.params(0)

    def run(self) -> None:
        atoms = self.get_data()
        rng = default_rng(self.seed)
        atoms_list = create_initial_configurations(
            atoms,
            self.displacement_range,
            self.n_configs,
            rng,
        )

        self.atoms = atoms_list
