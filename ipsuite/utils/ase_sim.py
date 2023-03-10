"""Utils that help running simulations with ASE."""
import typing

import ase
import numpy as np
from ase import units


def get_energy(atoms: ase.Atoms) -> typing.Tuple[float, float]:
    """Compute the temperature and the total energy.

    Parameters
    ----------
    atoms: ase.Atoms
        Atoms objects for which energy will be calculated

    Returns
    -------
    temperature: float
        temperature of the system
    np.squeeze(total): float
        total energy of the system

    """
    epot = atoms.get_potential_energy() / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)

    temperature = ekin / (1.5 * units.kB)
    total = epot + ekin

    return temperature, np.squeeze(total).item()


def get_desc(temperature: float, total_energy: float):
    """TQDM description."""
    return f"Temp: {temperature:.3f} K \t Energy {total_energy:.3f} eV - (TQDM in fs)"
