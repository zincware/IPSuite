import typing

import ase


def get_energy_terms(atoms: ase.Atoms) -> typing.Tuple[float, float, float]:
    """Returns total, kinetic and potentials energy terms.
    Useful for seeing whether the total energy is conserved in an NVE simulation.

    Parameters
    ----------
    atoms: ase.Atoms
        Atoms objects for which energy will be calculated

    Returns
    -------
    etot: float
        total energy per atom
    ekin: float
        kinetic energy per atom
    epot: float
        potential energy per atom

    """
    epot = atoms.get_potential_energy() / len(atoms)
    ekin = atoms.get_kinetic_energy() / len(atoms)

    etot = epot + ekin

    return etot, ekin, epot
