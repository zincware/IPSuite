"""Utils that help running simulations with ASE."""

import functools
import typing

import ase
import numpy as np
from ase import units
from ase.calculators.singlepoint import SinglePointCalculator


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


@functools.singledispatch
def freeze_copy_atoms(atoms) -> ase.Atoms:
    # TODO can we add the name of the original calculator?
    result = atoms.copy()
    result.calc = SinglePointCalculator(result)
    result.calc.results.update(atoms.calc.results)
    return result


@freeze_copy_atoms.register
def _(atoms: list) -> list[ase.Atoms]:
    return [freeze_copy_atoms(x) for x in atoms]


def get_box_from_density(
    data: list[list[ase.Atoms]], count: list[int], density: float
) -> list[float]:
    """Get the box size from the molar volume.

    Attributes
    ----------
    data: list[list[ase.Atoms]]
        List of list of atoms objects. The last atoms object is used to compute the
        molar volume.
    count: list[int]
        Number of molecules for each entry in data.
    density: float
        Density of the system in kg/m^3.
    """
    molar_mass = [sum(atoms[0].get_masses()) * count for atoms, count in zip(data, count)]
    molar_mass = sum(molar_mass)  #  g / mol
    molar_volume = molar_mass / density / 1000  # m^3 / mol

    # convert to particles / A^3
    volume = molar_volume * (ase.units.m**3) / ase.units.mol

    box = [volume ** (1 / 3) for _ in range(3)]
    return box


def get_density_from_atoms(atoms: ase.Atoms) -> float:
    """Compute the density of the atoms in kg/m3."""
    volume = atoms.get_volume()
    molar_volume = volume / (units.m**3 / units.mol)
    return atoms.get_masses().sum() / 1000 / molar_volume
