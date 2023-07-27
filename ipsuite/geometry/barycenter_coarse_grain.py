import numpy as np
from ase import Atoms


def coarse_grain_to_barycenter(molecules):
    coms = np.zeros(shape=(len(molecules), 3))

    # TODO: use rdkit instead to identify identical molecules

    identifier = {}
    atomic_numbers = []

    for ii, mol in enumerate(molecules):
        com = np.mean(mol.positions, axis=0)
        coms[ii] = com
        try:
            atomic_numbers.append(identifier[mol.get_chemical_formula()])
        except KeyError:
            identifier[mol.get_chemical_formula()] = len(identifier) + 1
            atomic_numbers.append(identifier[mol.get_chemical_formula()])

    cg_positions = np.stack(coms, axis=0)
    cg_atoms = Atoms(atomic_numbers, positions=cg_positions, cell=molecules[0].cell)

    return cg_atoms


def barycenter_backmapping(cg_atoms, molecules):
    coms = cg_atoms.positions
    shifted_molecules = []
    for ii, mol in enumerate(molecules):
        shifted_mol = mol.copy()
        mol_com = np.mean(mol.positions, axis=0)
        shifted_mol.positions += coms[ii] - mol_com
        shifted_molecules.append(shifted_mol)

    atoms = shifted_molecules[0]
    for i in range(1, len(shifted_molecules)):
        atoms += shifted_molecules[i]
    atoms.cell = cg_atoms.cell
    return atoms
