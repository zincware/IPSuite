import numpy as np


def compute_trans_forces(mol):
    """Compute translational forces of a molecule."""

    all_forces = np.sum(mol.get_forces(), axis=0)
    masses = mol.get_masses()
    mol_mas = np.sum(masses)
    return (masses / mol_mas)[:, None] * all_forces


def compute_intertia_tensor(centered_positions, masses):
    r_sq = np.linalg.norm(centered_positions, ord=2, axis=1) ** 2 * masses
    r_sq = np.sum(r_sq)
    A = np.diag(np.full((3,), r_sq))
    mr_k = centered_positions * masses[:, None]
    B = np.einsum("ki, kj -> ij", centered_positions, mr_k)

    I_ab = A - B
    return I_ab


def compute_rot_forces(mol):
    mol_positions = mol.get_positions()
    mol_positions -= mol.get_center_of_mass()
    masses = mol.get_masses()

    f_x_r = np.sum(np.cross(mol.get_forces(), mol_positions), axis=0)
    I_ab = compute_intertia_tensor(mol_positions, masses)
    I_ab_inv = np.linalg.inv(I_ab)

    mi_ri = masses[:, None] * mol_positions
    return np.cross(mi_ri, (I_ab_inv @ f_x_r))


def force_decomposition(atom, mapping):
    # TODO we should only need to do the mapping once
    _, molecules = mapping.forward_mapping(atom)
    full_forces = np.zeros_like(atom.positions)
    atom_trans_forces = np.zeros_like(atom.positions)
    atom_rot_forces = np.zeros_like(atom.positions)
    total_n_atoms = 0

    for molecule in molecules:
        n_atoms = len(molecule)
        full_forces[total_n_atoms : total_n_atoms + n_atoms] = molecule.get_forces()
        atom_trans_forces[total_n_atoms : total_n_atoms + n_atoms] = compute_trans_forces(
            molecule
        )
        atom_rot_forces[total_n_atoms : total_n_atoms + n_atoms] = compute_rot_forces(
            molecule
        )
        total_n_atoms += n_atoms

    atom_vib_forces = full_forces - atom_trans_forces - atom_rot_forces

    return atom_trans_forces, atom_rot_forces, atom_vib_forces
