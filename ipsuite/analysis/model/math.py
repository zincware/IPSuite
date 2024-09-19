import numpy as np
import uncertainty_toolbox as uct


def compute_trans_forces(mol):
    """Compute translational forces of a molecule."""

    all_forces = np.sum(mol.calc.results["forces"], axis=0)
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

    f_x_r = np.sum(np.cross(mol.calc.results["forces"], mol_positions), axis=0)
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
        full_forces[total_n_atoms : total_n_atoms + n_atoms] = molecule.calc.results[
            "forces"
        ]
        atom_trans_forces[total_n_atoms : total_n_atoms + n_atoms] = compute_trans_forces(
            molecule
        )
        atom_rot_forces[total_n_atoms : total_n_atoms + n_atoms] = compute_rot_forces(
            molecule
        )
        total_n_atoms += n_atoms

    atom_vib_forces = full_forces - atom_trans_forces - atom_rot_forces

    return atom_trans_forces, atom_rot_forces, atom_vib_forces


def decompose_stress_tensor(stresses):
    hydrostatic_stresses = []
    deviatoric_stresses = []

    for stress in stresses:
        hydrostatic = np.mean(np.diag(stress))
        deviatoric = stress - (np.eye(3) * hydrostatic)

        hydrostatic_stresses.append(hydrostatic)
        deviatoric_stresses.append(deviatoric)

    hydrostatic_stresses = np.array(hydrostatic_stresses)
    deviatoric_stresses = np.array(deviatoric_stresses)
    return hydrostatic_stresses, deviatoric_stresses


def compute_rmse(errors):
    rmse = np.sqrt(np.mean(errors**2))
    return rmse


def nlls(pred, std, true):
    errors = np.abs(pred - true)
    nll = 0.5 * ((errors / std) ** 2 + np.log(2 * np.pi * std**2))
    return nll

def nll(pred, std, true):
    tmp = nlls(pred, std, true)
    return np.mean(tmp)


def comptue_rll(inputs, std, target):
    """ Compute relative log likelihood
    Adapted from https://github.com/bananenpampe/DPOSE
    """
    
    mse = np.mean((inputs - target)**2)
    uncertainty_estimate = (inputs - target)**2
    
    ll_best = nll(inputs, np.sqrt(uncertainty_estimate), target)
    
    ll_worst_case_best_RMSE = nll(inputs, np.sqrt(np.ones_like(std)*mse), target)
    
    ll_actual = nll(inputs, std, target)
    
    rll = (ll_actual - ll_worst_case_best_RMSE) / ( ll_best - ll_worst_case_best_RMSE)

    return rll * 100


def compute_uncertainty_metrics(pred, std, true):
    mace = uct.mean_absolute_calibration_error(pred, std, true)
    rmsce = uct.root_mean_squared_calibration_error(pred, std, true)
    miscal = uct.miscalibration_area(pred, std, true)
    nll = np.mean(nlls(pred, std, true))
    rll = comptue_rll(pred, std, true)

    metrics = {
        "mace": mace,
        "rmsce": rmsce,
        "miscal": miscal,
        "nll": nll,
        "rll": rll,
    }
    return metrics
