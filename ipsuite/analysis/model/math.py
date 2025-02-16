import numpy as np
import uncertainty_toolbox as uct


def compute_trans_forces(mol, key: str = "forces"):
    """Compute translational forces of a molecule."""

    all_forces = np.sum(mol.calc.results[key], axis=0)
    masses = mol.get_masses()
    mol_mas = np.sum(masses)

    if key == "forces":
        mu = (masses / mol_mas)[:, None]
    elif key == "forces_ensemble":
        mu = (masses / mol_mas)[:, None, None]
    else:
        m = "translational forces aceepts keys 'forces' and 'forces_ensemble'"
        raise KeyError(m)

    result = mu * all_forces
    return result


def compute_intertia_tensor(centered_positions, masses):
    r_sq = np.linalg.norm(centered_positions, ord=2, axis=1) ** 2 * masses
    r_sq = np.sum(r_sq)
    A = np.diag(np.full((3,), r_sq))
    mr_k = centered_positions * masses[:, None]
    B = np.einsum("ki, kj -> ij", centered_positions, mr_k)

    I_ab = A - B
    return I_ab


def compute_rot_forces(mol, key: str = "forces"):
    mol_positions = mol.get_positions()
    mol_positions -= mol.get_center_of_mass()
    masses = mol.get_masses()

    if len(mol) <= 2:
        result = np.zeros((len(mol), 3))
        if key == "forces_ensemble":
            result = result[..., None]
        return result

    I_ab = compute_intertia_tensor(mol_positions, masses)
    I_ab_inv = np.linalg.inv(I_ab)

    masses = masses[:, None]
    mi_ri = masses * mol_positions

    contraction_idxs = "ab, b -> a"
    if key == "forces_ensemble":
        mol_positions = mol_positions[..., None]
        contraction_idxs = "ab, nb -> na"

    f_x_r = np.sum(
        np.cross(mol.calc.results[key], mol_positions, axisa=1, axisb=1), axis=0
    )

    # Iinv_fxr = I_ab_inv @ f_x_r but batched for ensembles
    Iinv_fxr = np.einsum(contraction_idxs, I_ab_inv, f_x_r)

    if key == "forces":
        result = np.cross(mi_ri, Iinv_fxr)
    elif key == "forces_ensemble":
        result = np.cross(mi_ri[:, None, :], Iinv_fxr[None, :, :], axisa=2, axisb=2)
        result = np.transpose(result, (0, 2, 1))
    else:
        m = "rotational forces aceepts keys 'forces' and 'forces_ensemble'"
        raise KeyError(m)
    return result


def force_decomposition(
    atom,
    mapping,
    full_forces: np.ndarray | None = None,
    key: str = "forces",
    map: np.ndarray | None = None,
):
    if key not in ["forces", "forces_ensemble"]:
        raise KeyError("Unknown force decomposition key")

    if full_forces is not None:
        if map is None:
            _, molecules, map = mapping.forward_mapping(atom, forces=full_forces)
        else:
            _, molecules, map = mapping.forward_mapping(atom, forces=full_forces, map=map)
        atom_trans_forces = np.zeros_like(full_forces)
        atom_rot_forces = np.zeros_like(full_forces)
        full_forces = np.zeros_like(full_forces)

    elif atom.calc is not None:
        try:
            _, molecules, map = mapping.forward_mapping(atom, map=map)
        except NameError:
            _, molecules, map = mapping.forward_mapping(atom)
        full_forces = np.zeros_like(atom.calc.results[key])
        atom_trans_forces = np.zeros_like(atom.calc.results[key])
        atom_rot_forces = np.zeros_like(atom.calc.results[key])

    total_n_atoms = 0

    for molecule in molecules:
        n_atoms = len(molecule)
        mol_slice = slice(total_n_atoms, total_n_atoms + n_atoms)
        # TODO: What if molecule indices are not ordered?
        full_forces[mol_slice] = molecule.calc.results[key]
        atom_rot_forces[mol_slice] = compute_rot_forces(molecule, key)
        atom_trans_forces[mol_slice] = compute_trans_forces(molecule, key)
        total_n_atoms += n_atoms
    # print(full_forces-test)
    atom_vib_forces = full_forces - atom_trans_forces - atom_rot_forces
    return atom_trans_forces, atom_rot_forces, atom_vib_forces, map


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
    """Compute relative log likelihood
    Adapted from https://github.com/bananenpampe/DPOSE
    """

    mse = np.mean((inputs - target) ** 2)
    uncertainty_estimate = (inputs - target) ** 2

    ll_best = nll(inputs, np.sqrt(uncertainty_estimate), target)

    ll_worst_case_best_RMSE = nll(inputs, np.sqrt(np.ones_like(std) * mse), target)

    ll_actual = nll(inputs, std, target)

    rll = (ll_actual - ll_worst_case_best_RMSE) / (ll_best - ll_worst_case_best_RMSE)

    return rll * 100


def compute_uncertainty_metrics(y_pred, y_std, y_true):
    mask = (y_std > 1e-7) | (y_pred > 1e-7) | (y_true > 1e-7)
    y_pred = y_pred[mask]
    y_std = y_std[mask]
    y_true = y_true[mask]

    mace = uct.mean_absolute_calibration_error(y_pred, y_std, y_true)
    rmsce = uct.root_mean_squared_calibration_error(y_pred, y_std, y_true)
    miscal = uct.miscalibration_area(y_pred, y_std, y_true)
    nll = np.mean(nlls(y_pred, y_std, y_true))
    rll = comptue_rll(y_pred, y_std, y_true)

    metrics = {
        "mace": mace,
        "rmsce": rmsce,
        "miscal": miscal,
        "nll": nll,
        "rll": rll,
    }
    return metrics
