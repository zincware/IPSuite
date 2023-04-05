import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interpn


def density_scatter(ax, x, y, bins, **kwargs) -> None:
    """Create a scatter plot colored by 2d histogram density.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x: np.ndarray
    y: np.ndarray
    bins: int
    kwargs
        any kwargs passed to 'ax.scatter'

    Returns
    -------

    References
    ----------
    Adapted from https://stackoverflow.com/a/53865762/10504481

    """
    # convert e.g. DataFrame to numpy array values
    x = np.array(x)
    y = np.array(y)

    if "cmap" not in kwargs:
        kwargs["cmap"] = "viridis"

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    points = (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1]))
    xi = np.vstack([x, y]).T
    z = interpn(points, data, xi, method="splinef2d", bounds_error=False)
    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, **kwargs)


def get_figure(
    true, prediction, datalabel: str, xlabel: str, ylabel: str, figsize: tuple = (10, 7)
) -> plt.Figure:
    """Create a correlation plot for true, prediction values.

    Parameters
    ----------
    true: the true values
    prediction: the predicted values
    datalabel: str, the label for the prediction, e.g. 'MAE: 0.123 meV'
    xlabel: str, the xlabel
    ylabel: str, the xlabel
    figsize: tuple, size of the figure

    Returns
    -------
    plt.Figure

    """
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(true, true, color="grey", zorder=0)  # plot the diagonal in the background
    bins = 500 if (len(true) / 10) > 500 else int(len(true) * 0.1)
    if bins < 20:
        # don't use density for very small datasets
        ax.scatter(true, prediction, marker="x", s=20.0, label=datalabel)
    else:
        density_scatter(
            ax, true, prediction, bins=bins, marker="x", s=20.0, label=datalabel
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig


def get_hist(data, label, xlabel, ylabel) -> typing.Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()

    sns.histplot(
        data,
        ax=ax,
        stat="percent",
        label=label,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return fig, ax


def compute_trans_forces(mol):
    """Compute translational forces of a molecule."""

    all_forces = np.sum(mol.get_forces(), axis=0)
    masses = mol.get_masses()
    mol_mas = np.sum(masses)
    res = (masses / mol_mas)[:, None] * all_forces
    return res


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
    res = np.cross(mi_ri, (I_ab_inv @ f_x_r))

    return res


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
