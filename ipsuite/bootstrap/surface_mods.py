import logging

import ase
import matplotlib.pyplot as plt
import numpy as np
import zntrack
from ase.cell import Cell
from numpy.random import default_rng
import znh5md
import typing
import h5py

from ipsuite import analysis, base

log = logging.getLogger(__name__)


class SurfaceRasterScan(base.ProcessSingleAtom):
    """This class generates periodic structures by creating a vacuum slab in the
    z-direction and adding additives at various positions. It is useful for generating
    input structures for surface training simulations or in combination with the
    SurfaceRasterMetrics class to analyze how well surface interactions are captured
    in the training.

    Attributes
    ----------
    symbol: str
        ASE symbol representing the additives.
    z_dist_list: typing.List[float]
         A list of z-distances at which additives will be added.
    n_conf_per_dist: typing.List[int]
        The number of configurations to generate per z-distance.
    cell_fraction: typing.List[float]
        Fractional scaling of the unit cell in x and y directions.
    random: bool
       If True, additives are placed randomly within the specified cell_fraction.
    max_rattel_shift: float
        Maximum random displacement for each atom.
    seed: int
        Seed for randomly distributing the additive.
    """

    symbol: str = zntrack.params()
    z_dist_list: typing.List[float] = zntrack.params()
    n_conf_per_dist: typing.List[int] = zntrack.params((5, 5))
    cell_fraction: typing.List[float] = zntrack.params((1, 1))
    random: bool = zntrack.params(False)
    max_rattel_shift: float = zntrack.params(None)
    seed: int = zntrack.params(1)

    def run(self) -> None:
        rng = default_rng(self.seed)

        atoms = self.get_data()

        cell = atoms.cell
        cellpar = cell.cellpar()
        cell = np.array(cell)

        z_max = max(atoms.get_positions()[:, 2])

        if not isinstance(self.n_conf_per_dist, list):
            self.n_conf_per_dist = [self.n_conf_per_dist, self.n_conf_per_dist]
        if not isinstance(self.cell_fraction, list):
            self.cell_fraction = [self.cell_fraction, self.cell_fraction]
        atoms_list = []
        for z_dist in self.z_dist_list:
            if cellpar[2] < z_max + z_dist + 10:
                cellpar[2] = z_max + z_dist + 10
                new_cell = Cell.fromcellpar(cellpar)
                atoms.set_cell(new_cell)
                log.warning("vacuum was extended")

            if not self.random:
                a_scaling = np.linspace(0, 1, self.n_conf_per_dist[0])
                b_scaling = np.linspace(0, 1, self.n_conf_per_dist[1])
            else:
                a_scaling = np.random.rand(self.n_conf_per_dist[0])
                a_scaling = np.sort(a_scaling)
                b_scaling = np.random.rand(self.n_conf_per_dist[1])
                b_scaling = np.sort(b_scaling)

            a_vec = cell[0, :2] * self.cell_fraction[0]
            scaled_a_vecs = a_scaling[:, np.newaxis] * a_vec
            b_vec = cell[1, :2] * self.cell_fraction[1]
            scaled_b_vecs = b_scaling[:, np.newaxis] * b_vec

            for a in scaled_a_vecs:
                for b in scaled_b_vecs:
                    if self.max_rattel_shift is not None:
                        new_atoms = atoms.copy()
                        displacement = rng.uniform(
                            -self.max_rattel_shift,
                            self.max_rattel_shift,
                            size=new_atoms.positions.shape,
                        )
                        new_atoms.positions += displacement
                        atoms_list.append(new_atoms)
                    else:
                        atoms_list.append(atoms.copy())

                    cart_pos = a + b
                    extension = ase.Atoms(
                        self.symbol, [[cart_pos[0], cart_pos[1], z_max + z_dist]]
                    )
                    atoms_list[-1].extend(extension)

        self.frames = atoms_list


class SurfaceRasterMetrics(analysis.PredictionMetrics):
    """This class analyzes the surface interaction of an additive with a surface.
    It is used to evaluate how well the surface structure is learned during training.
    Note that the bulk atoms should not be rattled in the SurfaceRasterScan node.

    Attributes
    ----------
    scan_node: SurfaceRasterScan()
       The node used for generating the structures

    """

    scan_node: SurfaceRasterScan = zntrack.deps()

    def get_plots(self, save=False):
        super().get_plots(save=True)
        self.plots_dir.mkdir(exist_ok=True)

        # get positions
        pos = []
        for atoms in self.data.atoms:
            pos.append(atoms.positions[-1])
        pos = np.array(pos)

        shape = [len(self.scan_node.z_dist_list)]
        shape.append(self.scan_node.n_conf_per_dist[0])
        shape.append(self.scan_node.n_conf_per_dist[1])

        x_pos = np.reshape(pos[:, 0], shape)
        x_pos = x_pos[0]
        for j in range(x_pos.shape[1]):
            x_pos[j, :] = x_pos[j, 0]

        y_pos = np.reshape(pos[:, 1], shape)
        y_pos = y_pos[0]

        # get energy
        true_energies = np.reshape(self.energy_df["true"], shape)
        pred_energies = np.reshape(self.energy_df["prediction"], shape)

        # get forces
        shape.append(3)

        forces = []
        for true_data, pred_data in zip(self.x, self.y):
            forces.append([true_data.get_forces(), pred_data.get_forces()])

        forces = np.array(forces)[:, :, -1, :] * 1000
        true_forces = np.reshape(forces[:, 0, :], shape)
        pred_forces = np.reshape(forces[:, 1, :], shape)

        for i, distance in enumerate(self.scan_node.z_dist_list):
            plot_ture_vs_pred(
                x_pos,
                y_pos,
                [true_energies[i, :], pred_energies[i, :]],
                "energies",
                distance,
                plots_dir=self.plots_dir,
            )
            plot_ture_vs_pred(
                x_pos,
                y_pos,
                [true_forces[i, :, :, 2], pred_forces[i, :, :, 2]],
                "forces",
                distance,
                plots_dir=self.plots_dir,
            )


def plot_ture_vs_pred(x, y, z, name, height, plots_dir):
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        cm = ax.pcolormesh(x, y, z[i])
        ax.axis("scaled")
        ax.set_xlabel(r"x position additiv $\AA$")
        ax.set_ylabel(r"y-position additiv $\AA$")
    axes[0].set_title(f"true-{name}")
    axes[1].set_title(f"predicted-{name}")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.015, 0.03, 0.87])
    fig.colorbar(cm, cax=cbar_ax)

    if name == "energies":
        cbar_ax.set_ylabel(r"Energy $E$ / meV/atom")
    if name == "forces":
        cbar_ax.set_ylabel(r"Magnetude of force per atom $|F|$ meV$ \cdot \AA^{-1}$")

    fig.suptitle(rf"Additive {height} $\AA$ over the surface")
    fig.savefig(plots_dir / f"{name}-{height}-heat.png")



def y_rot_mat(angle_radians):
    rotation_matrix = np.array([[np.cos(angle_radians), 0, -np.sin(angle_radians)],
                                [0, 1, 0],
                                [np.sin(angle_radians), 0, np.cos(angle_radians)],
                                ])
    return rotation_matrix

def z_rot_mat(angle_radians):
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                [np.sin(angle_radians), np.cos(angle_radians), 0],
                                [0, 0, 1],
                                ])
    return rotation_matrix

def position_velocitie_rotation(pos, velo, angle_degrees, rot_axis):
    rot_matrix = {"y": y_rot_mat,
                     "z": z_rot_mat}
    
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    
    rotation_matrix = rot_matrix[rot_axis](angle_radians)
    
    # Apply the rotation matrix to the vector
    rotated_pos = np.dot(rotation_matrix, pos)
    rotate_velo = np.dot(rotation_matrix, velo)
    return rotated_pos, rotate_velo

class PosVeloRotation(base.ProcessSingleAtom):
    """This class generates 
    """

    symbol: str = zntrack.params()
    y_rotation_angles: typing.List[float] = zntrack.params()
    z_rotation_angles: typing.List[float] = zntrack.params()
    additive_hight: float = zntrack.params()           #np.array([0., 0., 8.0*Ang,])
    velocitie: typing.List[float] = zntrack.params()          #np.array([0., 0., -8000.0*m/s,])
    n_conf_per_dist: typing.List[int] = zntrack.params() #(5, 5)
    cell_fraction: typing.List[float] = zntrack.params() #[1, 1]
    impact_position: typing.List[float] = zntrack.params(None)           #np.array([0., 0.])
    seed: int = zntrack.params(42)
    # output_file: str = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def run(self) -> None:
        np.random.seed(self.seed)
        self.y_rotation_angles = np.array(self.y_rotation_angles)
        self.z_rotation_angles = np.array(self.z_rotation_angles)
        self.velocitie = np.array(self.velocitie)
        
        # db = znh5md.IO(self.output_file)
        
        atoms = self.get_data()
        cell = atoms.cell
        cellpar = cell.cellpar()

        z_max = max(atoms.get_positions()[:, 2])
        if cellpar[2] < self.additive_hight + 10:
            cellpar[2] = self.additive_hight + 10
            new_cell = Cell.fromcellpar(cellpar)
            atoms.set_cell(new_cell)
            log.warning("vacuum was extended")

        cell = np.array(atoms.cell)  
                
        if self.impact_position is None:
            position = np.array([0, 0, self.additive_hight])
            a_scaling = np.random.uniform(0, 1, self.n_conf_per_dist[0])
            b_scaling = np.random.uniform(0, 1, self.n_conf_per_dist[1])
        else:
            fraction = [(2*(self.n_conf_per_dist[0]))**-1, (2*(self.n_conf_per_dist[1]))**-1]
            position = np.array([self.impact_position[0], self.impact_position[1], self.additive_hight])
            a_scaling = np.linspace(fraction[0], 1-fraction[0], self.n_conf_per_dist[0])
            b_scaling = np.linspace(fraction[1], 1-fraction[1], self.n_conf_per_dist[1])

        a_vec = cell[0, :2] * self.cell_fraction[0]
        scaled_a_vecs = a_scaling[:, np.newaxis] * a_vec
        b_vec = cell[1, :2] * self.cell_fraction[1]
        scaled_b_vecs = b_scaling[:, np.newaxis] * b_vec

        structures = []
        
        for z_angle in self.z_rotation_angles:
            for y_angle in self.y_rotation_angles:
                if self.impact_position is None:
                    position = np.array([0, 0, self.additive_hight])
                    a_scaling = np.random.uniform(0, 1, self.n_conf_per_dist[0])
                    b_scaling = np.random.uniform(0, 1, self.n_conf_per_dist[1])
                    
                    a_vec = cell[0, :2] * self.cell_fraction[0]
                    scaled_a_vecs = a_scaling[:, np.newaxis] * a_vec
                    b_vec = cell[1, :2] * self.cell_fraction[1]
                    scaled_b_vecs = b_scaling[:, np.newaxis] * b_vec
                    
                for a in scaled_a_vecs:
                    for b in scaled_b_vecs:
                        xy_impact_pos = np.array(a + b)


                        structures.append(atoms.copy())
                        
                        rot_pos, rot_velo = position_velocitie_rotation(position, self.velocitie, y_angle, "y")
                        rot_pos_z, rot_velo_z = position_velocitie_rotation(rot_pos, rot_velo, z_angle, "z")

                        final_pos = rot_pos_z + np.array([xy_impact_pos[0], xy_impact_pos[1], z_max])
                        additive = ase.Atoms(
                                self.symbol,
                                [final_pos],
                                velocities=rot_velo_z,
                        )
                        structures[-1].extend(additive)

        # db.extend(structures)
        self.frames = structures
        
    # @property         
    # def frames(self) -> typing.List[ase.Atoms]:
    #     with self.state.fs.open(self.output_file, "rb") as f:
    #         with h5py.File(f) as file:
    #             return znh5md.IO(file_handle=file)[:]