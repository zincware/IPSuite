import logging

import ase
import numpy as np
import zntrack
from ase.cell import Cell
from numpy.random import default_rng

import ipsuite as ips
from ipsuite import base, models

log = logging.getLogger(__name__)


class SurfaceRasterScan(base.ProcessSingleAtom):
    symbol: int = zntrack.zn.params()
    z_dist_list: list[int] = zntrack.zn.params()
    n_conf_per_dist: list[int] = zntrack.zn.params([5, 5])
    cell_fraction: list[float] = zntrack.zn.params([1, 1])
    random: bool = zntrack.zn.params(False)
    rattel: bool = zntrack.zn.params(False)
    max_shift: float = zntrack.zn.params(None)
    seed: bool = zntrack.zn.params(1)

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

            if self.rattel and self.max_shift is None:
                raise ValueError("max_shit must be set.")

            for a in scaled_a_vecs:
                for b in scaled_b_vecs:
                    if self.rattel:
                        new_atoms = atoms.copy()
                        displacement = rng.uniform(
                            -self.max_shift,
                            self.max_shift,
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

        self.atoms = atoms_list

# class SurfaceRasterMetrics(base.ProcessAtoms):
#     # von prediction metricz inheriten and get plot überschreiben?
#     model: models.MLModel = zntrack.deps()
#     seed: int = zntrack.zn.params(0)
    
#     def run(self):
#         atoms_list = self.get_data()

#         prediction = ips.analysis.Prediction(model=self.model, data=atoms_list)
#         pred_energies = prediction.atoms.get_potential_energies()

#         print(pred_energies)