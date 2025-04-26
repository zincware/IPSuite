import zntrack
from ipsuite import base
from ase.neighborlist import build_neighbor_list
from scipy import sparse
from ipsuite.utils.ase_sim import freeze_copy_atoms
import numpy as np


def delete_reflected_atoms(atoms, cutoff_plane):
    frames = [freeze_copy_atoms(atoms)]
    z_pos = atoms.positions[:, 2]
    idxs = np.where(z_pos > cutoff_plane)[0]

    nl = build_neighbor_list(atoms, self_interaction=False)
    matrix = nl.get_connectivity_matrix()
    _, component_list = sparse.csgraph.connected_components(matrix)

    del_atom_idxs = []
    del_mol_idxs = []
    for atom_idx in idxs:
        mol_idx = component_list[atom_idx]
        if mol_idx not in del_mol_idxs:
            del_mol_idxs.append(mol_idx)
            del_atom_idxs.extend(
                [i for i in range(len(component_list)) if component_list[i] == mol_idx]
            )

    new_frame = freeze_copy_atoms(atoms)

    if len(del_atom_idxs) == len(atoms):
        raise ValueError("Can not delete all Atoms of the System")
    elif del_atom_idxs:
        del new_frame[del_atom_idxs]
        frames = [new_frame]
        print(
            f"Molecule/s {del_mol_idxs} with Atom(s) {del_atom_idxs} was/were reflected and deleted.\n"
        )
    return frames


MODS = {
    "del-reflected-atoms": delete_reflected_atoms,
}


class ModFrames(base.ProcessSingleAtom):
    moddification: str = zntrack.params()
    run_kwargs: dict = zntrack.params()

    def run(self) -> None:
        atoms = self.get_data()
        self.frames = MODS[self.moddification](atoms, **self.run_kwargs)
