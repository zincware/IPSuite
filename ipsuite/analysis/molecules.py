import ase
import rdkit2ase
import tqdm
import zntrack

from ipsuite import base
from ipsuite.geometry import BarycenterMapping


class FindAllowedMolecules(base.IPSNode):
    """Search a given dataset for outliers.

    Iterates all structures in the dataset, uses covalent radii to determine
    the atoms in each molecule, and checks if the molecule is allowed.

    Attributes
    ----------
    data : list[ase.Atoms]
        The dataset to search.
    molecules : list[ase.Atoms], optional
        The molecules that are allowed.
    smiles : list[str], optional
        The SMILES strings of the allowed molecules.
    """

    data: list[ase.Atoms] = zntrack.deps()
    molecules: list[ase.Atoms] = zntrack.deps(default_factory=list)
    smiles: list[str] = zntrack.params(default_factory=list)

    outliers: list[int] = zntrack.outs()

    def run(self):
        molecules = self.molecules + [rdkit2ase.smiles2atoms(s) for s in self.smiles]
        mapping = BarycenterMapping()
        for idx, atoms in enumerate(tqdm.tqdm(self.data)):
            _, mols = mapping.forward_mapping(atoms)
            for mol in mols:
                # check if the atomic numbers are the same
                if sorted(mol.get_atomic_numbers()) in [
                    sorted(m.get_atomic_numbers()) for m in molecules
                ]:
                    continue
                self.outliers.append(idx)
