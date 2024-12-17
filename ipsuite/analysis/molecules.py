import ase
import rdkit2ase
import tqdm
import zntrack

from ipsuite import base
from ipsuite.geometry import BarycenterMapping


class AllowedStructuresFilter(base.IPSNode):
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
    fail: bool = zntrack.params(False)

    outliers: list[int] = zntrack.outs()

    def run(self):
        molecules = self.molecules + [rdkit2ase.smiles2atoms(s) for s in self.smiles]
        mapping = BarycenterMapping()
        self.outliers = []
        for idx, atoms in enumerate(tqdm.tqdm(self.data)):
            _, mols = mapping.forward_mapping(atoms)
            for mol in mols:
                # check if the atomic numbers are the same
                if sorted(mol.get_atomic_numbers()) in [
                    sorted(m.get_atomic_numbers()) for m in molecules
                ]:
                    continue
                if self.fail:
                    raise ValueError(f"Outlier found at index {idx} for molecule {mol}")
                else:
                    print(f"Outlier found at index {idx} for molecule {mol}")
                self.outliers.append(idx)

    
    @property
    def excluded_frames(self) -> list[ase.Atoms]:
        return [self.data[idx] for idx in self.outliers]
    
    @property
    def included_frames(self) -> list[ase.Atoms]:
        return [self.data[idx] for idx in range(len(self.data)) if idx not in self.outliers]
