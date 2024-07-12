import zntrack
import subprocess
import pathlib
import shutil
import re

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from ipsuite import base

from pint import UnitRegistry

# Initialize pint unit registry
ureg = UnitRegistry()


def smiles_to_pdb(
    smiles: str,
    file: str,
    identifier: str | None = None,
    cwd: pathlib.Path = pathlib.Path("."),
) -> Chem.Mol:
    """Convert a SMILES string to a PDB file and return the RDKit molecule object."""
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m)
    AllChem.UFFOptimizeMolecule(m)
    pdb_block = Chem.MolToPDBBlock(m)

    if identifier is not None:
        if identifier in pdb_block:
            raise ValueError(
                f"Can not use '{identifier}' because it is present in the file."
            )
        if len(identifier) != 3:
            raise ValueError("Identifier must be 3 characters long")
        pdb_block = pdb_block.replace("UNL", identifier)

    working_directory = pathlib.Path(cwd)

    with open(working_directory / file, "w") as f:
        f.write(pdb_block)
    return m


def get_box(density: float, molecules: list[Chem.Mol], counts: list[int]) -> float:
    """
    Compute the box size for a cubic box with the desired density and given molecules.

    Args:
        density (float): Desired density in g/cm^3.
        molecules (Chem.Mol): Variable number of RDKit molecule objects.

    Returns:
        float: The side length of the cubic box in angstroms.
    """
    total_mass = (
        sum(
            rdMolDescriptors.CalcExactMolWt(mol) * count
            for mol, count in zip(molecules, counts)
        )
        * ureg.dalton
    )  # in atomic mass units (amu)
    total_mass_g = total_mass.to(ureg.gram)  # convert amu to grams
    density_g_per_cm3 = density * (ureg.gram / ureg.centimeter**3)

    volume_cm3 = total_mass_g / density_g_per_cm3  # volume in cm^3
    volume_angstrom3 = volume_cm3.to(ureg.angstrom**3)  # convert cm^3 to angstrom^3
    side_length = volume_angstrom3 ** (
        1 / 3
    )  # side length of the cubic box in angstroms

    return side_length.magnitude


def create_pack_script(
    files: list[str],
    counts: list[int],
    box_size: float,
    cwd: pathlib.Path = pathlib.Path("."),
):
    """
    Create a PACKMOL input script to pack molecules into a box.

    Attributes
    ----------
    files: list[str]
        List of file names for the molecules to pack.
    counts: list[int]
        Number of each molecule to pack.
    box_size: float
        Side length of the cubic box in angstroms.
    """
    if len(files) != len(counts):
        raise ValueError("The number of files must match the number of counts")

    script = """tolerance 2.0
output box.pdb
filetype pdb
"""

    for file, count in zip(files, counts):
        structure_block = f"""structure {file}
  number {count}
  inside box 0. 0. 0. {box_size} {box_size} {box_size}
end structure
"""
        script += structure_block

    with open(cwd / "packmol.inp", "w") as f:
        f.write(script)


def extract_atomtypes(input_file: pathlib.Path, output_file: pathlib.Path):
    content = input_file.read_text()

    # Regular expression to match the [ atomtypes ] section
    atomtypes_regex = re.compile(r"(\[ atomtypes \].*?)(?=\n\[|\Z)", re.DOTALL)

    # Find the atomtypes section
    atomtypes_section = atomtypes_regex.search(content)

    if atomtypes_section:
        atomtypes_section_text = atomtypes_section.group(0)

        # Write the atomtypes section to the output file
        output_file.write_text(atomtypes_section_text)

        # Remove the atomtypes section from the original content
        modified_content = atomtypes_regex.sub("", content)

        # Write the modified content back to the input file
        input_file.write_text(modified_content)


class Smiles2Gromacs(base.IPSNode):
    """Gromacs Node.

    Attributes
    ----------
    smiles: list[str]
        List of SMILES strings for the molecules to be packed.
    count: list[int]
        Number of each molecule to pack.
    labels: list[str]
        List of 3-letter labels for each molecule.
    density: float
        Density of the packed box in g/cm^3.
    mdp_files: list[str | pathlib.Path]
        List of paths to the Gromacs MDP files.

    Installation
    ------------
    To install the required software, run the following commands:
    
    .. code-block:: bash
    
            pip install acpype

    """

    smiles: list[str] = zntrack.params()
    count: list[int] = zntrack.params()
    labels: list[str] = zntrack.params()
    density: float = zntrack.params()

    mdp_files: list[str | pathlib.Path] = zntrack.deps_path()

    output_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "gromacs")

    def _post_init_(self):
        if len(self.smiles) != len(self.count):
            raise ValueError("The number of smiles must match the number of counts")
        if len(self.smiles) != len(self.labels):
            raise ValueError("The number of smiles must match the number of labels")
        

        if isinstance(self.output_dir, str):
            self.output_dir = pathlib.Path(self.output_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.mdp_files = [pathlib.Path(mdp_file) for mdp_file in self.mdp_files]

    def _run_acpype(self):
        for label, charge in zip(self.labels, self.charges):
            cmd = ["acpype", "-i", f"{label}.pdb", "-n", str(charge), "-b", label]
            subprocess.run(cmd, check=True, cwd=self.output_dir)

    def _create_box_gro(self):
        cmd = [
            "echo",
            "0",
            "|",
            "gmx",
            "editconf",
            "-f",
            self.box,
            "-bt",
            "cubic",
            "-d",
            "1",
            "-c",
            "-princ",
            "-o",
            "box.gro",
        ]
        subprocess.run(" ".join(cmd), shell=True, check=True, cwd=self.output_dir)

    def _create_species_top_atomtypes(self):
        for label in self.labels:
            file = self.output_dir / f"{label}.acpype/{label}_GMX.itp"
            shutil.copy(file, self.output_dir / f"{label}.itp")
            shutil.copy(file.with_suffix(".top"), self.output_dir / f"{label}.top")
            extract_atomtypes(
                self.output_dir / f"{label}.itp",
                self.output_dir / f"{label}_atomtypes.itp",
            )

    def _create_box_top(self):
        with open(self.output_dir / "box.top", "w") as f:
            f.write("[ defaults ]")
            f.write("\n")
            f.write("; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ\n")
            f.write(
                "1               2               yes             0.5     0.8333333333\n"
            )
            f.write("\n")
            f.write("; Include atomtypes\n")
            for label in self.labels:
                f.write(f'#include "{label}_atomtypes.itp"\n')
            f.write("\n")
            f.write("; Include topology\n")
            for label in self.labels:
                f.write(f'#include "{label}.itp"\n')

            f.write("\n")
            f.write("[ system ]\n")
            f.write(" GMX\n")

            f.write("\n")
            f.write("[ molecules ]\n")
            for label, count in zip(self.labels, self.count):
                f.write(f"{label} {count}\n")

    def _run_gmx(self):
        for mdp_file in self.mdp_files:
            cmd = [
                "gmx",
                "grompp",
                "-f",
                mdp_file.resolve().as_posix(),
                "-c",
                "box.gro",
                "-p",
                "box.top",
                "-o",
                "box.tpr",
                "-v",
                "-maxwarn",
                "10",
            ]
            print(f"Running {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=self.output_dir)

            cmd = ["gmx", "mdrun", "-v", "-deffnm", "box"]
            subprocess.run(cmd, check=True, cwd=self.output_dir)

    def _pack_box(self):
        mols = []
        charges = []
        for smiles, label in zip(self.smiles, self.labels):
            mols.append(
                smiles_to_pdb(smiles, f"{label}.pdb", label, cwd=self.output_dir)
            )
            # get the charge of the molecule
            charges.append(Chem.GetFormalCharge(mols[-1]))
        self.charges = charges
        box_size = get_box(self.density, mols, self.count)
        create_pack_script(
            [f"{label}.pdb" for label in self.labels],
            self.count,
            box_size,
            cwd=self.output_dir,
        )
        cmd = ["packmol < packmol.inp"]
        subprocess.run(cmd, check=True, shell=True, cwd=self.output_dir)
        self.box = "box.pdb"

    def run(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._pack_box()
        self._run_acpype()
        self._create_box_gro()
        self._create_species_top_atomtypes()
        self._create_box_top()
        self._run_gmx()
