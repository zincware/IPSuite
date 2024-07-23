import pathlib
import re
import shutil
import subprocess

import MDAnalysis as mda
import numpy as np
import pandas as pd
import zntrack
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from pint import UnitRegistry
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from ipsuite import base, fields

# Initialize pint unit registry
ureg = UnitRegistry()


def smiles_to_pdb(
    smiles: str,
    file: str,
    identifier: str | None = None,
    cwd: pathlib.Path = pathlib.Path(),
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
    side_length = volume_angstrom3 ** (1 / 3)  # side length of the cubic box in angstroms

    return side_length.magnitude


def create_pack_script(
    files: list[str],
    counts: list[int],
    box_size: float,
    tolerance: float,
    cwd: pathlib.Path = pathlib.Path(),
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

    box_size -= tolerance

    script = f"""tolerance {tolerance}
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


def combine_atomtype_files(files: list[pathlib.Path], output_file: pathlib.Path):
    """Read all the files and write a single output file. Removes duplicates."""
    header = []
    atomtypes = []
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx in [0, 1]:
                    if len(header) < 2:
                        header.append(line)
                else:
                    atomtypes.append(line)

    atomtypes = list(set(atomtypes))
    with open(output_file, "w") as f:
        f.writelines(header)
        f.writelines(atomtypes)


def validate_mdp(path):
    necessary_keys = ["nstxout", "nstfout"]
    path = pathlib.Path(path)
    with path.open("r") as f:
        content = f.read()
        for key in necessary_keys:
            if key not in content:
                raise ValueError(
                    f"Key '{key}' is required in {path.name} for writing a trajectory"
                )


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
    itp_files: list[str | None]|None
        if given, for each label either the path to the
        ITP file or None.  The order must match the order
        of the labels.
    pdb_files: list[str | pathlib.Path]|None
        if given, for each label either the path to the
        PDB file or None.  The order must match the order
        of the labels.

    Installation
    ------------
    To install the required software, run the following commands:

    .. code-block:: bash

            conda install conda-forge::gromacs
            conda install conda-forge::acpype
            pip install MDAnalysis

    """

    smiles: list[str] = zntrack.params()
    count: list[int] = zntrack.params()
    labels: list[str] = zntrack.params()
    density: float = zntrack.params()
    fudgeLJ: float = zntrack.params(1.0)
    fudgeQQ: float = zntrack.params(1.0)
    tolerance: float = zntrack.params(1.0)

    mdp_files: list[str | pathlib.Path] = zntrack.deps_path()
    itp_files: list[str | None] = zntrack.deps_path(None)
    pdb_files: list[str | pathlib.Path] = zntrack.deps_path(None)

    atoms = fields.Atoms()

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
        for idx, (label, charge) in enumerate(zip(self.labels, self.charges)):
            if self.itp_files is not None and self.itp_files[idx] is not None:
                path = self.output_dir / f"{label}.acpype"
                path.mkdir(exist_ok=True)
                shutil.copy(self.itp_files[idx], path / f"{label}_GMX.itp")
            else:
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
            "-o",
            "box.gro",
            "-box",
            str((self.box_size * ureg.angstrom).to(ureg.nanometer).magnitude),
        ]
        subprocess.run(" ".join(cmd), shell=True, check=True, cwd=self.output_dir)

    def _create_species_top_atomtypes(self):
        for idx, label in enumerate(self.labels):
            if self.itp_files is not None and self.itp_files[idx] is not None:
                file = self.itp_files[idx]
            else:
                file = self.output_dir / f"{label}.acpype/{label}_GMX.itp"
            shutil.copy(file, self.output_dir / f"{label}.itp")
            # shutil.copy(file.with_suffix(".top"), self.output_dir / f"{label}.top")
            extract_atomtypes(
                self.output_dir / f"{label}.itp",
                self.output_dir / f"{label}_atomtypes.itp",
            )

        combine_atomtype_files(
            [self.output_dir / f"{label}_atomtypes.itp" for label in self.labels],
            self.output_dir / "atomtypes.itp",
        )

    def _create_box_top(self):
        with open(self.output_dir / "box.top", "w") as f:
            f.write("[ defaults ]")
            f.write("\n")
            f.write("; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ\n")
            f.write(
                f"1               2               yes             {self.fudgeLJ}    "
                f" {self.fudgeQQ}\n"
            )
            f.write("\n")
            f.write("; Include atomtypes\n")
            f.write('#include "atomtypes.itp"\n')
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
            ]
            print(f"Running {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=self.output_dir)
            cmd = ["gmx", "mdrun", "-ntmpi", "1", "-v", "-deffnm", "box"]
            subprocess.run(cmd, check=True, cwd=self.output_dir)

    def _pack_box(self):
        mols = []
        charges = []
        for idx, (smiles, label) in enumerate(zip(self.smiles, self.labels)):
            if self.pdb_files is not None and self.pdb_files[idx] is not None:
                shutil.copy(self.pdb_files[idx], self.output_dir / f"{label}.pdb")
                m = Chem.MolFromSmiles(smiles)
                m = Chem.AddHs(m)
                AllChem.EmbedMolecule(m)
                AllChem.UFFOptimizeMolecule(m)
                mols.append(m)
            else:
                mols.append(
                    smiles_to_pdb(smiles, f"{label}.pdb", label, cwd=self.output_dir)
                )
            # get the charge of the molecule
            charges.append(Chem.GetFormalCharge(mols[-1]))
        self.charges = charges
        self.box_size = get_box(self.density, mols, self.count)
        create_pack_script(
            [f"{label}.pdb" for label in self.labels],
            self.count,
            self.box_size,
            self.tolerance,
            cwd=self.output_dir,
        )
        cmd = ["packmol < packmol.inp"]
        subprocess.run(cmd, check=True, shell=True, cwd=self.output_dir)
        self.box = "box.pdb"

    def _extract_energies(self):
        cmd = ["echo", "8", "|", "gmx", "energy", "-f", "box.edr"]
        subprocess.run(cmd, check=True, cwd=self.output_dir)

        lineNumber = 1
        with (self.output_dir / "energy.xvg").open("r") as in_file:
            for i, line in enumerate(in_file, 1):
                if line.startswith(" "):
                    lineNumber = i
                    break
        df = pd.read_csv(
            "energy.xvg",
            skiprows=lineNumber,
            header=None,
            names=["time", "energy"],
            sep="\s+",
        )
        energies = df["energy"].iloc[:] * units.kcal / units.mol
        return energies

    def _convert_trajectory(self):
        atoms_template = read((self.output_dir / "box.gro").as_posix())
        trr = mda.coordinates.TRR.TRRReader((self.output_dir / "box.trr").as_posix())

        # energies = self._extract_energies()
        energies = np.zeros(len(trr))

        traj = []
        Z = atoms_template.numbers
        for frame, energy in zip(trr, energies):
            pos = frame.positions
            forces = frame.forces * units.kcal / units.mol / 10
            cell = frame.dimensions

            new_atoms = Atoms(numbers=Z, positions=pos, cell=cell)
            calc = SinglePointCalculator(new_atoms, energy=energy, forces=forces)
            new_atoms.calc = calc
            traj.append(new_atoms)

        self.atoms = traj

    def run(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        validate_mdp(self.mdp_files[-1])

        self._pack_box()
        self._create_box_gro()

        self._run_acpype()

        self._create_species_top_atomtypes()
        self._create_box_top()
        self._run_gmx()
        self._convert_trajectory()
