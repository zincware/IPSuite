import contextlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import typing as t

import h5py
import MDAnalysis as mda
import yaml
import znh5md
import zntrack
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers
from MDAnalysis.coordinates.timestep import Timestep
from pint import UnitRegistry
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from ipsuite import base

# Initialize pint unit registry
ureg = UnitRegistry()


def dict_to_mdp(data: dict) -> str:
    """Convert a dictionary to a Gromacs .mdp file."""
    # convert all values that are lists to strings
    # e.g. for `annealing-time`
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = " ".join([str(v) for v in value])

    return "\n".join([f"{key} = {value}" for key, value in data.items()])


def params_to_mdp(file: pathlib.Path, target: pathlib.Path):
    """Convert yaml/json files to mdp files."""
    if file.suffix in [".yaml", ".yml"]:
        with file.open("r") as f:
            data = yaml.safe_load(f)
            data = dict_to_mdp(data)
    elif file.suffix == ".json":
        with file.open("r") as f:
            data = json.load(f)
            data = dict_to_mdp(data)
    else:
        data = file.read_text()

    target.write_text(data)


def timestep_to_atoms(u: mda.Universe, ts: Timestep) -> t.Tuple[Atoms, float]:
    """Convert an MDAnalysis timestep to an ASE atoms object.

    Parameters
    ----------
    u : MDAnalysis.Universe
        The MDAnalysis Universe object containing topology information.
    ts : MDAnalysis.coordinates.Timestep
        The MDAnalysis Timestep object to convert.

    Returns
    -------
    ase.Atoms
        The converted ASE Atoms object.
    float
        The time of the timestep in femtoseconds.
    """
    # Set the universe's trajectory to the provided timestep
    u.trajectory.ts = ts

    # Extract positions from timestep
    positions = ts.positions

    # Extract masses and elements from the universe
    names = u.atoms.names
    cell = u.dimensions

    # Adapted from ASE gro reader
    symbols = []
    for name in names:
        if name in atomic_numbers:
            symbols.append(name)
        elif name[0] in atomic_numbers:
            symbols.append(name[0])
        elif name[-1] in atomic_numbers:
            symbols.append(name[-1])
        else:
            # not an atomic symbol
            # if we can not determine the symbol, we use
            # the dummy symbol X
            symbols.append("X")

    forces = ts.forces * ureg.kilocalories / ureg.mol / ureg.angstrom
    # convert to eV/Å
    forces.ito(ureg.eV / ureg.angstrom / ureg.particle)
    forces = forces.magnitude
    energy = 0
    with contextlib.suppress(KeyError):
        energy = ts.aux["Potential"]
        energy = energy * ureg.kilocalories / ureg.mol
        energy.ito(ureg.eV / ureg.particle)
        energy = energy.magnitude

    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

    with contextlib.suppress(KeyError):
        atoms.info["temperature"] = ts.aux["Temperature"].item()
    with contextlib.suppress(KeyError):
        atoms.info["pressure"] = ts.aux["Pressure"].item()
    with contextlib.suppress(KeyError):
        atoms.info["density"] = ts.aux["Density"].item()

    return atoms, (ts.time * ureg.picosecond).to(ureg.femtosecond).magnitude


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

    with (working_directory / file).open("w") as f:
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

    with (cwd / "packmol.inp").open("w") as f:
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
        with file.open("r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if idx in [0, 1]:
                    if len(header) < 2:
                        header.append(line)
                else:
                    atomtypes.append(line)

    atomtypes = list(set(atomtypes))
    with output_file.open("w") as f:
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
    config_files: list[str | pathlib.Path]
        Same like mdp_files but in the json or yaml format.
        These will run BEFORE the MDP files.
    itp_files: list[str | None]|None
        if given, for each label either the path to the
        ITP file or None.  The order must match the order
        of the labels.
    pdb_files: list[str | pathlib.Path]|None
        if given, for each label either the path to the
        PDB file or None.  The order must match the order
        of the labels.
    production_indices: list[int]|None
        The gromacs runs that should be stored in the
        trajectory file.  If None, the last run is stored.
        The order is always the same as the order of the
        MDP files.

    Installation
    ------------
    To install the required software, run the following commands:

    .. code-block:: bash

            conda install conda-forge::gromacs
            conda install conda-forge::acpype
            pip install MDAnalysis pyedr

    """

    smiles: list[str] = zntrack.params()
    count: list[int] = zntrack.params()
    labels: list[str] = zntrack.params()
    density: float = zntrack.params()
    fudgeLJ: float = zntrack.params(1.0)
    fudgeQQ: float = zntrack.params(1.0)
    tolerance: float = zntrack.params(2.0)
    production_indices: list[int] = zntrack.params(None)
    cleanup: bool = zntrack.params(True)
    maxwarn: int = zntrack.params(0)

    mdp_files: t.Sequence[str | pathlib.Path] = zntrack.deps_path(default_factory=list)
    config_files: t.Sequence[str | pathlib.Path] = zntrack.params_path(
        default_factory=list
    )
    itp_files: t.Sequence[str | pathlib.Path | None] = zntrack.deps_path(
        default_factory=list
    )
    pdb_files: t.Sequence[str | pathlib.Path | None] = zntrack.deps_path(
        default_factory=list
    )

    traj_file: list[Atoms] = zntrack.outs_path(zntrack.nwd / "structures.h5")

    output_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "gromacs")

    def __post_init__(self):
        if len(self.smiles) != len(self.count):
            raise ValueError("The number of smiles must match the number of counts")
        if len(self.smiles) != len(self.labels):
            raise ValueError("The number of smiles must match the number of labels")
        if self.production_indices is None:
            self.production_indices = [len(self.mdp_files) + len(self.config_files) - 1]

        if isinstance(self.output_dir, str):
            self.output_dir = pathlib.Path(self.output_dir)
        self.mdp_files = [pathlib.Path(mdp_file) for mdp_file in self.mdp_files]
        self.config_files = [
            pathlib.Path(config_file) for config_file in self.config_files
        ]
        # check that the file name without suffix is unique between all files
        names = [file.stem for file in self.mdp_files + self.config_files]
        if len(names) != len(set(names)):
            raise ValueError("The file names must be unique")

    @property
    def frames(self) -> list[Atoms]:
        with self.state.fs.open(self.traj_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]

    def _run_acpype(self):
        for idx, (label, charge) in enumerate(zip(self.labels, self.charges)):
            if len(self.itp_files) and self.itp_files[idx] is not None:
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
            if len(self.itp_files) and self.itp_files[idx] is not None:
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
        with (self.output_dir / "box.top").open("w") as f:
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
        for file in self.config_files + self.mdp_files:
            params_to_mdp(file, self.output_dir / file.with_suffix(".mdp").name)

        for mdp_file in self.config_files + self.mdp_files:
            cmd = [
                "gmx",
                "grompp",
                "-f",
                mdp_file.with_suffix(".mdp").name,
                "-c",
                "box.gro",
                "-p",
                "box.top",
                "-o",
                "box.tpr",
                "-v",
                "-maxwarn",
                str(self.maxwarn),
            ]
            print(f"Running {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=self.output_dir)
            NTMPI = os.environ.get("IPSUITE_GMX_NTMPI", "1")
            cmd = ["gmx", "mdrun", "-ntmpi", NTMPI, "-v", "-deffnm", "box"]
            subprocess.run(cmd, check=True, cwd=self.output_dir)

    def _pack_box(self):
        mols = []
        charges = []
        for idx, (smiles, label) in enumerate(zip(self.smiles, self.labels)):
            if len(self.pdb_files) and self.pdb_files[idx] is not None:
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

    def _convert_trajectory(self):
        io = znh5md.IO(self.traj_file, store="time")
        for idx in sorted(self.production_indices):
            if idx == len(self.mdp_files) + len(self.config_files) - 1:
                gro = self.output_dir / "box.gro"
                trr = self.output_dir / "box.trr"
                edr = self.output_dir / "box.edr"
            else:
                gro = self.output_dir / f"#box.gro.{idx + 1}#"
                trr = self.output_dir / f"#box.trr.{idx + 1}#"
                edr = self.output_dir / f"#box.edr.{idx + 1}#"
            u = mda.Universe(gro, trr, topology_format="GRO", format="TRR")
            aux = mda.auxiliary.EDR.EDRReader(edr)
            u.trajectory.add_auxiliary(auxdata=aux)

            atoms_list = []
            timesteps = []

            for ts in u.trajectory:
                atoms, timestep = timestep_to_atoms(u, ts)
                atoms_list.append(atoms)
                timesteps.append(timestep)

            if len(timesteps) > 1:
                io.timestep = timesteps[-1] - timesteps[-2]  # Assuming constant timestep
            else:
                io.timestep = 1

            io.extend(atoms_list)

    def run(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        files = self.config_files + self.mdp_files
        validate_mdp(files[-1])

        self._pack_box()
        self._create_box_gro()

        self._run_acpype()

        self._create_species_top_atomtypes()
        self._create_box_top()
        self._run_gmx()
        self._convert_trajectory()

        if self.cleanup:
            paths = list(self.output_dir.iterdir())
            with (self.output_dir / "info.txt").open("w") as f:
                f.write("The following data has been removed:\n")
                f.write("\n".join([file.name for file in paths]))
            for path in paths:
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
