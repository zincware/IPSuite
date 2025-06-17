import dataclasses
import functools
import logging
import os
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
import zntrack
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.cp2k import CP2K
from pint import UnitRegistry

from ipsuite.utils.helpers import lower_dict

try:
    from cp2k_input_tools.generator import CP2KInputGenerator
except ImportError as err:
    raise ImportError(
        "IPSuite requires `pip install cp2k-input-tools>0.9.1` due to "
        "incompatibility with pint and numpy 2 in earlier versions. "
        "See https://github.com/cp2k/cp2k-input-tools/issues/110 "
        "You can install the latest version using "
        "`pip install git+https://github.com/cp2k/cp2k-input-tools`."
    ) from err

ureg = UnitRegistry()

log = logging.getLogger(__name__)


@dataclasses.dataclass
class CP2KOutput:
    energy: float
    forces: np.ndarray
    stress: np.ndarray
    hirshfeld_charges: np.ndarray

    def __post_init__(self):
        if not len(self.forces) == len(self.hirshfeld_charges):
            raise ValueError("Length of forces and Hirshfeld charges must match.")

    @staticmethod
    def extract_energy(content: str) -> float:
        """Extract total energy in eV from CP2K output."""
        pattern = re.compile(
            r"ENERGY\|\s+Total FORCE_EVAL \( QS \) energy \[hartree\]\s+([-+]?\d+\.\d+)"
        )
        match = pattern.search(content)
        if not match:
            raise ValueError("Total energy not found in the file content.")
        hartree_energy = float(match.group(1))
        return hartree_energy * ureg("hartree").to("eV").magnitude

    @staticmethod
    def extract_forces(content: str) -> np.ndarray:
        """Extract forces in eV/angstrom from CP2K output."""
        pattern = re.compile(
            r"FORCES\|\s+\d+\s+([-+]?\d+\.\d+E[-+]\d+)\s+"
            r"([-+]?\d+\.\d+E[-+]\d+)\s+([-+]?\d+\.\d+E[-+]\d+)"
        )
        matches = pattern.findall(content)
        if not matches:
            raise ValueError("No forces found in the file content.")
        forces = np.array([[float(x), float(y), float(z)] for x, y, z in matches])
        return forces * ureg("hartree/bohr").to("eV/angstrom").magnitude

    @staticmethod
    def extract_stress_tensor(content: str) -> np.ndarray:
        """Extract 3x3 symmetric stress tensor in ASE Voigt format (eV/Å³)."""
        float_pattern = r"([-+]?\d+\.?\d*(?:[Ee][-+]?\d+))"
        pattern = re.compile(
            rf"^\s*STRESS\|\s*[xyz]\s+{float_pattern}\s+{float_pattern}\s+{float_pattern}$",
            re.MULTILINE,
        )
        rows = pattern.findall(content)
        if len(rows) != 3:
            raise ValueError("Stress tensor not found or incomplete.")

        stress = np.array(rows, dtype=float)
        if not np.allclose(stress, stress.T, atol=1e-6):
            raise ValueError("Stress tensor is not symmetric.")

        stress *= ureg("bar").to("eV/angstrom**3").magnitude
        # Convert to Voigt format: [xx, yy, zz, yz, xz, xy], CP2K uses opposite sign
        return -1.0 * np.array(
            [
                stress[0, 0],
                stress[1, 1],
                stress[2, 2],
                stress[1, 2],
                stress[0, 2],
                stress[0, 1],
            ]
        )

    @staticmethod
    def extract_hirshfeld_charges(content: str) -> np.ndarray:
        """Extract Hirshfeld net atomic charges."""
        header_pattern = re.compile(r"^\s*Hirshfeld Charges\s*$")
        lines = content.splitlines()
        charges = []

        for i, line in enumerate(lines):
            if header_pattern.match(line):
                data_lines = lines[i + 2 :]  # Skip the next two header lines
                for data_line in data_lines:
                    if not data_line.strip():
                        break
                    parts = data_line.split()
                    if len(parts) >= 6:
                        try:
                            charges.append(float(parts[5]))
                        except ValueError:
                            continue
                break

        if not charges:
            raise ValueError("No Hirshfeld charges found in the file content.")
        return np.array(charges, dtype=float)

    @classmethod
    def from_file_content(cls, content: str) -> "CP2KOutput":
        """Construct CP2KOutput from file content string."""
        return cls(
            energy=cls.extract_energy(content),
            forces=cls.extract_forces(content),
            stress=cls.extract_stress_tensor(content),
            hirshfeld_charges=cls.extract_hirshfeld_charges(content),
        )


class CustomCP2K(Calculator):
    """Custom ASE CP2K calculator to allow for custom input scripts."""

    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "charges",
    ]

    def __init__(self, cmd: str, inp: dict, path: str | Path, **kwargs):
        super().__init__(**kwargs)

        inp = lower_dict(inp)
        self._cmd = cmd
        self._inp = inp
        self._path = Path(path)

    def calculate(self, atoms, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        positions = atoms.get_positions()
        config = deepcopy(self._inp)
        # should be ATOM TYPE, X, Y, Z
        cp2k_positions = [
            f"{atom.symbol} {pos[0]} {pos[1]} {pos[2]}"
            for atom, pos in zip(atoms, positions)
        ]
        config["force_eval"]["subsys"]["coord"] = {"*": cp2k_positions}
        config["force_eval"]["subsys"]["cell"] = {
            "periodic": "XYZ",
            "A": atoms.get_cell().tolist()[0],
            "B": atoms.get_cell().tolist()[1],
            "C": atoms.get_cell().tolist()[2],
        }
        config["global"] = {"project_name": "cp2k"}
        # print forces
        config["force_eval"].setdefault("print", {}).setdefault("forces", {"_": "ON"})
        # print analytic stress tensor
        config["force_eval"]["stress_tensor"] = "analytical"
        config["force_eval"].setdefault("print", {}).setdefault(
            "stress_tensor", {"_": "ON"}
        )
        # print hirshfeld charges
        config["force_eval"]["dft"].setdefault("print", {}).setdefault(
            "hirshfeld", {"_": "ON"}
        )

        self._path.mkdir(parents=True, exist_ok=True)
        # keep the previous cp2k.inp and cp2k.out files if they exist
        for ext in ["inp", "out"]:
            existing_file = self._path / f"cp2k.{ext}"
            if existing_file.exists():
                i = 1
                target_file = self._path / f"cp2k_{i}.{ext}"
                while target_file.exists():
                    i += 1
                    target_file = self._path / f"cp2k_{i}.{ext}"
                existing_file.rename(target_file)

        # Write the input script to the cp2k.inp file
        log.info(f"Writing CP2K input script to {self._path / 'cp2k.inp'}")
        with (self._path / "cp2k.inp").open("w") as file:
            file.write("\n".join(CP2KInputGenerator().line_iter(config)))
        log.info(f"Running CP2K in {self._path}")
        result = subprocess.run(
            f"{self._cmd} -i cp2k.inp -o cp2k.out",
            cwd=self._path,
            shell=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"CP2K calculation failed with return code {result.returncode}. "
                f"Check the output in {self._path / 'cp2k.out'}."
            )

        output_file_content = (self._path / "cp2k.out").read_text()
        output = CP2KOutput.from_file_content(output_file_content)
        self.results = {
            "energy": output.energy,
            "forces": output.forces,
            "stress": output.stress,
            "charges": output.hirshfeld_charges,
        }


@dataclasses.dataclass
class CP2KModel:
    """CP2K ASE calculator model.

    Parameters
    ----------
    config : str | Path
        Path to the CP2K input file in YAML format.
        See https://github.com/cp2k/cp2k-input-tools
        for more information on the input file format.
    files : list[str | Path]
        List of files to copy to the cp2k directory.
        These files are typically basis sets and potential files.
    cmd : str | None
        Path to the cp2k executable.
        If not set, the environment variable IPSUITE_CP2K_SHELL is used.

    Examples
    --------
    >>> import ipsuite as ips
    >>> project = ips.Project()
    >>> cp2k = ips.CP2KModel(
    ...     config="cp2k.yaml",
    ...     files=["GTH_BASIS_SETS", "GTH_POTENTIALS"],
    ... )
    >>> with project:
    ...     water = ips.Smiles2Conformers(smiles="O", numConfs=100)
    ...     box = ips.MultiPackmol(
    ...         data=[water.frames], count=[16], density=1000, n_configurations=11,
    ...     )
    ...     ips.ApplyCalculator(
    ...         data=box.frames,
    ...         model=cp2k,
    ...     )
    >>> project.build()
    """

    config: str | Path = zntrack.params_path()
    files: list[str | Path] = zntrack.deps_path(default_factory=list)
    cmd: str | None = None

    def _update_cmd(self):
        if self.cmd is None:
            # Load from environment variable IPSUITE_CP2K_SHELL
            try:
                self.cmd = os.environ["IPSUITE_CP2K_SHELL"]
                log.info(f"Using IPSUITE_CP2K_SHELL={self.cmd}")
            except KeyError as err:
                raise RuntimeError(
                    "Please set the environment variable "
                    "'IPSUITE_CP2K_SHELL' or set the cp2k executable."
                ) from err

    def get_input_script(self) -> dict:
        """Return the input script."""
        with Path(self.config).open("r") as file:
            cp2k_input_dict = yaml.safe_load(file)
        return cp2k_input_dict

    def get_calculator(self, directory: str | Path, **kwargs) -> CP2K | CustomCP2K:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._update_cmd()
        for file in self.files:
            shutil.copy(file, directory)

        if os.environ.get("IPSUITE_CP2K_LEGACY") == "1":
            patch(
                "ase.calculators.cp2k.subprocess.Popen",
                wraps=functools.partial(subprocess.Popen, cwd=directory),
            ).start()

            return CP2K(
                command=self.cmd,
                inp="\n".join(CP2KInputGenerator().line_iter(self.get_input_script())),
                basis_set=None,
                basis_set_file=None,
                max_scf=None,
                cutoff=None,
                force_eval_method=None,
                potential_file=None,
                poisson_solver=None,
                pseudo_potential=None,
                stress_tensor=True,
                xc=None,
                print_level=None,
                label="cp2k",
                set_pos_file=True,
            )
        return CustomCP2K(
            cmd=self.cmd,
            inp=self.get_input_script(),
            path=directory,
        )
