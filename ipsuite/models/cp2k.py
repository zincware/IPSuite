import dataclasses
import functools
import logging
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import yaml
import zntrack
from ase.calculators.cp2k import CP2K
from cp2k_input_tools.generator import CP2KInputGenerator
from copy import deepcopy
import numpy as np
import re
from pint import UnitRegistry
ureg = UnitRegistry()

log = logging.getLogger(__name__)

from dataclasses import dataclass

@dataclass
class CP2KOutput:
    energy: float
    forces: np.ndarray
    stress: np.ndarray
    hirshfeld_charges: np.ndarray

    @staticmethod
    def extract_forces(content: str) -> np.ndarray:
        pattern = re.compile(
            r"FORCES\|\s+\d+\s+([-+]?\d+\.\d+E[-+]\d+)\s+([-+]?\d+\.\d+E[-+]\d+)\s+([-+]?\d+\.\d+E[-+]\d+)"
        )

        forces = np.array([
            [float(x), float(y), float(z)]
            for x, y, z in pattern.findall(content)
        ])
        # convert forces from cp2k units to ase units
        forces *= ureg("hartree/bohr").to(ureg("eV/angstrom")).magnitude
        return forces 
    
    @staticmethod
    def extract_energy(file_content: str) -> float:
        # use this line  ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -770.620374551554278
        pattern = re.compile(
            r"ENERGY\|\s+Total FORCE_EVAL \( QS \) energy \[hartree\]\s+([-+]?\d+\.\d+)"
        )
        match = pattern.search(file_content)
        if match:
            energy = float(match.group(1))
            # convert energy from hartree to eV
            return energy * ureg("hartree").to(ureg("eV")).magnitude
        else:
            raise ValueError("Total energy not found in the file content.")
        # converged= False
        # for line in file_content.splitlines():
        #     if "*** SCF run converged in" in line:
        #         converged = True
        #     if converged and "Total energy:" in line:
        #         match = re.search(r"Total energy:\s+([-+]?\d+\.\d+)", line)
        #         if match:
        #             energy = float(match.group(1))
        #             # convert energy from hartree to eV
        #             return energy * ureg("hartree").to(ureg("eV")).magnitude
        # raise ValueError("Total energy not found or SCF run did not converge.")

    @staticmethod
    def extract_hirshfeld_charges(file_content: str) -> np.ndarray:
        # Regex to find the line with exactly "Hirshfeld Charges"
        header_pattern = re.compile(r"^\s*Hirshfeld Charges\s*$")
        charges = []
        found = False
        lines = file_content.splitlines()

        for i, line in enumerate(lines):
            if header_pattern.match(line):
                found = True
                # Skip to data lines: we assume the next two lines are headers
                data_lines = lines[i + 2 :]
                for data_line in data_lines:
                    if not data_line.strip():
                        break  # Stop at first empty line
                    parts = data_line.split()
                    if len(parts) >= 6:
                        try:
                            charge = float(parts[5])  # Net charge is the 6th column (index 5)
                            charges.append(charge)
                        except ValueError:
                            continue
                break  # No need to continue iterating after the block

        if not found:
            raise ValueError("Hirshfeld charges section not found in the file content.")
        if not charges:
            raise ValueError("No Hirshfeld charges found in the file content.")

        return np.array(charges, dtype=float)
        

                
    @staticmethod
    def extract_stress_tensor(file_content: str) -> np.ndarray:
        float_pattern = r"([-+]?\d+\.?\d*(?:[Ee][-+]?\d+))"

        tensor_row_pattern = re.compile(
            r"^\s*STRESS\|\s*[xyz]\s*" + 
            float_pattern + r"\s+" + 
            float_pattern + r"\s+" + 
            float_pattern + r"$"
        , re.MULTILINE)

        matches = tensor_row_pattern.findall(file_content)

        if len(matches) == 3:
            stress = np.array(matches, dtype=float)
            # Convert stress tensor from atomic units to eV/angstrom^3
            stress *= ureg("bar").to(ureg("eV/angstrom**3")).magnitude

            assert np.all(stress == np.transpose(stress))  # should be symmetric
            # Convert 3x3 stress tensor to Voigt form as required by ASE
            stress = np.array([stress[0, 0], stress[1, 1], stress[2, 2],
                            stress[1, 2], stress[0, 2], stress[0, 1]])
            return -1.0 * stress  # cp2k uses the opposite sign
        else:
            raise ValueError("Stress tensor not found or incomplete in the file content.")

    @classmethod
    def from_file_content(cls, file_content: str) -> "CP2KOutput":
        energy = cls.extract_energy(file_content)
        forces = cls.extract_forces(file_content)
        stress = cls.extract_stress_tensor(file_content)
        hirshfeld_charges = cls.extract_hirshfeld_charges(file_content)

        return cls(energy=energy, forces=forces, stress=stress, hirshfeld_charges=hirshfeld_charges)


from ase.calculators.calculator import Calculator, all_changes


class CustomCP2K(Calculator):
    """Custom ASE CP2K calculator to allow for custom input scripts."""

    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "charges",
    ]

    def __init__(self, cmd: str, inp: dict, path: str, **kwargs):
        super().__init__(**kwargs)
        # make all keys in inp lowercase, iteratively
        def lower_dict(d):
            return {k.lower(): lower_dict(v) if isinstance(v, dict) else v for k, v in d.items()}
        inp = lower_dict(inp)
        self._cmd = cmd
        self._inp = inp
        self._path = Path(path)

    def calculate(self, atoms, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # Write the cp2k input script to self._path / "cp2k.inp"
        # call cp2k to run the calculation
        # read the results using cp2k_output_tools
        

        positions = atoms.get_positions()
        config = deepcopy(self._inp)
        # should be ATOM TYPE, X, Y, Z
        cp2k_positions = [
            f"{atom.symbol} {pos[0]} {pos[1]} {pos[2]}" for atom, pos in zip(atoms, positions)
        ]
        config["force_eval"]["subsys"]["coord"] = {"*": cp2k_positions}
        config["force_eval"]["subsys"]["cell"] = {"periodic": "XYZ", "A": atoms.get_cell().tolist()[0], 
                                                  "B": atoms.get_cell().tolist()[1], 
                                                  "C": atoms.get_cell().tolist()[2]}
        config["global"] = {"project_name": "cp2k"}
        # print forces
        # config["force_eval"]["print"]["forces"] = {"*": {"output": "FORCE"}}
        config["force_eval"].setdefault("print", {}).setdefault("forces", {"_": "ON"})
        # config["force_eval"].setdefault("print", {}).setdefault("energy", {"_": "ON"})
        config["force_eval"].setdefault("print", {}).setdefault("stress_tensor", {"_": "ON"})
        # config["force_eval"].setdefault("print", {}).setdefault("stress", {"*": {"output": "STRESS"}})
        # config["force_eval"].setdefault("print", {}).setdefault("energy", {"*": {"output": "ENERGY"}})
        # analytic stress tensor
        config["force_eval"]["stress_tensor"] = "analytical"

        # compute hirshfeld charges
        # config["force_eval"]["dft"]["print"] = {
        #     "hirshfeld": {"_": "ON"},
        # }
        config["force_eval"]["dft"].setdefault("print", {}).setdefault("hirshfeld", {"_": "ON"})
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

    def get_input_script(self):
        """Return the input script."""
        with Path(self.config).open("r") as file:
            cp2k_input_dict = yaml.safe_load(file)
        return cp2k_input_dict

        # return "\n".join(CP2KInputGenerator().line_iter(cp2k_input_dict))

    def get_calculator(self, directory: str | Path, **kwargs) -> CP2K:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._update_cmd()
        for file in self.files:
            shutil.copy(file, directory)

        return CustomCP2K(
            cmd=self.cmd,
            inp=self.get_input_script(),
            path=directory,
        )

        # patch(
        #     "ase.calculators.cp2k.subprocess.Popen",
        #     wraps=functools.partial(subprocess.Popen, cwd=directory),
        # ).start()

        # return CP2K(
        #     command=self.cmd,
        #     inp=self.get_input_script(),
        #     basis_set=None,
        #     basis_set_file=None,
        #     max_scf=None,
        #     cutoff=None,
        #     force_eval_method=None,
        #     potential_file=None,
        #     poisson_solver=None,
        #     pseudo_potential=None,
        #     stress_tensor=True,
        #     xc=None,
        #     print_level=None,
        #     label=f"cp2k",
        #     set_pos_file=True,
        # )
