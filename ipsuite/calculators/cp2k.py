"""CP2K interface without ASE calculator.

This interface is less restrictive than CP2K Single Point.
"""
import contextlib
import functools
import pathlib
import shutil
import subprocess
from unittest.mock import patch

import ase.io
import cp2k_output_tools
import pandas as pd
import tqdm
import yaml
import zntrack
from ase.calculators.singlepoint import SinglePointCalculator
from cp2k_input_tools.generator import CP2KInputGenerator


def _update_paths(cp2k_input_dict) -> dict:
    cp2k_input_dict["force_eval"]["DFT"]["basis_set_file_name"] = (
        pathlib.Path(cp2k_input_dict["force_eval"]["DFT"]["basis_set_file_name"])
        .resolve()
        .as_posix()
    )
    cp2k_input_dict["force_eval"]["DFT"]["potential_file_name"] = (
        pathlib.Path(cp2k_input_dict["force_eval"]["DFT"]["potential_file_name"])
        .resolve()
        .as_posix()
    )

    with contextlib.suppress(KeyError):
        cp2k_input_dict["force_eval"]["DFT"]["XC"]["vdw_potential"]["pair_potential"][
            "parameter_file_name"
        ] = (
            pathlib.Path(
                cp2k_input_dict["force_eval"]["DFT"]["XC"]["vdw_potential"][
                    "pair_potential"
                ]["parameter_file_name"]
            )
            .resolve()
            .as_posix()
        )

    with contextlib.suppress(KeyError):
        cp2k_input_dict["force_eval"]["XC"]["vdw_potential"]["non_local"][
            "kernel_file_name"
        ] = (
            pathlib.Path(
                cp2k_input_dict["force_eval"]["XC"]["vdw_potential"]["non_local"][
                    "kernel_file_name"
                ]
            )
            .resolve()
            .as_posix()
        )


class CP2KYamlNode(zntrack.Node):
    """Node for running CP2K Single point calculations."""

    cp2k_bin: str = zntrack.meta.Text("cp2k.psmp")
    cp2k_params = zntrack.dvc.params("cp2k.yaml")
    wfn_restart: str = zntrack.dvc.deps(None)

    cp2k_directory: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "cp2k")

    atoms_file = zntrack.dvc.deps()  # TODO allow both, atoms file and atoms object
    index: int = zntrack.zn.params(-1)

    def run(self):
        """ZnTrack run method."""
        self.cp2k_directory.mkdir(exist_ok=True)
        with open(self.cp2k_params, "r") as file:
            cp2k_input_dict = yaml.safe_load(file)

        atoms = list(ase.io.iread(self.atoms_file))
        atoms = atoms[self.index]
        # TODO assert that they do not exist
        cp2k_input_dict["force_eval"]["subsys"]["topology"] = {
            "COORD_FILE_FORMAT": "XYZ",
            "COORD_FILE_NAME": "atoms.xyz",
        }
        cp2k_input_dict["force_eval"]["subsys"]["cell"] = {
            "A": list(atoms.cell[0]),
            "B": list(atoms.cell[1]),
            "C": list(atoms.cell[2]),
        }

        _update_paths(cp2k_input_dict)

        cp2k_input_script = "\n".join(CP2KInputGenerator().line_iter(cp2k_input_dict))
        with self.operating_directory():
            self._run_cp2k(atoms, cp2k_input_script)

    def _run_cp2k(self, atoms, cp2k_input_script):
        ase.io.write(self.cp2k_directory / "atoms.xyz", atoms)
        input_file = self.cp2k_directory / "input.cp2k"
        input_file.write_text(cp2k_input_script)

        if self.wfn_restart is not None:
            shutil.copy(self.wfn_restart, self.cp2k_directory / "cp2k-RESTART.wfn")
        with subprocess.Popen(
            f"{self.cp2k_bin} -in input.cp2k",
            cwd=self.cp2k_directory,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            with open(self.cp2k_directory / "cp2k.log", "w") as file:
                for line in proc.stdout:
                    file.write(line.decode("utf-8"))
                    print(line.decode("utf-8"), end="")
        if proc.returncode not in [0, None]:
            raise RuntimeError(f"CP2K failed with return code {proc.returncode}:")

    @property
    def atoms(self):
        """Return the atoms object."""
        # TODO this is for single point, what about MD
        data = {}
        with open(self.cp2k_directory / "cp2k.log", "r") as file:
            for match in cp2k_output_tools.parse_iter(file.read()):
                data.update(match)

        forces_df = pd.DataFrame(data["forces"]["atomic"]["per_atom"])
        forces = forces_df[["x", "y", "z"]].values

        atoms = ase.io.read(self.cp2k_directory / "atoms.xyz")
        atoms.calc = SinglePointCalculator(
            atoms=atoms, energy=data["energies"]["total force_eval"], forces=forces
        )

        return atoms


class CP2KSinglePointNode(zntrack.Node):
    """Node for running CP2K Single point calculations."""

    cp2k_shell: str = zntrack.meta.Text("cp2k_shell.ssmp")
    cp2k_params = zntrack.dvc.params("cp2k.yaml")

    atoms = zntrack.zn.deps()
    atoms_file = zntrack.dvc.deps()
    output_file = zntrack.dvc.outs(zntrack.nwd / "atoms.extxyz")
    cp2k_directory = zntrack.dvc.outs(zntrack.nwd / "cp2k")

    def _post_init_(self):
        if self.atoms is None and self.atoms_file is None:
            raise TypeError("Either atoms or atoms_file must not be None")
        if self.atoms is not None and self.atoms_file is not None:
            raise TypeError("Can only use atoms or atoms_file")

    def run(self):
        """ZnTrack run method."""
        if self.atoms_file is not None:
            self.atoms = list(ase.io.iread(self.atoms_file))

        self.cp2k_directory.mkdir(exist_ok=True)
        with open(self.cp2k_params, "r") as file:
            cp2k_input_dict = yaml.safe_load(file)

        _update_paths(cp2k_input_dict)

        cp2k_input_script = "\n".join(CP2KInputGenerator().line_iter(cp2k_input_dict))

        with patch(
            "ase.calculators.cp2k.Popen",
            wraps=functools.partial(subprocess.Popen, cwd=self.cp2k_directory),
        ):
            calculator = ase.calculators.cp2k.CP2K(
                command=self.cp2k_shell,
                inp=cp2k_input_script,
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
            )

            for atom in tqdm.tqdm(self.atoms):
                atom.calc = calculator
                atom.get_potential_energy()
                ase.io.write(self.output_file.as_posix(), atom, append=True)

    @functools.cached_property
    def results(self):
        """Get the Atoms list."""
        # TODO this should probably be a Atoms object.
        return list(ase.io.iread(self.output_file))
