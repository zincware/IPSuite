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
import znh5md
import zntrack
from ase.calculators.singlepoint import SinglePointCalculator
from cp2k_input_tools.generator import CP2KInputGenerator

from ipsuite import base


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


class CP2KYaml(base.ProcessSingleAtom):
    """Node for running CP2K Single point calculations."""

    cp2k_bin: str = zntrack.meta.Text("cp2k.psmp")
    cp2k_params = zntrack.dvc.params("cp2k.yaml")
    wfn_restart: str = zntrack.dvc.deps(None)

    cp2k_directory: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "cp2k")

    def run(self):
        """ZnTrack run method."""
        self.cp2k_directory.mkdir(exist_ok=True)
        with pathlib.Path(self.cp2k_params).open("r") as file:
            cp2k_input_dict = yaml.safe_load(file)

        atoms = self.get_data()
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
            with (self.cp2k_directory / "cp2k.log").open("w") as file:
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
        with (self.cp2k_directory / "cp2k.log").open("r") as file:
            for match in cp2k_output_tools.parse_iter(file.read()):
                data.update(match)

        forces_df = pd.DataFrame(data["forces"]["atomic"]["per_atom"])
        forces = forces_df[["x", "y", "z"]].values

        atoms = ase.io.read(self.cp2k_directory / "atoms.xyz")
        atoms.calc = SinglePointCalculator(
            atoms=atoms, energy=data["energies"]["total force_eval"], forces=forces
        )

        return atoms


class CP2KSinglePoint(base.ProcessAtoms):
    """Node for running CP2K Single point calculations.

    Parameters
    ----------
    cp2k_shell : str
        The cmd to run cp2k.
    cp2k_params : str
        The path to the cp2k yaml input file. cp2k-input-tools is used to
        generate the input file from the yaml file.
    cp2k_files : str
        Additional dependencies for the cp2k calculation.
    wfn_restart_file : str, optional
        The path to the wfn restart file.
    wfn_restart_node : str, optional
        A cp2k Node that has a wfn restart file.
    """

    cp2k_shell: str = zntrack.meta.Text("cp2k_shell.ssmp")
    cp2k_params = zntrack.dvc.params("cp2k.yaml")
    cp2k_files = zntrack.dvc.deps(None)

    wfn_restart_file: str = zntrack.dvc.deps(None)
    wfn_restart_node = zntrack.zn.deps(None)
    output_file = zntrack.dvc.outs(zntrack.nwd / "atoms.h5")
    cp2k_directory = zntrack.dvc.outs(zntrack.nwd / "cp2k")

    def run(self):
        """ZnTrack run method."""

        db = znh5md.io.DataWriter(self.output_file)
        db.initialize_database_groups()

        calc = self.calc

        for atoms in tqdm.tqdm(self.get_data()):
            atoms.calc = calc
            atoms.get_potential_energy()
            db.add(znh5md.io.AtomsReader([atoms]))

        for file in self.cp2k_directory.glob("cp2k-RESTART.wfn.*"):
            # we don't need all restart files
            file.unlink()

    @property
    def atoms(self):
        """Return the atoms object."""
        return znh5md.ASEH5MD(self.output_file).get_atoms_list()

    def get_input_script(self):
        """Return the input script.

        We use cached_property, because this will also copy the restart file
        to the cp2k directory.
        """
        if not self.cp2k_directory.exists():
            self.cp2k_directory.mkdir(exist_ok=True)

            if self.wfn_restart_file is not None:
                shutil.copy(
                    self.wfn_restart_file, self.cp2k_directory / "cp2k-RESTART.wfn"
                )
            if self.wfn_restart_node is not None:
                shutil.copy(
                    self.wfn_restart_node.cp2k_directory / "cp2k-RESTART.wfn",
                    self.cp2k_directory / "cp2k-RESTART.wfn",
                )

        with pathlib.Path(self.cp2k_params).open("r") as file:
            cp2k_input_dict = yaml.safe_load(file)

        _update_paths(cp2k_input_dict)

        return "\n".join(CP2KInputGenerator().line_iter(cp2k_input_dict))

    @property
    def calc(self):
        """Return the calculator object."""

        # patch.start will patch the object permanently.

        patch(
            "ase.calculators.cp2k.Popen",
            wraps=functools.partial(subprocess.Popen, cwd=self.cp2k_directory),
        ).start()

        return ase.calculators.cp2k.CP2K(
            command=self.cp2k_shell,
            inp=self.get_input_script(),
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
