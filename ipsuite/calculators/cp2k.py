"""CP2K interface without ASE calculator.

This interface is less restrictive than CP2K Single Point.
"""

import functools
import logging
import os
import pathlib
import shutil
import subprocess
import typing
from unittest.mock import patch

import ase.calculators.cp2k
import ase.io
import h5py
import tqdm
import yaml
import znh5md
import zntrack

try:
    from cp2k_input_tools.generator import CP2KInputGenerator
except ImportError as err:
    raise ImportError(
        "Please install the newest development version of cp2k-input-tools: 'pip install"
        " git+https://github.com/cp2k/cp2k-input-tools.git'"
    ) from err


from ipsuite import base

log = logging.getLogger(__name__)


def _update_cmd(cp2k_cmd: str | None, env="IPSUITE_CP2K_SHELL") -> str:
    """Update the shell command to run cp2k."""
    if cp2k_cmd is None:
        # Load from environment variable IPSUITE_CP2K_SHELL
        try:
            cp2k_cmd = os.environ[env]
            log.info(f"Using IPSUITE_CP2K_SHELL={cp2k_cmd}")
        except KeyError as err:
            raise RuntimeError(
                f"Please set the environment variable '{env}' or set the cp2k executable."
            ) from err
    return cp2k_cmd


class CP2KSinglePoint(base.IPSNode):
    """Node for running CP2K Single point calculations.

    Parameters
    ----------
    cp2k_shell : str, default=None
        The cmd to run cp2k. If None, the environment variable
        IPSUITE_CP2K_SHELL is used.
    cp2k_params : str
        The path to the cp2k yaml input file. cp2k-input-tools is used to
        generate the input file from the yaml file.
    cp2k_files : list[str]
        Additional dependencies for the cp2k calculation.
    wfn_restart_file : str, optional
        The path to the wfn restart file.
    wfn_restart_node : str, optional
        A cp2k Node that has a wfn restart file.
    """

    data: list[ase.Atoms] = zntrack.deps()

    cp2k_shell: str | None = zntrack.params(None)
    cp2k_params: str = zntrack.params_path("cp2k.yaml")
    cp2k_files: list[str] = zntrack.deps_path(None)

    wfn_restart_file: str = zntrack.deps_path(None)
    wfn_restart_node: zntrack.Node = zntrack.deps(None)
    output_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "structures.h5")
    cp2k_directory: pathlib.Path = zntrack.outs_path(zntrack.nwd / "cp2k")

    def run(self):
        """ZnTrack run method.

        Raises
        ------
        RuntimeError
            If the cp2k_shell is not set.
        """

        self.cp2k_shell = _update_cmd(self.cp2k_shell)

        db = znh5md.IO(self.output_file)
        calc = self.get_calculator()

        for atoms in tqdm.tqdm(self.data, ncols=70):
            atoms.calc = calc
            atoms.get_potential_energy()
            db.append(atoms)

        for file in self.cp2k_directory.glob("cp2k-RESTART.wfn.*"):
            # we don't need all restart files
            file.unlink()

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.output_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]

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

        return "\n".join(CP2KInputGenerator().line_iter(cp2k_input_dict))

    def get_calculator(self, directory: str = None):
        self.cp2k_shell = _update_cmd(self.cp2k_shell)

        if directory is None:
            directory = self.cp2k_directory
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        else:
            restart_wfn = self.cp2k_directory / "cp2k-RESTART.wfn"
            if restart_wfn.exists():
                pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
                shutil.copy(restart_wfn, directory / "cp2k-RESTART.wfn")

        for file in self.cp2k_files:
            shutil.copy(file, directory / pathlib.Path(file).name)

        patch(
            "ase.calculators.cp2k.subprocess.Popen",
            wraps=functools.partial(subprocess.Popen, cwd=directory),
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
