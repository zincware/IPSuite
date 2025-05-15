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

log = logging.getLogger(__name__)


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
    config: str | Path  = zntrack.params_path()
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
                    "Please set the environment variable 'IPSUITE_CP2K_SHELL' or set the cp2k executable."
                ) from err

    def get_input_script(self):
        """Return the input script."""
        with Path(self.config).open("r") as file:
            cp2k_input_dict = yaml.safe_load(file)

        return "\n".join(CP2KInputGenerator().line_iter(cp2k_input_dict))

    def get_calculator(self, directory: str | Path, index: int = 0, **kwargs) -> CP2K:
        directory = Path(directory) / f"cp2k-{index}"
        directory.mkdir(parents=True, exist_ok=True)
        self._update_cmd()
        for file in self.files:
            shutil.copy(file, directory)

        patch(
            "ase.calculators.cp2k.subprocess.Popen",
            wraps=functools.partial(subprocess.Popen, cwd=directory),
        ).start()

        return CP2K(
            command=self.cmd,
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
            label=f"cp2k-{index}",
        )
