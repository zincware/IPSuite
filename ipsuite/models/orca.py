import dataclasses
import logging
import os
from pathlib import Path
from ase.calculators.orca import ORCA, OrcaProfile
from ase.calculators.calculator import Calculator


log = logging.getLogger(__name__)

@dataclasses.dataclass
class ORCAModel:
    """ORCA ASE calculator model.

    Parameters
    ----------
    simpleinput : str
        The ORCA input string.
        For example: "B3LYP def2-TZVP enGrad TightSCF" to 
        compute the energy and forces of a system using
        the B3LYP functional with the def2-TZVP basis set.
        See [1]_ for more information.
    blocks : str
        The ORCA blocks string to select the number of processors
        and other settings.
        For example: "%pal nprocs 2 end".
    cmd : str | None
        The command to run ORCA.
        If not set, the environment variable IPSUITE_ORCA_SHELL is used.

    Examples
    --------
    >>> import ipsuite as ips
    >>> project = ips.Project()
    >>> orca = ips.ORCAModel(
    ...     simpleinput="B3LYP def2-TZVP enGrad TightSCF",
    ...     blocks="%pal nprocs 2 end",
    ... )
    >>> with project:
    ...     water = ips.Smiles2Conformers(smiles="O", numConfs=100)
    ...     ips.ApplyCalculator(
    ...         data=water.frames,
    ...         model=orca,
    ...     )
    >>> project.build()

    .. [1] https://orca-manual.mpi-muelheim.mpg.de/index.html
    """
    simpleinput: str = "B3LYP def2-TZVP enGrad TightSCF"
    blocks: str = "%pal nprocs 2 end"
    cmd: str | None = None

    def _update_cmd(self):
        if self.cmd is None:
            try:
                self.cmd = os.environ["IPSUITE_ORCA_SHELL"]
            except KeyError as err:
                raise RuntimeError(
                    "Please set the environment variable "
                    "'IPSUITE_ORCA_SHELL' or set the `cmd`."
                ) from err
        log.info(f"Using IPSUITE_ORCA_SHELL={self.cmd}")

    def get_calculator(self, directory: str | Path, **kwargs) -> Calculator:
        directory = Path(directory) / "orca"
        directory.mkdir(parents=True, exist_ok=True)
        self._update_cmd()

        profile = OrcaProfile(command=self.cmd)

        calc = ORCA(
            profile=profile,
            orcasimpleinput=self.simpleinput,
            orcablocks=self.blocks,
            directory=directory.as_posix(),
        )
        return calc
