import dataclasses
import logging
import os
from pathlib import Path
from ase.calculators.orca import ORCA, OrcaProfile
from ase.calculators.calculator import Calculator


log = logging.getLogger(__name__)

@dataclasses.dataclass
class ORCAModel:
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
