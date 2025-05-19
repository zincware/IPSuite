from pathlib import Path

import ase
import zntrack
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.plumed import Plumed

from ipsuite.abc import NodeWithCalculator
from ipsuite.base import IPSNode


class NonOverwritingPlumed(Plumed):
    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = self.compute_energy_and_forces(
            self.atoms.get_positions(), self.istep
        )
        self.istep += 1
        self.results = {f"model_{k}": v for k, v in self.calc.results.items()}
        self.results["energy"], self.results["forces"] = energy, forces


class PlumedModel(IPSNode):
    """Plumed interface.

    Parameters
    ----------
    data: list[ase.Atoms]
        List of ase atoms objects used to initialize the calculator.
    data_id: int
        Index of the ase atoms object to use for initialization.
    model: NodeWithCalculator
        The node that provides the calculator to compute
        unbiased energy and forces.
    config: str | Path
        Path to the plumed input file.
    temperature: float
        Temperature of the simulation in Kelvin.
    timestep: float
        Timestep of the simulation in fs.

    Example
    -------
    An example config file for plumed can look like this:

    .. code-block:: text

        hoh-c: DISTANCE ATOMS=48,3
        c-r1: DISTANCE ATOMS=2,3
        metad: METAD ARG=hoh-c,c-r1 PACE=100 HEIGHT=0.75 SIGMA=0.5,0.5 \
               BIASFACTOR=10 TEMP=400 FILE=HILLS GRID_MIN=1.15,1.15 \
               GRID_MAX=8.0,8.0 GRID_BIN=200,200
        PRINT ARG=hoh-c,c-r1 STRIDE=10 FILE=COLVAR

    References
    ----------
    [1] Plumed manual: https://www.plumed.org/doc-master/user-doc/html/index.html
    [2] Plumed : https://www.plumed.org/

    """

    data: list[ase.Atoms] = zntrack.deps()
    model: NodeWithCalculator = zntrack.deps()
    config: str | Path = zntrack.deps_path()

    temperature: float = zntrack.params()
    timestep: float = zntrack.params()
    data_id: int = zntrack.params(default=-1)

    def get_calculator(self, directory: str | Path) -> Plumed:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with Path(self.config).open("r") as file:
            lines = file.readlines()

        # check if "UNITS" is in any line
        if any("UNITS" in line for line in lines):
            raise ValueError(
                "The plumed input file should not contain the UNITS keyword. "
                "This is automatically added by the PlumedModel."
            )
        # check if "TIME" is in any line
        if any("TIME" in line for line in lines):
            raise ValueError(
                "The plumed input file should not contain the TIME keyword. "
                "This is automatically added by the PlumedModel."
            )
        # check if "ENERGY" is in any line
        if any("ENERGY" in line for line in lines):
            raise ValueError(
                "The plumed input file should not contain the ENERGY keyword. "
                "This is automatically added by the PlumedModel."
            )
        lines.insert(
            0, f"UNITS LENGTH=A TIME={1 * units.fs} ENERGY={units.mol / units.kJ} \n"
        )

        for i, line in enumerate(lines):
            if "FILE=" in line:
                # move file paths to NWD
                lines[i] = line.replace("FILE=", f"FILE={directory}/")

        # Write plumed input file
        with (directory / "plumed.dat").open("w") as file:
            for line in lines:
                file.write(line)

        kT = units.kB * self.temperature

        return NonOverwritingPlumed(
            calc=self.model.get_calculator(),
            atoms=self.data[self.data_id],
            input=lines,
            timestep=float(self.timestep * units.fs),
            kT=float(kT),
            log=(directory / "plumed.log").as_posix(),
        )

    def run(self):
        pass
