import typing as t
from pathlib import Path

import ase
import zntrack
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.plumed import Plumed

from ipsuite.abc import NodeWithCalculator


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


class PlumedModel(zntrack.Node):
    data: list[ase.Atoms] = zntrack.deps()
    model: NodeWithCalculator = zntrack.deps()
    config: str|Path = zntrack.deps_path()

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
