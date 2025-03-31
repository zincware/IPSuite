import pathlib
import typing

import ase
import zntrack
from ase import units
from ase.calculators.plumed import Plumed

import ipsuite as ips


class PlumedCalculator(ips.base.IPSNode):
    data: list[ase.Atoms] = (
        zntrack.deps()
    )  # Plumed only works with a single ase.Atoms object!!!
    data_id: int = zntrack.params(-1)
    model: typing.Any = zntrack.deps()

    input_script_path: str = zntrack.deps_path(None)  # plumed.dat
    input_string: str = zntrack.params(None)

    temperature_K: float = zntrack.params(None)  # in Kelvin! Important for Metadynamics
    timestep: float = zntrack.params(None)

    plumed_directory: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plumed")

    def check_input_instructions(self):
        if self.input_script_path is None and self.input_string is None:
            raise ValueError(
                "No plumed input instructions are specified! Please provide a "
                "filepath to a plumed setupfile or set the `input_string` variable."
            )
        if self.input_script_path is not None and self.input_string is not None:
            raise ValueError(
                "Both `input_script_path` and `input_string` are set. However, "
                "only one of the two can be set at a time!"
            )

        if self.input_script_path is not None:
            with pathlib.Path.open(self.input_script_path, "r") as file:
                self.setup = file.read().splitlines()
        elif self.input_string is not None:
            self.setup = self.input_string

        # needed for ase units:
        units_string = (
            f"""UNITS LENGTH=A TIME={1/(1000 * units.fs)} ENERGY={units.mol/units.kJ}"""
        )
        self.setup.insert(0, units_string)
        with pathlib.Path.open(
            (self.plumed_directory / "setup.dat").as_posix(), "w"
        ) as file:
            for line in self.setup:
                file.write(line + "\n")

    def run(self):
        self.plumed_directory.mkdir(parents=True, exist_ok=True)
        (self.plumed_directory / "outs.txt").write_text("Lorem Ipsum")

    def get_calculator(self, directory: str = None):
        self.check_input_instructions()  # get setup instructions
        return Plumed(
            calc=self.model.get_calculator(),
            atoms=self.data[self.data_id],
            input=self.setup,
            timestep=self.timestep,
            kT=self.temperature_K * units.kB,
            log=(self.plumed_directory / "plumed.log").as_posix(),
        )
