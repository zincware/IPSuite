import pathlib
import typing

import ase
import zntrack
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.plumed import Plumed

import ipsuite as ips


class NonOverwritingPlumed(Plumed):
    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        comp = self.compute_energy_and_forces(self.atoms.get_positions(), self.istep)
        energy, forces = comp
        self.istep += 1
        # This line ensures the preservation of important model results!
        self.results = {f"model_{k}": v for k, v in self.calc.results.items()}
        self.results["energy"], self.results["forces"] = energy, forces


class PlumedCalculator(ips.base.IPSNode):
    """Interface for the enhanced-sampling software PLUMED.

    Parameters
    ----------
    data : list[ase.Atoms]
        List of `ase.Atoms` objects representing the system, which will be used
        in combination with the calculator.

    data_id : int, default=-1
        Index of the `ase.Atoms` object from the `data` list that will be used
        to initialize the PLUMED calculator.

    model : typing.Any
        The model to be used with the PLUMED calculator. (Provide a more detailed
        description if applicable.)

    input_script_path : str
        Path to the input script required for PLUMED. Instructions for PLUMED
        can be provided either as a dedicated file (e.g., `plumed.dat`) or as
        a string (see `input_string`). If `input_string` is used, this parameter
        should not be set.

    input_string : str
        Instructions for PLUMED provided as a list of strings instead of a file.
        This parameter must not be set simultaneously with `input_script_path`.

    temperature : float
        Simulation temperature in Kelvin. This parameter is required, even if
        not directly used, and is particularly important for metadynamics.

    timestep : float
        Timestep used in the simulation, in femtoseconds (fs).
    """

    data: list[ase.Atoms] = (
        zntrack.deps()
    )  # Plumed only works with a single ase.Atoms object!!!
    data_id: int = zntrack.params(-1)
    model: typing.Any = zntrack.deps()

    input_script_path: str = zntrack.deps_path(None)  # plumed.dat
    input_string: list[str] = zntrack.params(None)

    temperature: float = zntrack.params(None)  # in Kelvin! Important for Metadynamics
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
        print("got here!!!!")
        # needed for ase units:
        units_string = f"""UNITS LENGTH=A TIME={1 / (1000 * units.fs)} \
            ENERGY={units.mol / units.kJ}"""

        self.setup.insert(0, units_string)
        with pathlib.Path.open(
            (self.plumed_directory / "setup.dat").as_posix(), "w"
        ) as file:
            for line in self.setup:
                file.write(line + "\n")
        print(self.setup)

    def run(self):
        self.plumed_directory.mkdir(parents=True, exist_ok=True)
        (self.plumed_directory / "outs.txt").write_text("Lorem Ipsum")

    def get_calculator(self, directory: str = None):
        self.check_input_instructions()  # get setup instructions
        return NonOverwritingPlumed(
            calc=self.model.get_calculator(),
            atoms=self.data[self.data_id],
            input=self.setup,
            timestep=self.timestep,
            kT=self.temperature_K * units.kB,
            log=(self.plumed_directory / "plumed.log").as_posix(),
        )
