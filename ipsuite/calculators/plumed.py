import pathlib
import typing

import ase
import h5py
import znh5md
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

    input_script_path: str = zntrack.params_path(None)  # plumed.dat
    input_string: str = zntrack.params(None)

    temperature_K: float = zntrack.params(None)  # in Kelvin! Important for Metadynamics
    timestep: float = zntrack.params(None)

    plumed_directory: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plumed")
    plumed_logfile: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plumed.log")
    output_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "structures.h5")

    @property
    def frames(self) -> list[ase.Atoms]:
        with self.state.fs.open(self.output_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]

    def check_input_instructions(self):
        if self.input_script_path is None and self.input_string is None:
            raise ValueError(
                "No plumed input instuctions are specified! Please provide a "
                "filepath to a plumed setupfile or set the `input_string` variable."
            )
        if self.input_script_path is not None and self.input_string is not None:
            raise ValueError(
                "Both `input_script_path` and `input_string` are set. However, "
                "only one of the two can be set at a time!"
            )

        if self.input_script_path is not None:
            with open(self.input_script_path, "r") as file:
                self.setup = file.read().splitlines()
        elif self.input_string is not None:
            self.setup = self.input_string

        # needed for ase units:
        units_string = f"""UNITS LENGTH=A TIME={1/(1000 * units.fs)} 
                        ENERGY={units.mol/units.kJ}"""
        self.setup.insert(0, units_string)

    def run(self):
        self.check_input_instructions()
        self.atoms = self.data[self.data_id]
        db = znh5md.IO(self.output_file)
        self.atoms.calc = self.get_calculator(atoms=self.atoms)
        db.append(self.atoms)

    def get_calculator(self, atoms: ase.Atoms = None, directory: str = None):
        if directory is None:
            directory = self.plumed_directory
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        return Plumed(
            calc=self.model,
            atoms=atoms,
            input=self.setup,
            timestep=self.timestep,
            kT=self.temperature_K * units.kB,
            log=self.plumed_logfile.as_posix(),
        )
