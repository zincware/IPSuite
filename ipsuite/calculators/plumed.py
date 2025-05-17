import typing as t
from dataclasses import dataclass
from pathlib import Path

import ase
import zntrack
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.plumed import Plumed

from ipsuite.abc import NodeWithCalculator


def cv_to_dataclass(cv: dict):
    """
    Convert a CV dictionary to a PLUMED string.
    """
    cv = cv.copy()
    _type = cv.pop("_type", None)
    if _type == DistanceCV.__name__:
        return DistanceCV(**cv)
    elif _type == TorsionCV.__name__:
        return TorsionCV(**cv)
    else:
        raise ValueError(f"Unknown CV type: {_type}")


# TODO: consider 0index instead of 1index of plumed?
@dataclass
class DistanceCV:
    """DistanceCV class for PLUMED.

    Reference
    ----------
    https://www.plumed.org/doc-master/user-doc/html/DISTANCE/
    """

    name: str
    atoms: list[int]
    grid_min: float | str
    grid_max: float | str
    grid_bin: int

    _type: str = "DistanceCV"

    def __post_init__(self):
        if len(self.atoms) != 2:
            raise ValueError("DistanceCV requires exactly 2 atoms.")
        if not all(isinstance(atom, int) for atom in self.atoms):
            raise TypeError("Atoms must be a list of integers.")

    def to_plumed(self) -> str:
        return f"{self.name}: DISTANCE ATOMS={','.join(map(str, self.atoms))}"

    def __hash__(self):
        return hash((self.name, tuple(self.atoms), self._type))


@dataclass
class TorsionCV:
    """TorsionCV class for PLUMED.

    Reference
    ----------
    https://www.plumed.org/doc-master/user-doc/html/TORSION/
    """

    name: str
    atoms: list[int]
    grid_min: float | str
    grid_max: float | str
    grid_bin: int

    _type: str = "TorsionCV"  # need this, because dataclass.asdict() doesn't include the class name

    def __post_init__(self):
        if len(self.atoms) != 4:
            raise ValueError("TorsionCV requires exactly 4 atoms.")
        if not all(isinstance(atom, int) for atom in self.atoms):
            raise TypeError("Atoms must be a list of integers.")

    def to_plumed(self) -> str:
        return f"{self.name}: TORSION ATOMS={','.join(map(str, self.atoms))}"

    def __hash__(self):
        return hash((self.name, tuple(self.atoms), self._type))


@dataclass
class MetadBias:
    """MetadBias class for PLUMED.

    Reference
    ----------
    https://www.plumed.org/doc-master/user-doc/html/METAD/
    """

    cvs: list[t.Union[DistanceCV, TorsionCV, dict]]
    pace: int = 500
    height: float = 1.2
    sigma: float | list[float] = 0.3
    biasfactor: float = 10
    temp: float = 300
    file: str = "HILLS"
    name: str = "metad"

    def __post_init__(self):
        if isinstance(self.sigma, list):
            if len(self.sigma) != len(self.cvs):
                raise ValueError("Length of SIGMA must match the number of CVs.")

    def to_plumed(self, directory: Path) -> list[str]:
        cvs = [cv_to_dataclass(cv) if isinstance(cv, dict) else cv for cv in self.cvs]
        if not cvs:
            raise ValueError("MetadBias requires at least one CV.")

        arg_names = ",".join(cv.name for cv in cvs)
        file_path = Path(directory) / self.file

        # Prepare comma-separated values for grid settings
        grid_min = ",".join(str(cv.grid_min) for cv in cvs)
        grid_max = ",".join(str(cv.grid_max) for cv in cvs)
        grid_bin = ",".join(str(cv.grid_bin) for cv in cvs)

        if isinstance(self.sigma, list):
            sigma_str = ",".join(str(s) for s in self.sigma)
        else:
            sigma_str = ",".join(str(self.sigma) for _ in self.cvs)

        metad_line = (
            f"{self.name}: METAD ARG={arg_names} "
            f"PACE={self.pace} HEIGHT={self.height} SIGMA={sigma_str} "
            f"BIASFACTOR={self.biasfactor} TEMP={self.temp} "
            f"FILE={file_path.as_posix()} "
            f"GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin}"
        )

        return [metad_line]


@dataclass
class PrintAction:
    cvs: list[t.Union[DistanceCV, TorsionCV]]
    stride: int = 1
    file: str = "COLVAR"

    def to_plumed(self, directory: Path) -> list[str]:
        cvs = [cv_to_dataclass(cv) if isinstance(cv, dict) else cv for cv in self.cvs]
        arg_names = ",".join(cv.name for cv in cvs)
        file_path = Path(directory) / self.file
        return [f"PRINT ARG={arg_names} STRIDE={self.stride} FILE={file_path.as_posix()}"]


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
    actions: list[t.Union[MetadBias, PrintAction]] = zntrack.deps()

    temperature: float = zntrack.params()
    timestep: float = zntrack.params()
    data_id: int = zntrack.params(default=-1)

    def get_calculator(self, directory: str | Path) -> Plumed:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        lines = []
        cvs = []
        for action in self.actions:
            action_cvs = [
                cv_to_dataclass(cv) if isinstance(cv, dict) else cv for cv in action.cvs
            ]
            cvs.extend(action_cvs)
            block = action.to_plumed(directory)

            if isinstance(block, str):
                lines.append(block)
            elif isinstance(block, list):
                lines.extend(block)
            else:
                raise TypeError("Unexpected return type from to_plumed()")

        # Add the unique CVs to the plumed input
        for cv in set(cvs):
            lines.insert(0, cv.to_plumed())
        lines.insert(
            0, f"UNITS LENGTH=A TIME={1 * units.fs} ENERGY={units.mol / units.kJ}"
        )

        # Write plumed input file
        with (directory / "plumed.dat").open("w") as file:
            for line in lines:
                file.write(line + "\n")

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
