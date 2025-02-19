import collections.abc
import dataclasses
import logging
import pathlib
import typing

import ase
import ase.constraints
import ase.geometry
import h5py
import numpy as np
import pandas as pd
import znh5md
import zntrack
from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from tqdm import trange

from ipsuite import base
from ipsuite.calculators.integrators import StochasticVelocityCellRescaling
from ipsuite.utils.ase_sim import freeze_copy_atoms, get_box_from_density, get_energy

log = logging.getLogger(__name__)


@dataclasses.dataclass
class RescaleBoxModifier:
    cell: int | None = None
    density: float | None = None
    _initial_cell = None

    def __post_init__(self):
        if self.density is not None and self.cell is not None:
            raise ValueError("Only one of density or cell can be given.")
        if self.density is None and self.cell is None:
            raise ValueError("Either density or cell has to be given.")

    # Currently not possible due to a ZnTrack bug

    def modify(self, thermostat, step, total_steps):
        # we use the thermostat, so we can also modify e.g. temperature
        if self.cell is None:
            self.cell = get_box_from_density([[thermostat.atoms]], [1], self.density)
        if isinstance(self.cell, int):
            self.cell = np.array(
                [[self.cell, 0, 0], [0, self.cell, 0], [0, 0, self.cell]]
            )
        elif isinstance(self.cell, list):
            self.cell = np.array(
                [[self.cell[0], 0, 0], [0, self.cell[1], 0], [0, 0, self.cell[2]]]
            )

        if self._initial_cell is None:
            self._initial_cell = thermostat.atoms.get_cell()
        percentage = step / (total_steps - 1)
        new_cell = (1 - percentage) * self._initial_cell + percentage * self.cell
        thermostat.atoms.set_cell(new_cell, scale_atoms=True)


@dataclasses.dataclass
class BoxOscillatingRampModifier:
    """Ramp the simulation cell to a specified end cell with some oscillations.

    Attributes
    ----------
    end_cell: float, list[float], optional
        cell to ramp to, cubic or tetragonal. If None, the cell will oscillate
        around the initial cell.
    cell_amplitude: float
        amplitude in oscillations of the diagonal cell elements
    num_oscillations: float
        number of oscillations. No oscillations will occur if set to 0.
    interval: int, default 1
        interval in which the box size is changed.
    num_ramp_oscillations: float, optional
        number of oscillations to ramp the box size to the end cell.
        This value has to be smaller than num_oscillations.
        For LotF applications, this can prevent a loop of ever decreasing cell sizes.
        To ensure this use a value of 0.5.
    """

    def __post_init__(self):
        if self.num_ramp_oscillations is not None:
            if self.num_ramp_oscillations > self.num_oscillations:
                raise ValueError(
                    "num_ramp_oscillations has to be smaller than num_oscillations."
                )

    cell_amplitude: typing.Union[float, list[float]]
    num_oscillations: float
    end_cell: int | None = None
    num_ramp_oscillations: float | None = None
    interval: int = 1
    _initial_cell = None

    def modify(self, thermostat, step, total_steps):
        if self.end_cell is None:
            self.end_cell = thermostat.atoms.get_cell()
        if self._initial_cell is None:
            self._initial_cell = thermostat.atoms.get_cell()
            if isinstance(self.end_cell, (float, int)):
                self.end_cell = np.array(
                    [
                        [self.end_cell, 0, 0],
                        [0, self.end_cell, 0],
                        [0, 0, self.end_cell],
                    ]
                )
            elif isinstance(self.end_cell, list):
                self.end_cell = np.array(
                    [
                        [self.end_cell[0], 0, 0],
                        [0, self.end_cell[1], 0],
                        [0, 0, self.end_cell[2]],
                    ]
                )

        percentage = step / (total_steps - 1)
        # if num_ramp_oscillations is set, the cell size is ramped to end_cell within
        # num_ramp_oscillations instead of num_oscillations. This can prevent a loop of
        # ever decreasing cell sizes in LoTF applications where simulations
        # can be aborted at small cell sizes.
        if self.num_ramp_oscillations is not None:
            percentage_per_oscillation = (
                percentage * self.num_oscillations / self.num_ramp_oscillations
            )
            percentage_per_oscillation = min(percentage_per_oscillation, 1)
        else:
            # ramp over all oscillations
            percentage_per_oscillation = percentage

        ramp = percentage_per_oscillation * (self.end_cell - self._initial_cell)
        oscillation = self.cell_amplitude * np.sin(
            2 * np.pi * percentage * self.num_oscillations
        )
        oscillation = np.eye(3) * oscillation
        new_cell = self._initial_cell + ramp + oscillation

        if step % self.interval == 0:
            thermostat.atoms.set_cell(new_cell, scale_atoms=True)


@dataclasses.dataclass
class TemperatureRampModifier:
    """Ramp the temperature from start_temperature to temperature.

    Attributes
    ----------
    start_temperature: float, optional
        temperature to start from, if None, the temperature of the thermostat is used.
    temperature: float
        temperature to ramp to.
    interval: int, default 1
        interval in which the temperature is changed.
    """

    temperature: float
    start_temperature: float | None = None
    interval: int = 1

    def modify(self, thermostat, step, total_steps):
        # we use the thermostat, so we can also modify e.g. temperature
        if self.start_temperature is None:
            # different thermostats call the temperature attribute differently
            if hasattr(thermostat, "temp"):
                start_temperature = thermostat.temp
            elif hasattr(thermostat, "temperature"):
                start_temperature = thermostat.temperature
            self.start_temperature = start_temperature / units.kB

        percentage = step / (total_steps - 1)
        new_temperature = (
            1 - percentage
        ) * self.start_temperature + percentage * self.temperature
        if step % self.interval == 0:
            thermostat.set_temperature(temperature_K=new_temperature)


@dataclasses.dataclass
class TemperatureOscillatingRampModifier:
    """Ramp the temperature from start_temperature to temperature with some oscillations.

    Attributes
    ----------
    start_temperature: float, optional
        temperature to start from, if None, the temperature of the thermostat is used.
    end_temperature: float
        temperature to ramp to.
    temperature_amplitude: float
        amplitude of temperature oscillations.
    num_oscillations: float
        number of oscillations. No oscillations will occur if set to 0.
    interval: int, default 1
        interval in which the temperature is changed.
    """

    end_temperature: float
    temperature_amplitude: float
    num_oscillations: float
    start_temperature: float | None = None
    interval: int = 1

    def modify(self, thermostat, step, total_steps):
        # we use the thermostat, so we can also modify e.g. temperature
        if self.start_temperature is None:
            # different thermostats call the temperature attribute differently
            if hasattr(thermostat, "temp"):
                start_temperature = thermostat.temp
            elif hasattr(thermostat, "temperature"):
                start_temperature = thermostat.temperature
            self.start_temperature = start_temperature / units.kB

        ramp = step / total_steps * (self.end_temperature - self.start_temperature)
        oscillation = self.temperature_amplitude * np.sin(
            2 * np.pi * step / total_steps * self.num_oscillations
        )
        new_temperature = self.start_temperature + ramp + oscillation

        new_temperature = max(0, new_temperature)  # prevent negative temperature

        if step % self.interval == 0:
            thermostat.set_temperature(temperature_K=new_temperature)


@dataclasses.dataclass
class PressureRampModifier:
    """Ramp the temperature from start_temperature to temperature.
    Works only for the NPT thermostat (not NPTBerendsen).

    Attributes
    ----------
    start_pressure_au: float, optional
        pressure to start from, if None, the pressure of the thermostat is used.
        Uses ASE units.
    end_pressure_au: float
        pressure to ramp to. Uses ASE units.
    interval: int, default 1
        interval in which the pressure is changed.
    """

    end_pressure_au: float
    start_pressure_au: float | None = None
    interval: int = 1

    def modify(self, thermostat, step, total_steps):
        if self.start_pressure_au is None:
            self.start_pressure_au = thermostat.externalstress

        frac = step / total_steps
        new_pressure = (-self.start_pressure_au[0]) ** (1 - frac)
        new_pressure *= self.end_pressure_au ** (frac)

        if step % self.interval == 0:
            thermostat.set_stress(new_pressure)


@dataclasses.dataclass
class LangevinThermostat:
    """Initialize the langevin thermostat

    Attributes
    ----------
    time_step: float
        time step of simulation

    temperature: float
        temperature in K to simulate at

    friction: float
        friction of the Langevin simulator

    """

    time_step: int
    temperature: float
    friction: float

    def get_thermostat(self, atoms):
        thermostat = Langevin(
            atoms=atoms,
            timestep=self.time_step * units.fs,
            temperature_K=self.temperature,
            friction=self.friction,
        )
        return thermostat


@dataclasses.dataclass
class VelocityVerletDynamic:
    """Initialize the Velocity Verlet dynamics

    Attributes
    ----------
    time_step: float
        time step of simulation
    """

    time_step: int

    def get_thermostat(self, atoms):
        dyn = VelocityVerlet(
            atoms=atoms,
            timestep=self.time_step * units.fs,
        )
        return dyn


@dataclasses.dataclass
class NPTThermostat:
    """Initialize the ASE NPT barostat
    (Nose Hoover temperature coupling + Parrinello Rahman pressure coupling).

    Attributes
    ----------
    time_step: float
        time step of simulation

    temperature: float
        temperature in K to simulate at

    pressure: float
        pressure in ASE units

    ttime: float
        characteristic temperature coupling time in ASE units

    pfactor: float
        characteristic pressure coupling time in ASE units

    tetragonal_strain: bool
        if True allows only the diagonal elements of the box to change,
        i.e. box angles are constant

    fraction_traceless: Union[int, float]
        How much of the traceless part of the virial to keep.
        If set to 0, the volume of the cell can change, but the shape cannot.
    """

    time_step: float
    temperature: float
    pressure: float
    ttime: float
    pfactor: float
    tetragonal_strain: bool = True
    fraction_traceless: typing.Union[int, float] = 1

    def get_thermostat(self, atoms):
        if self.tetragonal_strain:
            mask = np.array(
                [
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                ]
            )
        else:
            mask = None
        self.time_step *= units.fs
        thermostat = NPT(
            atoms,
            self.time_step,
            temperature_K=self.temperature,
            externalstress=self.pressure,
            ttime=self.ttime,
            pfactor=self.pfactor,
            mask=mask,
        )
        thermostat.set_fraction_traceless(self.fraction_traceless)
        return thermostat


@dataclasses.dataclass
class SVCRBarostat:
    """Initialize the CSVR thermostat

    Attributes
    ----------
    time_step: float
        time step of simulation

    temperature: float
        temperature in K to simulate at
    betaT: float
        Very approximate compressibility of the system.
    pressure_au: float
        Pressure in atomic units.
    taut: float
        Temperature coupling time scale.
    taup: float
        Pressure coupling time scale.
    """

    time_step: int
    temperature: float
    betaT: float = 4.57e-5
    pressure_au: float = 1.01325
    taut: float = 100
    taup: typing.Optional[float] = None

    def get_thermostat(self, atoms):
        if self.taup:
            taup = self.taup * units.fs
        else:
            taup = self.taup

        thermostat = StochasticVelocityCellRescaling(
            atoms=atoms,
            timestep=self.time_step * units.fs,
            temperature_K=self.temperature,
            betaT=self.betaT / units.bar,
            pressure_au=self.pressure_au * units.bar,
            taut=self.taut * units.fs,
            taup=taup,
        )
        return thermostat


@dataclasses.dataclass
class Berendsen:
    """Initialize the Berendsen thermostat

    Attributes
    ----------
    time_step: float
        time step of simulation
    temperature: float
        temperature in K to simulate at
    taut: float
        Temperature coupling time scale.
    """

    time_step: float
    temperature: float
    taut: float = 100

    def get_thermostat(self, atoms):
        thermostat = NVTBerendsen(
            atoms=atoms,
            timestep=self.time_step * units.fs,
            temperature_K=self.temperature,
            taut=self.taut * units.fs,
        )
        return thermostat


@dataclasses.dataclass
class FixedSphereConstraint:
    """Attributes
    ----------
    atom_id: int
        The id to use as the center of the sphere to fix.
        If None, the closed atom to the center will be picked.
    atom_type: str, optional
        The type of the atom to fix. E.g. if
        atom_type = H, atom_id = 1, the first
        hydrogen atom will be fixed. If None,
        the first atom will be fixed, no matter the type.
    radius: float
    """

    radius: float
    atom_id: int | None = None
    atom_type: str | None = None

    def __post_init__(self):
        if self.atom_type is not None and self.atom_id is None:
            raise ValueError("If atom_type is given, atom_id must be given as well.")

    def get_selected_atom_id(self, atoms: ase.Atoms) -> int:
        if self.atom_type is not None:
            return np.where(np.array(atoms.get_chemical_symbols()) == self.atom_type)[0][
                self.atom_id
            ]

        elif self.atom_id is not None:
            return self.atom_id
        else:
            _, dist = ase.geometry.get_distances(
                atoms.get_positions(), np.diag(atoms.get_cell() / 2)
            )
            return np.argmin(dist)

    def get_constraint(self, atoms):
        r_ij, d_ij = ase.geometry.get_distances(
            atoms.get_positions(), cell=atoms.cell, pbc=True
        )
        selected_atom_id = self.get_selected_atom_id(atoms)

        indices = np.nonzero(d_ij[selected_atom_id] < self.radius)[0]
        return ase.constraints.FixAtoms(indices=indices)


@dataclasses.dataclass
class FixedLayerConstraint:
    """Class to fix a layer of atoms within a MD
        simulation

    Attributes
    ----------
    upper_limit: float
        all atoms with a lower z pos will be fixed.
    lower_limit: float
        all atoms with a higher z pos will be fixed.
    """

    upper_limit: float
    lower_limit: float

    def get_constraint(self, atoms):
        z_coordinates = atoms.positions[:, 2]

        self.indices = np.where(
            (self.lower_limit <= z_coordinates) & (z_coordinates <= self.upper_limit)
        )[0]

        return ase.constraints.FixAtoms(indices=self.indices)


def get_desc(temperature: float, total_energy: float, time: float, total_time: float):
    """TQDM description."""
    return (
        f"Temp.: {temperature:.3f} K \t Energy {total_energy:.3f} eV \t Time"
        f" {time:.1f}/{total_time:.1f} fs"
    )


def update_metrics_dict(atoms, metrics_dict, checks, step):
    temperature, energy = get_energy(atoms)
    metrics_dict["energy"].append(energy)
    metrics_dict["temperature"].append(temperature)
    metrics_dict["step"].append(step)
    for check in checks:
        metric = check.get_value(atoms)
        if metric is not None:
            metrics_dict[check.get_quantity()].append(metric)

    return metrics_dict


class ASEMD(base.IPSNode):
    """Class to run a MD simulation with ASE.

    Attributes
    ----------
    model: zntrack.Node
        A node that implements a 'get_calculation' method
    data: list[ase.Atoms]
        The atoms data to process. This must be an input to the Node.
        It can either a single atoms object or a list of atoms objects
        with a given 'data_id'.
    data_id: int | -1
        The id of the atoms object to process. If None, the last
        atoms object is used. Only relevant if 'data' is a list.
    data_ids: list[int] | None
        The ids of the atoms object to process. Only relevant if the
        mapped function is used.
        ```
        mapped_asemd = zn.apply(ips.ASEMD, method='map')(**kwargs)
        ```
    checks: list[Check]
        checks, which track various metrics and stop the
        simulation if some criterion is met.
    constraints: list[Constraint]
        constrains the atoms within the md simulation.
    modifiers: list[Modifier]
        modifies e.g. temperature or cell during the simulation.
    thermostat: ase dynamics
        dynamics method used for simulation
    steps: int
        total number of steps of the simulation
    sampling_rate: int
        number defines after how many md steps a structure
        is loaded to the cache
    repeat: float
        number of repeats
    dump_rate: int, default=1000
        Keep a cache of the last 'dump_rate' atoms and
        write them to the trajectory file every 'dump_rate' steps.
    pop_last : bool
        Option to pop last, default false.
    use_momenta : bool
        Option to use momenta to init the simulation, default false.
    seed : int
        Random seed for the simulation.
    wrap: bool
        Keep the atoms in the cell if true, default false.
    """

    model: typing.Any = zntrack.deps()

    data: list[ase.Atoms] = zntrack.deps()

    data_id: typing.Optional[int] = zntrack.params(-1)
    data_ids: typing.Optional[int] = zntrack.params(None)

    model_outs: pathlib.Path = zntrack.outs_path(zntrack.nwd / "model/")
    checks: list = zntrack.deps(None)
    constraints: list = zntrack.deps(None)
    modifiers: list = zntrack.deps(None)
    thermostat: typing.Any = zntrack.deps()

    steps: int = zntrack.params()
    sampling_rate: int = zntrack.params(1)
    repeat: typing.Tuple[bool, bool, bool] = zntrack.params((1, 1, 1))
    dump_rate: int = zntrack.params(1000)
    pop_last: bool = zntrack.params(False)
    use_momenta: bool = zntrack.params(False)
    seed: int = zntrack.params(42)
    wrap: bool = zntrack.params(False)

    metrics_dict: pd.DataFrame = zntrack.plots()

    steps_before_stopping: dict = zntrack.metrics()

    structures: typing.Any = zntrack.outs()
    traj_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "structures.h5")

    def get_atoms(self, method="run") -> ase.Atoms | typing.List[ase.Atoms]:
        """Get the atoms object to process given the 'data' and 'data_id'.

        Returns
        -------
        ase.Atoms | list[ase.Atoms]
            The atoms object to process
        """
        if self.data is not None:
            if isinstance(self.data, (list, collections.abc.Sequence)):
                atoms = self.data.copy()
            else:
                atoms = list(self.data.copy())
        else:
            raise ValueError("No data given.")

        if method == "run":
            return atoms[self.data_id]
        else:
            return atoms

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.traj_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]

    def initialize_md(self):
        # np.random.seed(self.seed)

        if self.checks is None:
            self.checks = []
        if self.modifiers is None:
            self.modifiers = []
        if self.constraints is None:
            self.constraints = []

        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")

        self.db = znh5md.IO(self.traj_file)

    def initialize_metrics(self, atoms):
        metrics_dict = {
            "energy": [],
            "temperature": [],
            "step": [],
        }
        for check in self.checks:
            check.initialize(atoms)
            if check.get_quantity() is not None:
                metrics_dict[check.get_quantity()] = []

        return metrics_dict

    def adjust_sim_time(self, time_step):
        sampling_iterations = self.steps / self.sampling_rate
        if sampling_iterations % 1 != 0:
            sampling_iterations = np.round(sampling_iterations)
            self.steps = int(sampling_iterations * self.sampling_rate)
            log.warning(
                "The sampling_rate is not a devisor of steps."
                f"Steps were adjusted to {self.steps}"
            )
        sampling_iterations = int(sampling_iterations)
        total_fs = self.steps * time_step
        return sampling_iterations, total_fs

    def apply_modifiers(self, thermostat, current_inner_step):
        for modifier in self.modifiers:
            modifier.modify(
                thermostat,
                step=current_inner_step,
                total_steps=self.steps,
            )

    def run_md(self, atoms):  # noqa: C901
        rng = np.random.default_rng(self.seed)
        atoms.repeat(self.repeat)
        atoms.calc = self.model.get_calculator(directory=self.model_outs)

        init_temperature = self.thermostat.temperature
        if not self.use_momenta:
            MaxwellBoltzmannDistribution(atoms, temperature_K=init_temperature, rng=rng)

        # initialize thermostat
        time_step = self.thermostat.time_step
        thermostat = self.thermostat.get_thermostat(atoms=atoms)

        metrics_dict = self.initialize_metrics(atoms)
        sampling_iterations, total_fs = self.adjust_sim_time(time_step)

        for constraint in self.constraints:
            atoms.set_constraint(constraint.get_constraint(atoms))

        # Run simulation
        atoms_cache = []
        self.steps_before_stopping = -1
        current_step = 0
        with trange(
            self.steps,
            leave=True,
            ncols=120,
        ) as pbar:
            for idx_outer in range(sampling_iterations):
                desc = []
                stop = []

                # run MD for sampling_rate steps
                for idx_inner in range(self.sampling_rate):
                    self.apply_modifiers(
                        thermostat, idx_outer * self.sampling_rate + idx_inner
                    )

                    if self.wrap:
                        atoms.wrap()

                    thermostat.run(1)

                for check in self.checks:
                    stop.append(check.check(atoms))
                    if stop[-1]:
                        log.critical(str(check))

                if any(stop):
                    self.steps_before_stopping = (
                        idx_outer * self.sampling_rate + idx_inner
                    )
                    break
                else:
                    metrics_dict = update_metrics_dict(
                        atoms, metrics_dict, self.checks, current_step
                    )
                    atoms_cache.append(freeze_copy_atoms(atoms))
                    if len(atoms_cache) == self.dump_rate:
                        self.db.extend(atoms_cache)
                        atoms_cache = []

                    time = (idx_outer + 1) * self.sampling_rate * time_step
                    temperature = metrics_dict["temperature"][-1]
                    energy = metrics_dict["energy"][-1]
                    desc = get_desc(temperature, energy, time, total_fs)
                    pbar.set_description(desc)
                    pbar.update(self.sampling_rate)
                    current_step += 1

        if not self.pop_last and self.steps_before_stopping != -1:
            metrics_dict = update_metrics_dict(
                atoms, metrics_dict, self.checks, current_step
            )
            atoms_cache.append(freeze_copy_atoms(atoms))
            current_step += 1

        self.db.extend(atoms_cache)
        return metrics_dict, current_step

    def run(self):
        """Run the simulation."""
        self.initialize_md()

        atoms = self.get_atoms()
        metrics_dict, _ = self.run_md(atoms=atoms)

        self.structures = []

        self.metrics_dict = pd.DataFrame(metrics_dict)

    def map(self):  # noqa: A003
        self.initialize_md()

        metrics_list = []
        if self.data_ids is not None:
            structures = [self.get_atoms(method="map")[idx] for idx in self.data_ids]
        else:
            structures = self.get_atoms(method="map")

        self.structures = []
        for atoms in structures:
            metrics, current_step = self.run_md(atoms=atoms)
            metrics_list.append(metrics)
            self.structures.append(self.frames[-current_step:])

        # Flatten metrics dictionary
        flattened_metrics = {}
        for key in metrics_list[0].keys():
            flattened_metrics[key] = []

        for metrics in metrics_list:
            for key, value in metrics.items():
                flattened_metrics[key].extend(value)

        self.metrics_dict = pd.DataFrame(flattened_metrics)


class ASEMDSafeSampling(ASEMD):
    temperature_reduction_factor: float = zntrack.params(0.9)

    def run_md(self, atoms):  # noqa: C901
        rng = np.random.default_rng(self.seed)
        atoms.repeat(self.repeat)
        original_atoms = atoms.copy()

        atoms.calc = self.model.get_calculator(directory=self.model_outs)

        init_temperature = self.thermostat.temperature
        # if not self.use_momenta:
        MaxwellBoltzmannDistribution(atoms, temperature_K=init_temperature)

        # initialize thermostat
        time_step = self.thermostat.time_step
        thermostat = self.thermostat.get_thermostat(atoms=atoms)

        metrics_dict = self.initialize_metrics(atoms)
        sampling_iterations, total_fs = self.adjust_sim_time(time_step)

        for constraint in self.constraints:
            atoms.set_constraint(constraint.get_constraint(atoms))

        # Run simulation
        atoms_cache = []
        self.steps_before_stopping = -1
        current_step = 0
        with trange(
            self.steps,
            leave=True,
            ncols=120,
        ) as pbar:
            for idx_outer in range(sampling_iterations):
                desc = []
                stop = []

                # run MD for sampling_rate steps
                for idx_inner in range(self.sampling_rate):
                    self.apply_modifiers(
                        thermostat, idx_outer * self.sampling_rate + idx_inner
                    )

                    if self.wrap:
                        atoms.wrap()

                    thermostat.run(1)

                for check in self.checks:
                    stop.append(check.check(atoms))
                    if stop[-1]:
                        log.critical(str(check))

                if any(stop):
                    atoms = original_atoms.copy()
                    atoms.calc = self.model.get_calculator(directory=self.model_outs)
                    init_temperature *= self.temperature_reduction_factor
                    MaxwellBoltzmannDistribution(
                        atoms, temperature_K=init_temperature, rng=rng
                    )
                    thermostat = self.thermostat.get_thermostat(atoms=atoms)
                    thermostat.set_temperature(temperature_K=init_temperature)

                else:
                    metrics_dict = update_metrics_dict(
                        atoms, metrics_dict, self.checks, current_step
                    )
                    atoms_cache.append(freeze_copy_atoms(atoms))
                    if len(atoms_cache) == self.dump_rate:
                        self.db.extend(atoms_cache)
                        atoms_cache = []

                    time = (idx_outer + 1) * self.sampling_rate * time_step
                    temperature = metrics_dict["temperature"][-1]
                    energy = metrics_dict["energy"][-1]
                    desc = get_desc(temperature, energy, time, total_fs)
                    pbar.set_description(desc)
                    pbar.update(self.sampling_rate)
                    current_step += 1

        if not self.pop_last and self.steps_before_stopping != -1:
            metrics_dict = update_metrics_dict(
                atoms, metrics_dict, self.checks, current_step
            )
            atoms_cache.append(freeze_copy_atoms(atoms))
            current_step += 1

        self.db.extend(atoms_cache)
        return metrics_dict, current_step
