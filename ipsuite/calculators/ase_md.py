import functools
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
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tqdm import trange

from ipsuite import base
from ipsuite.utils.ase_sim import freeze_copy_atoms, get_energy, get_desc

log = logging.getLogger(__name__)


class RescaleBoxModifier(base.IPSNode):
    cell: int = zntrack.zn.params()
    _initial_cell = None

    def modify(self, thermostat, step, total_steps):
        # we use the thermostat, so we can also modify e.g. temperature
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


class BoxOscillatingRampModifier(base.IPSNode):
    """Ramp the simulation cell to a specified end cell with some oscillations.

    Attributes
    ----------
    end_cell: float, list[float]
        cell to ramp to, cubic or tetragonal.
    cell_amplitude: float
        amplitude in oscillations of the diagonal cell elements
    num_oscillations: float
        number of oscillations. No oscillations will occur if set to 0.
    interval: int, default 1
        interval in which the box size is changed.
    """

    end_cell: int = zntrack.zn.params()
    cell_amplitude: typing.Union[float, list[float]] = zntrack.zn.params()
    num_oscillations: float = zntrack.zn.params()
    interval: int = zntrack.zn.params(1)
    _initial_cell = None

    def modify(self, thermostat, step, total_steps):
        if self._initial_cell is None:
            self._initial_cell = thermostat.atoms.get_cell()
            if isinstance(self.end_cell, (float, int)):
                self.end_cell = np.array(
                    [[self.end_cell, 0, 0], [0, self.end_cell, 0], [0, 0, self.end_cell]]
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
        ramp = percentage * (self.end_cell - self._initial_cell)
        oscillation = self.cell_amplitude * np.sin(
            2 * np.pi * percentage * self.num_oscillations
        )
        oscillation = np.eye(3) * oscillation
        new_cell = self._initial_cell + ramp + oscillation

        if step % self.interval == 0:
            thermostat.atoms.set_cell(new_cell, scale_atoms=True)


class TemperatureRampModifier(base.IPSNode):
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

    start_temperature: float = zntrack.zn.params(None)
    temperature: float = zntrack.zn.params()
    interval: int = zntrack.zn.params(1)

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


class TemperatureOscillatingRampModifier(base.IPSNode):
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

    start_temperature: float = zntrack.zn.params(None)
    end_temperature: float = zntrack.zn.params()
    temperature_amplitude: float = zntrack.zn.params()
    num_oscillations: float = zntrack.zn.params()
    interval: int = zntrack.zn.params(1)

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


class PressureRampModifier(base.IPSNode):
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

    start_pressure_au: float = zntrack.zn.params(None)
    end_pressure_au: float = zntrack.zn.params()
    interval: int = zntrack.zn.params(1)

    def modify(self, thermostat, step, total_steps):
        if self.start_pressure_au is None:
            self.start_pressure_au = thermostat.externalstress

        frac = step / total_steps
        new_pressure = (-self.start_pressure_au[0]) ** (1 - frac)
        new_pressure *= self.end_pressure_au ** (frac)

        if step % self.interval == 0:
            thermostat.set_stress(new_pressure)


class LangevinThermostat(base.IPSNode):
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

    time_step: int = zntrack.zn.params()
    temperature: float = zntrack.zn.params()
    friction: float = zntrack.zn.params()

    def get_thermostat(self, atoms):
        self.time_step *= units.fs
        thermostat = Langevin(
            atoms=atoms,
            timestep=self.time_step,
            temperature_K=self.temperature,
            friction=self.friction,
        )
        return thermostat


class NPTThermostat(base.IPSNode):
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

    """

    time_step: float = zntrack.zn.params()
    temperature: float = zntrack.zn.params()
    pressure: float = zntrack.zn.params()
    ttime: float = zntrack.zn.params()
    pfactor: float = zntrack.zn.params()
    tetragonal_strain: bool = zntrack.zn.params(True)

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
        return thermostat


class ASEMD(base.ProcessSingleAtom):
    """Class to run a MD simulation with ASE.

    Attributes
    ----------
    atoms_lst: list
        list of atoms objects to start simulation from
    start_id: int
        starting id to pick from list of atoms
    model: zntrack.Node
        A node that implements a 'get_calculation' method
    checker_list: list[CheckNodes]
        checker, which tracks various metrics and stops the
        simulation after a threshold is exceeded.
    thermostat: ase dynamics
        dynamics method used for simulation
    init_temperature: float
        temperature in K to initialize velocities
    init_velocity: np.array()
        starting velocities to continue a simulation
    steps: int
        number of steps to simulate
    sampling_rate: int
        number of sample runs
    metrics_dict:
        saved total energy and all metrics from the check nodes
    repeat: float
        number of repeats
    traj_file: Path
        path where to save the trajectory
    dump_rate: int, default=1000
        Keep a cache of the last 'dump_rate' atoms and
        write them to the trajectory file every 'dump_rate' steps.
    """

    model = zntrack.zn.deps()
    model_outs = zntrack.dvc.outs(zntrack.nwd / "model/")
    checker_list: list = zntrack.zn.nodes(None)
    modifier: list = zntrack.zn.nodes(None)
    thermostat = zntrack.zn.nodes()

    steps: int = zntrack.zn.params()
    init_temperature: float = zntrack.zn.params(None)
    init_velocity = zntrack.zn.params(None)
    sampling_rate = zntrack.zn.params(1)
    repeat = zntrack.zn.params((1, 1, 1))
    dump_rate = zntrack.zn.params(1000)

    metrics_dict = zntrack.zn.plots()

    steps_before_stopping = zntrack.zn.metrics()

    velocity_cache = zntrack.zn.outs()

    traj_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "trajectory.h5")

    def get_constraint(self):
        return []

    def get_atoms(self) -> ase.Atoms:
        atoms: ase.Atoms = self.get_data()
        return atoms.repeat(self.repeat)

    @property
    def atoms(self) -> typing.List[ase.Atoms]:
        def file_handle(filename):
            file = self.state.fs.open(filename, "rb")
            return h5py.File(file)

        return znh5md.ASEH5MD(
            self.traj_file,
            format_handler=functools.partial(
                znh5md.FormatHandler, file_handle=file_handle
            ),
        ).get_atoms_list()

    def run(self):  # noqa: C901
        """Run the simulation."""
        if self.checker_list is None:
            self.checker_list = []
        if self.modifier is None:
            self.modifier = []

        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        atoms = self.get_atoms()
        atoms.calc = self.model.get_calculator(directory=self.model_outs)
        if (self.init_velocity is None) and (self.init_temperature is None):
            self.init_temperature = self.thermostat.temperature

        if self.init_temperature is not None:
            # Initialize velocities
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.init_temperature)
        else:
            # Continue with last md step
            atoms.set_velocities(self.init_velocity)

        # initialize thermostat
        time_step = self.thermostat.time_step
        thermostat = self.thermostat.get_thermostat(atoms=atoms)

        # initialize Atoms calculator and metrics_dict
        _, _ = get_energy(atoms)
        metrics_dict = {"energy": [], "temp": []}
        for checker in self.checker_list:
            checker.initialize(atoms)
            metrics_dict[checker.checker_id()] = []

        # Run simulation
        total_fs = int(self.steps * time_step * self.sampling_rate)

        atoms.set_constraint(self.get_constraint())

        atoms_cache = []

        db = znh5md.io.DataWriter(self.traj_file)
        db.initialize_database_groups()

        with trange(
            total_fs,
            leave=True,
            ncols=120,
        ) as pbar:
            for idx in range(self.steps):
                desc = []
                stop = []
                for modifier in self.modifier:
                    modifier.modify(thermostat, step=idx, total_steps=self.steps)
                thermostat.run(self.sampling_rate)
                temperature, energy = get_energy(atoms)
                metrics_dict["energy"].append(energy)
                metrics_dict["temp"].append(temperature)

                for checker in self.checker_list:
                    stop.append(checker.check(atoms))
                    if stop[-1]:
                        log.critical(str(checker))
                    metric = checker.get_value()
                    metrics_dict[checker.checker_id()].append(metric)
                    
                if "stress" in atoms.calc.implemented_properties:
                    atoms.get_stress()

                atoms_cache.append(freeze_copy_atoms(atoms))
                if len(atoms_cache) == self.dump_rate:
                    db.add(
                        znh5md.io.AtomsReader(
                            atoms_cache,
                            frames_per_chunk=self.dump_rate,
                            step=1,
                            time=self.sampling_rate,
                        )
                    )
                    atoms_cache = []

                if idx % (1 / time_step) == 0:
                    desc = get_desc(temperature, energy)
                    pbar.set_description(desc)
                    pbar.update(self.sampling_rate)

                if any(stop):
                    self.steps_before_stopping = len(metrics_dict["energy"])
                    break

        db.add(
            znh5md.io.AtomsReader(
                atoms_cache,
                frames_per_chunk=self.dump_rate,
                step=1,
                time=self.sampling_rate,
            )
        )

        self.velocity_cache = atoms.get_velocities()
        self.metrics_dict = pd.DataFrame(metrics_dict)

        self.metrics_dict.index.name = "step"
        self.steps_before_stopping = -1


class FixedSphereASEMD(ASEMD):
    """Attributes
    ----------
    atom_id: int
        The id to use as the center of the sphere to fix.
        If None, the closed atom to the center will be picked.
    radius: float
    """

    atom_id = zntrack.zn.params(None)
    selected_atom_id = zntrack.zn.outs()
    radius = zntrack.zn.params()

    def get_constraint(self):
        atoms = self.get_atoms()
        r_ij, d_ij = ase.geometry.get_distances(atoms.get_positions())
        if self.atom_id is not None:
            self.selected_atom_id = self.atom_id
        else:
            _, dist = ase.geometry.get_distances(
                atoms.get_positions(), np.diag(atoms.get_cell() / 2)
            )
            self.selected_atom_id = np.argmin(dist)

        if isinstance(self.selected_atom_id, np.generic):
            self.selected_atom_id = self.selected_atom_id.item()

        indices = np.nonzero(d_ij[self.selected_atom_id] < self.radius)[0]
        return ase.constraints.FixAtoms(indices=indices)
