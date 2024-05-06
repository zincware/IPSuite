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
from ipsuite.utils.ase_sim import freeze_copy_atoms, get_box_from_density, get_energy

log = logging.getLogger(__name__)


class RescaleBoxModifier(base.Modifier):
    cell: int = zntrack.params(None)
    density: float = zntrack.params(None)
    _initial_cell = None

    # def _post_init_(self):
    #     super()._post_init_()
    #     if self.density is not None and self.cell is not None:
    #         raise ValueError("Only one of density or cell can be given.")
    #     if self.density is None and self.cell is None:
    #         raise ValueError("Either density or cell has to be given.")
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


class BoxOscillatingRampModifier(base.Modifier):
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

    def _post_init_(self):
        super()._post_init_()
        if self.num_ramp_oscillations is not None:
            if self.num_ramp_oscillations > self.num_oscillations:
                raise ValueError(
                    "num_ramp_oscillations has to be smaller than num_oscillations."
                )

    end_cell: int = zntrack.params(None)
    cell_amplitude: typing.Union[float, list[float]] = zntrack.params()
    num_oscillations: float = zntrack.params()
    num_ramp_oscillations: float = zntrack.params(None)
    interval: int = zntrack.params(1)
    _initial_cell = None

    def modify(self, thermostat, step, total_steps):
        if self.end_cell is None:
            self.end_cell = thermostat.atoms.get_cell()
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


class TemperatureRampModifier(base.Modifier):
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

    start_temperature: float = zntrack.params(None)
    temperature: float = zntrack.params()
    interval: int = zntrack.params(1)

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


class TemperatureOscillatingRampModifier(base.Modifier):
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

    start_temperature: float = zntrack.params(None)
    end_temperature: float = zntrack.params()
    temperature_amplitude: float = zntrack.params()
    num_oscillations: float = zntrack.params()
    interval: int = zntrack.params(1)

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


class PressureRampModifier(base.Modifier):
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

    start_pressure_au: float = zntrack.params(None)
    end_pressure_au: float = zntrack.params()
    interval: int = zntrack.params(1)

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

    time_step: int = zntrack.params()
    temperature: float = zntrack.params()
    friction: float = zntrack.params()

    def get_thermostat(self, atoms):
        thermostat = Langevin(
            atoms=atoms,
            timestep=self.time_step * units.fs,
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

    fraction_traceless: Union[int, float]
        How much of the traceless part of the virial to keep.
        If set to 0, the volume of the cell can change, but the shape cannot.
    """

    time_step: float = zntrack.params()
    temperature: float = zntrack.params()
    pressure: float = zntrack.params()
    ttime: float = zntrack.params()
    pfactor: float = zntrack.params()
    tetragonal_strain: bool = zntrack.params(True)
    fraction_traceless: typing.Union[int, float] = zntrack.params(1)

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


class FixedSphereConstraint(base.IPSNode):
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

    atom_id = zntrack.params(None)
    atom_type = zntrack.params(None)
    radius = zntrack.params()

    def _post_init_(self):
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


class FixedLayerConstraint(base.IPSNode):
    """Class to fix a layer of atoms within a MD
        simulation

    Attributes
    ----------
    upper_limit: float
        all atoms with a lower z pos will be fixed.
    lower_limit: float
        all atoms with a higher z pos will be fixed.
    """

    upper_limit = zntrack.params()
    lower_limit = zntrack.params()

    def get_constraint(self, atoms):
        z_coordinates = atoms.positions[:, 2]

        self.indices = np.where(
            (self.lower_limit <= z_coordinates) & (z_coordinates <= self.upper_limit)
        )[0]

        return ase.constraints.FixAtoms(indices=self.indices)


class FixedBondLengthConstraint(base.IPSNode):
    """Fix the Bondlength between two atoms

    Attributes
    ----------
    atom_id_1: int
        index of atom 1
    atom_id_2: int
        index of atom 2

    Returns
    -------
    ase.constraints.FixBondLengths
        Constraint that fixes the bond Length between atom_id_1 and atom_id_2
    """

    atom_id_1 = zntrack.params()
    atom_id_2 = zntrack.params()

    def get_constraint(self, atoms: ase.Atoms):
        return ase.constraints.FixBondLength(self.atom_id_1, self.atom_id_2)
    

class HookeanConstraint(base.IPSNode):
    """Applies a Hookean (spring) force between a pair of atoms.

    Attributes
    ----------
    atom_id_1: int
        index of atom 1.
    atom_id_2: int
        index of atom 2.
    k: float
        Hookes law (spring) constant to apply when distance exceeds threshold_length. 
        Units of eV A^-2.
    rt: float
        The threshold length below which there is no force. 


    Returns
    -------
    ase.constraints.Hookean
        Constraint that fixes the bond Length between atom_id_1 and atom_id_2
    """

    atom_id_1 = zntrack.params()
    atom_id_2 = zntrack.params()
    k = zntrack.params()
    rt = zntrack.params(None)

    def get_constraint(self, atoms: ase.Atoms):
        return ase.constraints.Hookean(self.atom_id_1, self.atom_id_2, self.k, self.rt)


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
    checks: list[Check]
        checks, which track various metrics and stop the
        simulation if some criterion is met.
    constraints: list[Constraint]
        constrains the atoms within the md simulation
    modifiers: list[Modifier]
    thermostat: ase dynamics
        dynamics method used for simulation
    init_temperature: float
        temperature in K to initialize velocities
    init_velocity: np.array()
        starting velocities to continue a simulation
    steps: int
        total number of steps of the simulation
    sampling_rate: int
        number defines after how many md steps a structure
        is loaded to the cache
    metrics_dict:
        saved total energy and metrics from the check nodes
    repeat: float
        number of repeats
    traj_file: Path
        path where to save the trajectory
    dump_rate: int, default=1000
        Keep a cache of the last 'dump_rate' atoms and
        write them to the trajectory file every 'dump_rate' steps.
    wrap: bool
        Keep the atoms in the cell.
    """

    model = zntrack.deps()

    model_outs = zntrack.outs_path(zntrack.nwd / "model/")
    checks: list = zntrack.deps(None)
    constraints: list = zntrack.deps(None)
    modifiers: list = zntrack.deps(None)
    thermostat = zntrack.deps()

    steps: int = zntrack.params()
    sampling_rate = zntrack.params(1)
    repeat = zntrack.params((1, 1, 1))
    dump_rate = zntrack.params(1000)
    pop_last = zntrack.params(False)
    use_momenta = zntrack.params(False)
    seed: int = zntrack.params(42)
    wrap: bool = zntrack.params(False)

    metrics_dict = zntrack.plots()

    steps_before_stopping = zntrack.metrics()

    traj_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "structures.h5")

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
        np.random.seed(self.seed)

        if self.checks is None:
            self.checks = []
        if self.modifiers is None:
            self.modifiers = []
        if self.constraints is None:
            self.constraints = []

        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        atoms = self.get_atoms()
        atoms.calc = self.model.get_calculator(directory=self.model_outs)

        if not self.use_momenta:
            init_temperature = self.thermostat.temperature
            MaxwellBoltzmannDistribution(atoms, temperature_K=init_temperature)

        # initialize thermostat
        time_step = self.thermostat.time_step
        thermostat = self.thermostat.get_thermostat(atoms=atoms)

        # initialize Atoms calculator and metrics_dict
        metrics_dict = {"energy": [], "temperature": []}
        for checker in self.checks:
            checker.initialize(atoms)
            if checker.get_quantity() is not None:
                metrics_dict[checker.get_quantity()] = []

        # Run simulation
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

        for constraint in self.constraints:
            atoms.set_constraint(constraint.get_constraint(atoms))

        atoms_cache = []

        db = znh5md.io.DataWriter(self.traj_file)
        db.initialize_database_groups()
        self.steps_before_stopping = -1

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
                    for modifier in self.modifiers:
                        modifier.modify(
                            thermostat,
                            step=idx_outer * self.sampling_rate + idx_inner,
                            total_steps=self.steps,
                        )
                    if self.wrap:
                        atoms.wrap()
                    thermostat.run(1)

                    for checker in self.checks:
                        stop.append(checker.check(atoms))
                        if stop[-1]:
                            log.critical(str(checker))

                    if any(stop):
                        break

                if any(stop):
                    self.steps_before_stopping = (
                        idx_outer * self.sampling_rate + idx_inner
                    )
                    break
                else:
                    metrics_dict = update_metrics_dict(atoms, metrics_dict, self.checks)
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

                    time = (idx_outer + 1) * self.sampling_rate * time_step
                    temperature = metrics_dict["temperature"][-1]
                    energy = metrics_dict["energy"][-1]
                    desc = get_desc(temperature, energy, time, total_fs)
                    pbar.set_description(desc)
                    pbar.update(self.sampling_rate)

        if not self.pop_last and self.steps_before_stopping != -1:
            metrics_dict = update_metrics_dict(atoms, metrics_dict, self.checks)
            atoms_cache.append(freeze_copy_atoms(atoms))

        db.add(
            znh5md.io.AtomsReader(
                atoms_cache,
                frames_per_chunk=self.dump_rate,
                step=1,
                time=self.sampling_rate,
            )
        )
        self.metrics_dict = pd.DataFrame(metrics_dict)

        self.metrics_dict.index.name = "step"


def get_desc(temperature: float, total_energy: float, time: float, total_time: float):
    """TQDM description."""
    return (
        f"Temp.: {temperature:.3f} K \t Energy {total_energy:.3f} eV \t Time"
        f" {time:.1f}/{total_time:.1f} fs"
    )


def update_metrics_dict(atoms, metrics_dict, checks):
    temperature, energy = get_energy(atoms)
    metrics_dict["energy"].append(energy)
    metrics_dict["temperature"].append(temperature)
    for checker in checks:
        metric = checker.get_value(atoms)
        if metric is not None:
            metrics_dict[checker.get_quantity()].append(metric)

    return metrics_dict
