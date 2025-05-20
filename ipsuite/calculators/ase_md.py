import collections.abc
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
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from tqdm import trange

from ipsuite import base
from ipsuite.abc import NodeWithThermostat
from ipsuite.utils.ase_sim import freeze_copy_atoms, get_energy

log = logging.getLogger(__name__)


class IterationsPerSecondColumn(ProgressColumn):
    def render(self, task):
        if task.finished:
            speed = task.completed / task.finished_time if task.finished_time else 0
        else:
            elapsed = task.elapsed or 0
            speed = task.completed / elapsed if elapsed > 0 else 0
        return Text(f"{speed:5.2f} it/s", style="magenta")


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
    thermostat: NodeWithThermostat = zntrack.deps()

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

    def run_md(self, atoms: ase.Atoms):  # noqa: C901
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

        atoms_cache = []
        self.steps_before_stopping = {"index": None}

        for constraint in self.constraints:
            atoms.set_constraint(constraint.get_constraint(atoms))
        # Replace tqdm tbar with Rich setup
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Progress"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            IterationsPerSecondColumn(),
            transient=True,
        )

        task = progress.add_task("Simulation", total=self.steps)

        def build_info_panel(metrics_dict, i):
            table = Table.grid(padding=(0, 1), expand=True)
            table.add_column(justify="left", style="bold")
            table.add_column(justify="right")

            for key, values in metrics_dict.items():
                if isinstance(values, (list, np.ndarray)) and len(values) > i:
                    val = values[i]
                    if isinstance(val, float):
                        table.add_row(f"{key}:", f"{val:.3f}")
                    else:
                        table.add_row(f"{key}:", str(val))

            return Panel(
                table, title="Simulation Info", border_style="cyan", padding=(1, 2)
            )

        with Live(console=progress.console, refresh_per_second=10) as live:
            for step in range(self.steps):
                self.apply_modifiers(thermostat, step)
                if self.wrap:
                    atoms.wrap()
                try:
                    thermostat.run(1)
                except Exception as e:
                    log.error(f"MD simulation failed: {e}")
                    self.steps_before_stopping = {"index": step}
                    break

                check_trigger = []
                for check in self.checks:
                    check_trigger.append(check.check(atoms))
                    if check_trigger[-1]:
                        log.critical(str(check))
                if any(check_trigger):
                    self.steps_before_stopping = {"index": step}
                    break

                try:
                    metrics_dict = update_metrics_dict(
                        atoms, metrics_dict, self.checks, step
                    )
                except Exception as e:
                    log.error(f"MD simulation failed: {e}")
                    self.steps_before_stopping = {"index": step}
                    break

                if step % self.sampling_rate == 0:
                    atoms_cache.append(freeze_copy_atoms(atoms))
                if len(atoms_cache) == self.dump_rate:
                    self.db.extend(atoms_cache)
                    atoms_cache = []

                progress.update(task, advance=1)
                info_panel = build_info_panel(metrics_dict, step)
                layout = Table.grid(padding=1)
                layout.add_row(progress)
                layout.add_row(info_panel)
                live.update(layout)

        if not self.pop_last and self.steps_before_stopping["index"] is not None:
            metrics_dict = update_metrics_dict(atoms, metrics_dict, self.checks, step)
            atoms_cache.append(freeze_copy_atoms(atoms))
            step += 1

        self.db.extend(atoms_cache)
        return metrics_dict, step

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
    """Similar to the ASEMD node. Instead of terminating the trajectory upon
    triggering a check, the system is reverted to the initial structure and
    the simulation continues with new momenta.
    This is repeated until the maximum number of outer steps is reached.

    Attributes
    ----------
    temperature_reduction_factor: float
        Factor by which the temperature is decreased every time the simulation restarts.
    refresh_calculator: bool
        Whether or not to reinitialize the calculator each time the simulation restarts.
        Turning this on may cause problems for certain calculators (e.g. xTB, Apax).

    """

    temperature_reduction_factor: float = zntrack.params(0.9)
    refresh_calculator: bool = zntrack.params(False)

    def run_md(self, atoms):  # noqa: C901
        rng = np.random.default_rng(self.seed)
        atoms.repeat(self.repeat)
        original_atoms = atoms.copy()

        calc = self.model.get_calculator(directory=self.model_outs)
        atoms.calc = calc

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
                    if self.refresh_calculator:
                        atoms.calc = self.model.get_calculator(directory=self.model_outs)
                    else:
                        atoms.calc = calc
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
