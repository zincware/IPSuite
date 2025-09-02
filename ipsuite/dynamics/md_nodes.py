import logging
import sys
import typing as t
from pathlib import Path

import ase
import h5py
import numpy as np
import pandas as pd
import znh5md
import zntrack
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from laufband import Laufband
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from ipsuite.interfaces import NodeWithCalculator, NodeWithThermostat
from ipsuite.utils.ase_sim import freeze_copy_atoms, get_energy

log = logging.getLogger(__name__)


# TODO: move somewhere else
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


def get_current_metrics(atoms: ase.Atoms, checks: list, time: float, index: int) -> dict:
    """Get current metrics from atoms."""
    temperature, energy = get_energy(atoms)
    metrics = {
        "energy": energy,
        "temperature": temperature,
        "time": time,
        "index": index,
    }
    for check in checks:
        metrics[check.get_quantity()] = check.get_value(atoms)
    return metrics


def build_info_panel(metrics: dict):
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(justify="left", style="bold")
    table.add_column(justify="right")

    for key, val in metrics.items():
        if isinstance(val, float):
            table.add_row(f"{key}:", f"{val:.3f}")
        else:
            table.add_row(f"{key}:", str(val))

    return Panel(table, title="Simulation Info", border_style="cyan", padding=(1, 2))


class ASEMD(zntrack.Node):
    """
    Molecular Dynamics simulation node using ASE.

    Parameters
    ----------
    model : NodeWithCalculator
        The computational model/calculator used for force and energy calculations.
    data : list[ase.Atoms]
        List of atomic structures to simulate.
    data_ids : int or list[int], default -1
        Indices of structures from data to simulate. If -1, simulates all structures.
    checks : list, optional
        List of simulation checks/monitors to apply during the simulation.
    constraints : list, optional
        List of constraints to apply to the atomic system.
    modifiers : list, optional
        List of modifiers to dynamically change simulation parameters.
    thermostat : NodeWithThermostat
        Thermostat object for temperature control during simulation.
    steps : int
        Total number of MD steps to perform.
    sampling_rate : int, default 1
        Frequency of data sampling (every N steps).
    repeat : tuple[bool, bool, bool], default (1, 1, 1)
        Cell repetition factors in x, y, z directions.
    dump_rate : int, default 1000
        Frequency of writing trajectory data to disk.
    use_momenta : bool, default False
        Whether to use existing atomic momenta or initialize with Maxwell-Boltzmann.
    seed : int, default 42
        Random seed for reproducible simulations.

    Attributes
    ----------
    metrics : Path
        Output path for simulation metrics (CSV files).
    frames_path : Path
        Output path for trajectory frames (HDF5 files).
    model_outs : Path
        Output path for model-specific output files.
    laufband_path : Path
        Path to the job queue database file.
    frames : list[ase.Atoms]
        Property that returns all trajectory frames from saved files.
    structures : list[list[ase.Atoms]]
        Property that returns structures organized by simulation run.

    Examples
    --------
    >>> import ipsuite as ips
    >>> project = ips.Project():
    >>> thermostat = ips.LangevinThermostat(temperature=300, friction=0.05, time_step=0.5)
    >>> model = ips.MACEMPModel()
    >>> with project:
    ...     data = ips.AddData(file="seed.xyz")
    ...     md = ips.ASEMD(
    ...         model=model,
    ...         data=data.frames,
    ...         thermostat=thermostat,
    ...         steps=1000,
    ...     )
    >>> project.build()
    """
    model: NodeWithCalculator = zntrack.deps()
    data: list[ase.Atoms] = zntrack.deps()
    data_ids: int | list[int] = zntrack.params(-1)

    checks: list = zntrack.deps(default_factory=list)
    constraints: list = zntrack.deps(default_factory=list)
    modifiers: list = zntrack.deps(default_factory=list)
    thermostat: NodeWithThermostat = zntrack.deps()

    steps: int = zntrack.params()
    sampling_rate: int = zntrack.params(1)
    repeat: t.Tuple[bool, bool, bool] = zntrack.params((1, 1, 1))
    dump_rate: int = zntrack.params(1000)
    use_momenta: bool = zntrack.params(False)
    seed: int = zntrack.params(42)

    metrics: Path = zntrack.outs_path(zntrack.nwd / "metrics")

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames")
    model_outs: Path = zntrack.outs_path(zntrack.nwd / "model")
    laufband_path: Path = zntrack.outs_path(zntrack.nwd / "laufband.sqlite")

    @property
    def frames(self) -> list[ase.Atoms]:
        files = list(self.state.fs.glob((self.frames_path / "*.h5").as_posix()))
        frames = []
        for file in files:
            with self.state.fs.open(file, "rb") as f:
                with h5py.File(f) as file:
                    frames.extend(znh5md.IO(file_handle=file)[:])
        return frames

    @property
    def structures(self) -> list[list[ase.Atoms]]:
        """Return the structures as a list of lists of Atoms."""
        files = list(self.state.fs.glob((self.frames_path / "*.h5").as_posix()))
        structures = []
        for file in files:
            with self.state.fs.open(file, "rb") as f:
                with h5py.File(f) as file:
                    structures.append(znh5md.IO(file_handle=file)[:])
        return structures

    def initialize_md(self):
        self.model_outs.mkdir(parents=True, exist_ok=True)
        self.frames_path.mkdir(parents=True, exist_ok=True)
        self.laufband_path.parent.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        self.rng = np.random.default_rng(self.seed)

    def initialize_atoms(self, idx: int, atoms: ase.Atoms) -> ase.Atoms:
        directory = self.model_outs / f"{idx}"
        directory.mkdir(parents=True, exist_ok=True)

        atoms.repeat(self.repeat)
        atoms.calc = self.model.get_calculator(directory=directory)
        for constraint in self.constraints:
            atoms.set_constraint(constraint.get_constraint(atoms))
        if not self.use_momenta:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=self.thermostat.temperature, rng=self.rng
            )

        return atoms

    def apply_modifiers(self, thermostat, step: int) -> None:
        for modifier in self.modifiers:
            modifier.modify(
                thermostat,
                step=step,
                total_steps=self.steps - 1,  # starting from 0, so we subtract 1
            )

    def initalize_progress_bar(self) -> t.Tuple[Progress, TaskID]:
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
        return progress, task

    def save_metrics(self, metrics_list: list[dict], idx: int) -> None:
        self.metrics.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(metrics_list)
        df.to_csv(self.metrics / f"{idx}.csv", index=False)

    def initialize_checks(self, atoms: ase.Atoms) -> None:
        for check in self.checks:
            check.initialize(atoms)

    def run_md(self, idx: int, atoms: ase.Atoms) -> int:  # noqa: C901
        atoms = self.initialize_atoms(idx=idx, atoms=atoms)
        self.initialize_checks(atoms)
        metrics_list = []

        # initialize thermostat
        thermostat = self.thermostat.get_thermostat(atoms=atoms)

        atoms_cache = []

        progress, task = self.initalize_progress_bar()

        tty_available = sys.stdout.isatty()
        tbar = tqdm(
            range(self.steps),
            desc="Simulation",
            total=self.steps,
            disable=tty_available,  # only show tqdm if rich is not available
            ncols=120,
        )
        io = znh5md.IO(
            self.frames_path / f"{idx}.h5",
        )
        # We do not save the starting configuration. E.g. step 0 is not saved!
        with Live(console=progress.console, refresh_per_second=10) as live:
            for step in tbar:
                self.apply_modifiers(thermostat, step)
                try:
                    thermostat.run(1)
                except Exception as e:
                    log.error(f"MD simulation failed: {e}")
                    break

                check_trigger = []
                for check in self.checks:
                    check_trigger.append(check.check(atoms))
                    if check_trigger[-1]:
                        log.critical(str(check))
                if any(check_trigger):
                    break

                # TODO: only update metrics dict every sampling_rate steps?
                try:
                    metrics = get_current_metrics(
                        atoms, self.checks, step * self.thermostat.time_step, idx
                    )
                except Exception as e:
                    log.error(f"MD simulation failed: {e}")
                    break

                if step % self.sampling_rate == 0:
                    metrics_list.append(metrics)
                    atoms_cache.append(freeze_copy_atoms(atoms))

                if len(atoms_cache) == self.dump_rate:
                    io.extend(atoms_cache)
                    atoms_cache = []

                if tty_available:  # might help with performance a tiny bit
                    progress.update(task, advance=1)
                    info_panel = build_info_panel(metrics)
                    layout = Table.grid(padding=1)
                    layout.add_row(progress)
                    layout.add_row(info_panel)
                    live.update(layout)
                else:
                    time = step * self.thermostat.time_step
                    temperature = metrics["temperature"]
                    energy = metrics["energy"]
                    desc = get_desc(
                        temperature, energy, time, self.steps * self.thermostat.time_step
                    )
                    tbar.set_description(desc)
                    tbar.update(1)

        io.extend(atoms_cache)
        self.save_metrics(metrics_list, idx)
        return step

    def run(self):
        """Run the simulation."""
        self.initialize_md()
        ids = self.data_ids if isinstance(self.data_ids, list) else [self.data_ids]
        worker = Laufband(ids, com=self.laufband_path, disable=True)
        for data_id in worker:
            self.run_md(idx=data_id, atoms=self.data[data_id])


class ASEMDSafeSampling(ASEMD):
    temperature_reduction_factor: float = zntrack.params(0.9)
    # refresh_calculator: bool = zntrack.params(False)
    # # TODO: this won't work with the directory argument,
    # need some way of freeing up the calculator instead.

    def run(self):
        """Run the simulation."""
        if not isinstance(self.data_ids, int):
            raise ValueError(f"{self.__class__.__name__} only supports single data_id")
        self.initialize_md()
        simulated_steps = 0
        idx = 0
        atoms = self.data[self.data_ids]
        full_steps = self.steps - 1
        while simulated_steps < full_steps:
            steps = self.run_md(idx=idx, atoms=atoms.copy())
            simulated_steps += steps
            print(f"Restarting simulation. Missing {full_steps - simulated_steps} steps.")
            self.thermostat.temperature *= self.temperature_reduction_factor
            idx += 1
            self.steps -= steps

        self.laufband_path.write_text("Lorem Ipsum")
