import logging
import pathlib
import typing

import ase.io
import ase.optimize
import h5py
import znh5md
import zntrack

from ipsuite import base
from ipsuite.utils.ase_sim import freeze_copy_atoms

log = logging.getLogger(__name__)


class ASEGeoOpt(base.IPSNode):
    """Perform a structure relaxation using ASE.

    Use any ASE calculator to perform a geometry optimization on
    a given structure.

    Parameters
    ----------
    data: list[ase.Atoms]
        List of atoms objects to select the starting configuration from.
    data_id: int
        Index of the atoms object to use from ``data``.
    model: zntrack.Node
        A node that implements 'get_calculator'.
    optimizer: str
        The optimizer to use. Default is ``FIRE``.
        Select from ``ase.optimize``, e.g. ``BFGS``, ``FIRE``, ``LBFGS``.
    maxstep: int, optional
        Maximum number of steps to perform.
    checks: list[dataclasses.dataclass], optional
        List of checks to perform during the optimization.
        A failed check will stop the optimization.
    constraints: list[dataclasses.dataclass], optional
        List of constraints to apply to the atoms object.
    run_kwargs: dict, optional
        Keyword arguments to pass to the optimizer when running.
        Default is ``{"fmax": 0.05}``.
    init_kwargs: dict, optional
        Keyword arguments to pass to the optimizer when initializing.
        Default is ``{}``.
    repeat: list[int], optional
        List of integers to repeat the atoms object.
        Default is ``(1, 1, 1)``.
    dump_rate: int, optional
        Number of steps to perform before dumping the current state.
        Default is ``1000``.

    Attributes
    ----------
    model_outs: pathlib.Path
        Path to the directory where the model outputs are stored.
    traj_file: pathlib.Path
        Path to the file where the trajectory is stored.
    """

    data: list[ase.Atoms] = zntrack.deps()
    data_id: int = zntrack.params(-1)

    model: typing.Any = zntrack.deps()
    model_outs: pathlib.Path = zntrack.outs_path(zntrack.nwd / "model_outs")
    optimizer: str = zntrack.params("FIRE")
    checks: list = zntrack.deps(None)
    constraints: list = zntrack.deps(None)

    repeat: list = zntrack.params((1, 1, 1))
    run_kwargs: dict = zntrack.params(default_factory=lambda: {"fmax": 0.05})
    init_kwargs: dict = zntrack.params(default_factory=dict)
    dump_rate: int = zntrack.params(1000)
    maxstep: int = zntrack.params(None)

    traj_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "structures.h5")

    def run(self):
        if self.checks is None:
            self.checks = []
        if self.constraints is None:
            self.constraints = []

        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        calculator = self.model.get_calculator(directory=self.model_outs)

        atoms = self.data[self.data_id]
        atoms = atoms.repeat(self.repeat)
        atoms.calc = calculator

        for constraint in self.constraints:
            atoms.set_constraint(constraint.get_constraint(atoms))

        atoms_cache = []

        db = znh5md.IO(self.traj_file)

        optimizer = getattr(ase.optimize, self.optimizer)
        dyn = optimizer(atoms, **self.init_kwargs)

        for step, _ in enumerate(dyn.irun(**self.run_kwargs)):
            stop = []
            atoms_cache.append(freeze_copy_atoms(atoms))
            if len(atoms_cache) == self.dump_rate:
                db.extend(atoms_cache)
                atoms_cache = []

            for check in self.checks:
                stop.append(check.check(atoms))
                if stop[-1]:
                    log.critical(
                        f"\n {type(check).__name__} returned false."
                        "Simulation was stopped."
                    )

            if any(stop):
                dyn.log()
                break

            if self.maxstep is not None and step >= self.maxstep:
                break

        db.extend(atoms_cache)

    def get_atoms(self) -> ase.Atoms:
        atoms: ase.Atoms = self.get_data()
        return atoms

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.traj_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]
