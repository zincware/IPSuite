import logging
import pathlib
import typing

import ase.io
from ase.io.trajectory import TrajectoryWriter
import ase.optimize
import h5py
import znh5md
import zntrack

from ipsuite import base
from ipsuite.utils.ase_sim import freeze_copy_atoms

log = logging.getLogger(__name__)


class ASEGeoOpt(base.ProcessSingleAtom):
    """Class to run a geometry optimization with ASE.

    Parameters
    ----------
    model: zntrack.Node
        A node that implements 'get_calculator'.
    maxstep: int, optional
        Maximum number of steps to perform.
    """

    model = zntrack.deps()
    model_outs = zntrack.outs_path(zntrack.nwd / "model_outs")
    optimizer: str = zntrack.params("FIRE")
    checks: list = zntrack.deps(None)
    constraints: list = zntrack.deps(None)

    repeat: list = zntrack.params([1, 1, 1])
    run_kwargs: dict = zntrack.params({"fmax": 0.05})
    init_kwargs: dict = zntrack.params({})
    dump_rate = zntrack.params(1000)
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

        atoms = self.get_data()
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

            for checker in self.checks:
                stop.append(checker.check(atoms))
                if stop[-1]:
                    log.critical(
                        f"\n {type(checker).__name__} returned false."
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
    def atoms(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.traj_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]
