import functools
import logging
import pathlib
import typing

import ase.io
import ase.optimize
import h5py
import znh5md
import zntrack
from ase.io.trajectory import TrajectoryWriter

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
    checker_list: list = zntrack.deps(None)
    constraint_list: list = zntrack.deps(None)

    repeat: list = zntrack.params([1, 1, 1])
    run_kwargs: dict = zntrack.params({"fmax": 0.05})
    init_kwargs: dict = zntrack.params({})
    dump_rate = zntrack.params(1000)

    maxstep: int = zntrack.params(None)

    traj_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "trajectory.h5")

    def run(self):
        if self.checker_list is None:
            self.checker_list = []
        if self.constraint_list is None:
            self.constraint_list = []

        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        calculator = self.model.get_calculator(directory=self.model_outs)

        atoms = self.get_data()
        atoms = atoms.repeat(self.repeat)
        atoms.calc = calculator

        for constraint in self.constraint_list:
            atoms.set_constraint(constraint.get_constraint(atoms))

        atoms_cache = []

        db = znh5md.io.DataWriter(self.traj_file)
        db.initialize_database_groups()

        optimizer = getattr(ase.optimize, self.optimizer)
        dyn = optimizer(atoms, **self.init_kwargs)

        for step, _ in enumerate(dyn.irun(**self.run_kwargs)):
            stop = []
            atoms_cache.append(freeze_copy_atoms(atoms))
            if len(atoms_cache) == self.dump_rate:
                db.add(
                    znh5md.io.AtomsReader(
                        atoms_cache,
                        frames_per_chunk=self.dump_rate,
                        step=1,
                        time=1,
                    )
                )
                atoms_cache = []

            for checker in self.checker_list:
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

        db.add(
            znh5md.io.AtomsReader(
                atoms_cache,
                frames_per_chunk=self.dump_rate,
                step=1,
                time=1,
            )
        )

    def get_atoms(self) -> ase.Atoms:
        atoms: ase.Atoms = self.get_data()
        return atoms

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
