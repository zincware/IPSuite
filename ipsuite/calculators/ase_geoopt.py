import functools
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
    """

    model = zntrack.zn.deps()
    model_outs = zntrack.dvc.outs(zntrack.nwd / "model_outs")
    optimizer: str = zntrack.zn.params("FIRE")
    checker_list: list = zntrack.zn.nodes(None)

    repeat: list = zntrack.zn.params([1, 1, 1])
    run_kwargs: dict = zntrack.zn.params({"fmax": 0.05})
    init_kwargs: dict = zntrack.zn.params({})
    dump_rate = zntrack.zn.params(1000)

    traj_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "trajectory.h5")

    def run(self):
        if self.checker_list is None:
            self.checker_list = []

        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        calculator = self.model.get_calculator(directory=self.model_outs)

        atoms = self.get_data()
        atoms = atoms.repeat(self.repeat)
        atoms.calc = calculator

        atoms_cache = []

        db = znh5md.io.DataWriter(self.traj_file)
        db.initialize_database_groups()

        optimizer = getattr(ase.optimize, self.optimizer)
        dyn = optimizer(atoms, **self.init_kwargs)

        for _ in dyn.irun(**self.run_kwargs):
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


class BatchASEGeoOpt(base.ProcessAtoms):
    """Class to run a geometry optimization with ASE.

    Parameters
    ----------
    model: zntrack.Node
        A node that implements 'get_calculator'.
    """

    model = zntrack.zn.deps()
    model_outs = zntrack.dvc.outs(zntrack.nwd / "model_outs")
    optimizer: str = zntrack.zn.params("FIRE")
    traj: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "optim.traj")
    optimized_structures: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "final.traj")

    repeat: list = zntrack.zn.params([1, 1, 1])
    run_kwargs: dict = zntrack.zn.params({"fmax": 0.05})
    init_kwargs: dict = zntrack.zn.params({})

    def run(self):
        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        calculator = self.model.get_calculator(directory=self.model_outs)
        atoms_list = self.get_data()

        opt_structures = TrajectoryWriter(self.optimized_structures, mode="a")

        for atoms in atoms_list:
            atoms = atoms.repeat(self.repeat)
            if self.optimizer is not None:
                atoms.calc = calculator
                optimizer = getattr(ase.optimize, self.optimizer)

                dyn = optimizer(atoms, trajectory=self.traj.as_posix(), **self.init_kwargs)
                dyn.run(**self.run_kwargs)
                opt_structures.write(atoms)

    @property
    def atoms(self):
        return list(ase.io.iread(self.optimized_structures.as_posix()))
    
    @property
    def trajectories(self):
        return list(ase.io.iread(self.traj.as_posix()))
