import logging
import pathlib
import typing

import ase.io
import ase.optimize
import h5py
import znh5md
import zntrack

from ipsuite import base
from ipsuite.interfaces import NodeWithCalculator
from ipsuite.utils.ase_sim import freeze_copy_atoms

log = logging.getLogger(__name__)


class ASEGeoOpt(base.IPSNode):
    """Geometry optimization using ASE optimizers.

    Parameters
    ----------
    data : list[ase.Atoms]
        Input atomic configurations to optimize.
    data_id : int, default=-1
        Index of the configuration to optimize from the data list.
    model : NodeWithCalculator
        A model node that implements 'get_calculator' method.
    optimizer : str, default="FIRE"
        Name of the ASE optimizer to use (e.g., "FIRE", "BFGS", "LBFGS").
    repeat : tuple[int, int, int], default=(1, 1, 1)
        Repeat the unit cell in x, y, z directions.
    run_kwargs : dict, default={"fmax": 0.05}
        Keyword arguments passed to the optimizer run method.
    init_kwargs : dict, default={}
        Keyword arguments passed to the optimizer initialization.
    maxstep : int, optional
        Maximum number of optimization steps to perform.
    sampling_rate : int, default=1
        How often to sample atomic configurations during optimization.
    dump_rate : int, default=1000
        How many configurations to cache before writing to file.
    checks : list, optional
        List of check nodes for monitoring optimization progress.
    constraints : list, optional
        List of constraint nodes to apply during optimization.

    Attributes
    ----------
    frames : list[ase.Atoms]
        Optimized atomic configurations from the trajectory.
    traj_file : Path
        Path to the HDF5 trajectory file containing optimization steps.

    Examples
    --------
    >>> model = ips.MACEMPModel()
    >>> with project:
    ...     data = ips.AddData(file="ethanol.xyz")
    ...     geoopt = ips.ASEGeoOpt(data=data.frames, model=model, optimizer="FIRE")
    >>> project.repro()
    >>> print(f"Optimization completed with {len(geoopt.frames)} steps")
    Optimization completed with 74 steps
    """

    data: typing.List[ase.Atoms] = zntrack.deps()
    data_id: int = zntrack.params(-1)

    model: NodeWithCalculator = zntrack.deps()
    model_outs: pathlib.Path = zntrack.outs_path(zntrack.nwd / "model_outs")
    optimizer: str = zntrack.params("FIRE")
    checks: list = zntrack.deps(None)
    constraints: list = zntrack.deps(None)

    repeat: tuple[int, int, int] = zntrack.params((1, 1, 1))
    run_kwargs: dict = zntrack.params(default_factory=lambda: {"fmax": 0.05})
    init_kwargs: dict = zntrack.params(default_factory=dict)
    dump_rate: int = zntrack.params(1000)
    sampling_rate: int = zntrack.params(1)
    maxstep: int = zntrack.params(None)

    traj_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "structures.h5")

    def run(self):  # noqa: C901
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
        for check in self.checks:
            check.initialize(atoms)

        atoms_cache = []

        db = znh5md.IO(self.traj_file)

        optimizer = getattr(ase.optimize, self.optimizer)
        dyn = optimizer(atoms, **self.init_kwargs)

        for step, _ in enumerate(dyn.irun(**self.run_kwargs)):
            stop = []

            if step % self.sampling_rate == 0:
                atoms_cache.append(freeze_copy_atoms(atoms))

            if len(atoms_cache) >= self.dump_rate:
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
        if len(atoms_cache) > 0:
            db.extend(atoms_cache)

    def get_atoms(self) -> ase.Atoms:
        atoms: ase.Atoms = self.get_data()
        return atoms

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.traj_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]
