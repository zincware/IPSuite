import logging
import pathlib
import typing
import numpy as np

import ase.io
import ase.optimize
from ase.constraints import UnitCellFilter
import h5py
import znh5md
import zntrack

from ipsuite import base
from ipsuite.utils.ase_sim import freeze_copy_atoms


log = logging.getLogger(__name__)


class ASECellOpt(base.IPSNode):
    """Class to run a cell optimization with ASE.

    Parameters
    ----------
    model: zntrack.Node
        A node that implements 'get_calculator'.
    maxstep: int, optional
        Maximum number of steps to perform.
    """

    data: typing.List[ase.Atoms] = zntrack.deps()
    data_id: int = zntrack.params(-1)

    model: typing.Any = zntrack.deps()
    model_outs: pathlib.Path = zntrack.outs_path(zntrack.nwd / "model_outs")
    optimizer: str = zntrack.params("BFGS")
    checks: list = zntrack.deps(None)
    constraints: list = zntrack.deps(None)

    repeat: list = zntrack.params((1, 1, 1))
    run_kwargs: dict = zntrack.params(default_factory=lambda: {"fmax": 0.01})
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
        ucf = UnitCellFilter(atoms, mask=[True, True, True, True, True, True])
        dyn = optimizer(ucf, **self.init_kwargs)

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



class VCSQMN(base.IPSNode):
    """Class to run a cell optimization with ASE.

    Parameters
    ----------
    model: zntrack.Node
        A node that implements 'get_calculator'.
    maxstep: int, optional
        Maximum number of steps to perform.
    """

    data: typing.List[ase.Atoms] = zntrack.deps()
    data_id: int = zntrack.params(-1)

    model: typing.Any = zntrack.deps()
    model_outs: pathlib.Path = zntrack.outs_path(zntrack.nwd / "model_outs")
    checks: list = zntrack.deps(None)
    constraints: list = zntrack.deps(None)

    repeat: list = zntrack.params((1, 1, 1))
    dump_rate: int = zntrack.params(250)
    maxstep: int = zntrack.params(250)
    max_f: float = zntrack.params(0.05)
    
    traj_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "structures.h5")

    def run(self):
        from sqnm.vcsqnm_for_ase import aseOptimizer

        if self.checks is None:
            self.checks = []
        if self.constraints is None:
            self.constraints = []

        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        calculator = self.model.get_calculator(directory=self.model_outs)

        atoms = self.data[self.data_id]

        atoms.calc = calculator

        for constraint in self.constraints:
            atoms.set_constraint(constraint.get_constraint(atoms))

        atoms_cache = []

        db = znh5md.IO(self.traj_file)
        
        dyn = aseOptimizer(atoms, vc_relax=True)

        step = 0
        # for step in range(self.maxstep):
        while(step < self.maxstep and dyn._getDerivativeNorm() > self.max_f):
            if dyn._getDerivativeNorm() <= self.max_f:
                log.info(f'Optimisation converged f_max = {dyn._getDerivativeNorm()}')
                
            stop = []
            
            dyn.step(atoms)
            log.info("Relaxation step: %d energy: %f norm of forces: %f, norm of lattice derivative: %f"% (step, atoms.get_potential_energy(), np.max(np.abs(atoms.get_forces())), np.max(np.abs(dyn._getLatticeDerivative()))) )
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
            step += 1
            
        db.extend(atoms_cache)

    def get_atoms(self) -> ase.Atoms:
        atoms: ase.Atoms = self.get_data()
        return atoms

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.traj_file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]