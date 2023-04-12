import logging
import pathlib

import ase.io
import ase.optimize
import zntrack

from ipsuite import base

log = logging.getLogger(__name__)


class ASEGeoOpt(base.ProcessSingleAtomCalc):
    optimizer: str = zntrack.zn.params("FIRE")
    traj: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "optim.traj")

    repeat: list = zntrack.zn.params([1, 1, 1])
    run_kwargs: dict = zntrack.zn.params({"fmax": 0.05})
    init_kwargs: dict = zntrack.zn.params({})

    def run(self):
        atoms = self.get_data()
        atoms = atoms.repeat(self.repeat)
        if self.optimizer is not None:
            atoms.calc = self.get_calc()
            optimizer = getattr(ase.optimize, self.optimizer)

            dyn = optimizer(atoms, trajectory=self.traj.as_posix(), **self.init_kwargs)
            dyn.run(**self.run_kwargs)

    @property
    def atoms(self):
        return list(ase.io.iread(self.traj.as_posix()))
