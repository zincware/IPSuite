import logging
import pathlib

import ase.optimize
import zntrack

from ipsuite import base

log = logging.getLogger(__name__)


class ASEGeoOpt(base.ProcessSingleAtom):
    calculator = zntrack.zn.deps()

    optimizer: str = zntrack.zn.params("FIRE")
    traj: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "optim.traj")

    repeat: list = zntrack.zn.params([1, 1, 1])
    run_kwargs: dict = zntrack.zn.params({"fmax": 0.05})
    init_kwargs: dict = zntrack.zn.params({})

    def run(self):
        atoms = self.get_data()
        atoms = atoms.repeat(self.repeat)
        if self.optimizer is not None:
            atoms.calc = self.calculator
            optimizer = getattr(ase.optimize, self.optimizer)

            dyn = optimizer(atoms, trajectory=self.traj.as_posix(), **self.init_kwargs)
            dyn.run(**self.run_kwargs)
        self.atoms = [atoms]
