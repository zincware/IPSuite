import logging
import pathlib

import ase.io
import ase.optimize
import zntrack

from ipsuite import base

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
    traj: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "optim.traj")

    repeat: list = zntrack.zn.params([1, 1, 1])
    run_kwargs: dict = zntrack.zn.params({"fmax": 0.05})
    init_kwargs: dict = zntrack.zn.params({})

    def run(self):
        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        calculator = self.model.get_calculator(directory=self.model_outs)
        atoms = self.get_data()
        atoms = atoms.repeat(self.repeat)
        if self.optimizer is not None:
            atoms.calc = calculator
            optimizer = getattr(ase.optimize, self.optimizer)

            dyn = optimizer(atoms, trajectory=self.traj.as_posix(), **self.init_kwargs)
            dyn.run(**self.run_kwargs)

    @property
    def atoms(self):
        return list(ase.io.iread(self.traj.as_posix()))
