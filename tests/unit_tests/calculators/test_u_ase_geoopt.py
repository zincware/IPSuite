import ase
import numpy as np
import pytest
import zntrack

import ipsuite as ips
from ipsuite import base


class DebugCheck(base.Check):
    """A check that interrupts the dynamics after a fixed amount of iterations.
    For testing purposes.

    Attributes
    ----------
    n_iterations: int
        number of iterations before stopping
    """

    n_iterations: int = zntrack.params(10)

    def _post_init_(self) -> None:
        self.counter = 0
        self.status = self.__class__.__name__

    def check(self, atoms):
        if self.counter >= self.n_iterations:
            return True
        self.counter += 1
        return False


def test_ase_geoopt(proj_path, cu_box):
    cu_box = cu_box[0]
    cu_box.rattle(0.5)
    ase.io.write("cu_box.xyz", cu_box)

    n_iterations = 5

    check = DebugCheck(n_iterations=n_iterations)

    with ips.Project() as project:
        data = ips.AddData(file="cu_box.xyz")
        model = ips.calculators.LJSinglePoint(data=data.atoms)
        opt = ips.calculators.ASEGeoOpt(
            data=data.atoms,
            model=model,
            optimizer="FIRE",
            checks=[check],
            run_kwargs={"fmax": 0.05},
        )

        opt_max_step = ips.calculators.ASEGeoOpt(
            data=data.atoms,
            model=model,
            optimizer="FIRE",
            checks=[check],
            run_kwargs={"fmax": 0.05},
            maxstep=2,
            name="opt_max_step",
        )

    project.repro()

    assert len(opt.atoms) == n_iterations + 1
    assert len(opt_max_step.atoms) == 3

    forces = np.linalg.norm(opt.atoms[0].get_forces(), 2, 1)
    fmax_start = np.max(forces)

    forces = np.linalg.norm(opt.atoms[-1].get_forces(), 2, 1)
    fmax_end = np.max(forces)

    assert fmax_end < fmax_start
