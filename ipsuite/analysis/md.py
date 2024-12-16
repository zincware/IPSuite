import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zntrack

from ipsuite import base
from ipsuite.calculators.ase_md import ASEMD
from ipsuite.utils.ase_sim import get_density_from_atoms


class AnalyseDensity(base.AnalyseAtoms):
    window: int = zntrack.params(1000)
    start: int = zntrack.params(0)
    end: int = zntrack.params(None)

    density: dict = zntrack.metrics()
    results: pd.DataFrame = zntrack.plots()

    figure: pathlib.Path = zntrack.plots_path(zntrack.nwd / "density.png")

    def run(self):
        densities = [get_density_from_atoms(x) for x in self.data]

        fig, ax = plt.subplots()
        ax.plot(densities)
        ax.plot(np.convolve(densities, np.ones(self.window) / self.window, mode="valid"))
        ax.set_ylabel(r"Density $\rho$ / kg $\cdot$ m$^{-3}$")
        ax.set_xlabel("Step")
        fig.tight_layout()
        fig.savefig(self.figure)

        self.density = {
            "density": np.mean(densities[self.start : self.end]),
            "std": np.std(densities[self.start : self.end]),
        }

        self.results = pd.DataFrame(densities, columns=["density"])


class CollectMDSteps(base.IPSNode):
    mds: list[ASEMD] = zntrack.deps()
    metrics: dict = zntrack.metrics()

    def run(self):
        steps: list[int] = [x.steps_before_stopping for x in self.mds]

        self.metrics = {
            "total": int(np.sum(steps)),
            "mean": float(np.mean(steps)),
            "std": float(np.std(steps)),
        }
