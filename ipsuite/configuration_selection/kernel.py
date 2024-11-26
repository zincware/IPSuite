"""Use a Kernel and some initial configuration to select further configurations."""

from __future__ import annotations

import logging
import typing

import ase
import numpy as np
import tqdm
import zntrack

from ipsuite.configuration_selection.base import ConfigurationSelection

if typing.TYPE_CHECKING:
    import ipsuite


log = logging.getLogger(__name__)


class KernelSelection(ConfigurationSelection):
    """Use the chosen kernel to selected configurations furthes apart.

    Attributes
    ----------
    n_configurations: int
        number of configurations to select
    kernel: ConfigurationComparison = zntrack.Nodes()
    points_per_cycle: int
        Number of configurations to add before recomputing the MMK
    correlation_time: int
        Ideally the correlation time of the data to only sample from uncorrelated data.
        This will only sample from configurations that are configuration_time apart.
        The smaller, the slower is the selection but the number of looked at
        configuration is larger giving potentially better results.
    seed: int
        seed selection in case of random picking initial configuration
    threshold: float
        threshold to stop the selection. The maximum number of configurations
        is still n_configurations. If the threshold is reached before, the
        selection stops. Typical values can be 0.995
    """

    n_configurations: int = zntrack.params()
    # kernel: "ipsuite.configuration_comparison.ConfigurationComparison" = zntrack.deps()
    initial_configurations: typing.List[ase.Atoms] = zntrack.deps(None)
    points_per_cycle: int = zntrack.params(1)
    kernel_results: typing.List[typing.List[float]] = zntrack.outs()
    seed: int = zntrack.params(1234)
    threshold: float = zntrack.params(None)

    # TODO what if the correlation time restricts the number of atoms to
    #  be less than n_configurations?
    correlation_time: int = zntrack.params(1)

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Atom Selection method.

        Parameters
        ----------
        atoms_lst: typing.List[ase.Atoms]
            list of atoms objects

        Returns
        -------
        typing.List[int]:
            list containing the taken indices
        """
        if self.initial_configurations is None:
            np.random.seed(self.seed)
            initial_configurations = [atoms_lst[np.random.randint(len(atoms_lst))]]
            self.n_configurations -= 1
        else:
            initial_configurations = self.initial_configurations
        selected_atoms = []
        # we don't change the analyte, so we don't want to recompute the
        # SOAP vector every time.
        self.kernel.analyte = atoms_lst[:: self.correlation_time]
        self.kernel.remove_database = False
        self.kernel.load_analyte = False
        self.kernel.disable_tqdm = True

        self.kernel_results = []
        hist_results = []

        # TODO do not use the atoms in atoms_list but store the ids directly
        try:
            for idx in tqdm.trange(self.n_configurations, ncols=70):
                self.kernel.reference = initial_configurations + selected_atoms
                self.kernel.run()

                minimum_indices = np.argsort(self.kernel.result)[: self.points_per_cycle]
                selected_atoms += [self.kernel.analyte[x.item()] for x in minimum_indices]
                # There is currently no check in place to ensure that an atom is only
                # selected once. This should inherently be ensured by the way the
                # MMK selects configurations.
                self.kernel.load_analyte = True
                self.kernel_results.append(self.kernel.result)
                if self.threshold is not None:
                    hist, bins = np.histogram(
                        self.kernel.result, bins=np.linspace(0, 1, 10000), density=True
                    )
                    bins = bins[:-1]
                    hist[bins > self.threshold] = 0
                    hist_results.append(np.trapz(hist, bins))
                    if hist_results[-1] == 0:
                        log.warning(
                            f"Threshold {self.threshold} reached before"
                            f" {self.n_configurations} configurations were selected -"
                            f" stopping after selecting {idx + 1} configurations."
                        )
                        break
        finally:
            self.kernel.unlink_database()

        if self.initial_configurations is None:
            # include the randomly selected configuration
            selected_atoms += initial_configurations
            self.n_configurations += 1

        selected_ids = [
            idx for idx, atom in enumerate(atoms_lst) if atom in selected_atoms
        ]
        if self.threshold is None:
            if len(selected_ids) != self.n_configurations:
                print(f"{self.initial_configurations = }")
                raise ValueError(
                    f"Unable to select {self.n_configurations}. Could only select"
                    f" {len(selected_ids)}"
                )

        return selected_ids

    def plot_kernel(self, duration: int = 1000, remove: bool = True):
        """Generate an animation of the Kernel change while extending the reference.

        Raises
        ------
        ImportError: the imageio package is not shipped with mlsuite by default but is
                        required for generating the animation.
        """
        try:
            import imageio
        except ImportError as err:
            raise ImportError(
                "Package 'imageio' is required for generating a gif"
            ) from err

        import pathlib
        import shutil

        import matplotlib.pyplot as plt

        img_dir = pathlib.Path("img")

        img_dir.mkdir()
        for idx, results in enumerate(self.kernel_results):
            plt.plot(results)
            plt.title(f"Iteration {idx}")
            plt.ylabel("Kernel value")
            plt.savefig(img_dir / f"{str(idx).zfill(4)}.png")
            plt.close()

        with imageio.get_writer(
            "kernel_selection.gif", mode="I", duration=duration, loop=0
        ) as writer:
            for filename in sorted(img_dir.glob("*.png")):
                image = imageio.v2.imread(filename)
                writer.append_data(image)

        if remove:
            shutil.rmtree(img_dir)
