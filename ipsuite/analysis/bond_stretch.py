import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import zntrack

from ipsuite.base import ProcessAtoms


class BondStretchAnalyses(ProcessAtoms):
    """Analyses a Model by evaluating the elongation of a bond
        in terms of energy, forces and optionally the uncertainty.

    Attributes
    ----------
    ase_calculator: ase.calculator
        ase calculator to use for simulation
    idxs: int

    R_min: float
        
    R_max: float
        
    n_steps: int
        
    data_id: int
        
    fig_size: (float, float)
        
    """

    ase_calculator = zntrack.zn.deps()

    idxs = zntrack.zn.params()
    R_min = zntrack.zn.params()
    R_max = zntrack.zn.params()
    n_steps = zntrack.zn.params()
    data_id: typing.Optional[int] = zntrack.zn.params(0)
    fig_size = zntrack.zn.params((10, 7))

    plots_dir: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "plots")

    def run(self):
        atoms_list = self.get_data()
        atoms = atoms_list[self.data_id].copy()

        train_bond_length = []
        for atoms in atoms_list:
            train_bond_length.append(atoms.get_distance(self.idxs[0], self.idxs[1]))

        atoms.calc = self.ase_calculator
        bond_lengths = np.linspace(self.R_min, self.R_max, self.n_steps)

        results = {}

        ens_energies = []
        ens_forces = []
        e_unvertainties = []
        f_uncertainties = []

        self.atoms = []
        for i in range(self.n_steps):
            atoms.set_distance(self.idxs[0], self.idxs[1], bond_lengths[i], fix=0)
            ens_energies.append(atoms.get_total_energy())
            ens_forces.append(atoms.get_forces())
            if "energy_uncertainty" in atoms.calc.results:
                e_unvertainties.append(atoms.calc.results.get("energy_uncertainty"))
            if "forces_uncertainty" in atoms.calc.results:
                f_uncertainties.append(atoms.calc.results.get("forces_uncertainty"))

            self.atoms.append(atoms.copy())

        results["energy"] = np.asarray(ens_energies)
        results["forces"] = np.asarray(ens_forces)
        if "energy_uncertainty" in atoms.calc.results:
            results["energy_uncertainty"] = np.asarray(e_unvertainties)
        if "forces_uncertainty" in atoms.calc.results:
            results["forces_uncertainty"] = np.asarray(f_uncertainties)

        chem_symbols = atoms.get_chemical_symbols()
        chem_symbols = [
            chem_symbols[self.idxs[0]],
            chem_symbols[self.idxs[1]],
        ]

        e_fig, f_fig = self.get_plots(
            train_bond_length,
            bond_lengths,
            results,
            chem_symbols,
            figsize=self.fig_size,
        )

        e_fig.savefig(self.plots_dir / f"energy_{chem_symbols[0]}_{chem_symbols[1]}.png")
        f_fig.savefig(self.plots_dir / f"force_{chem_symbols[0]}_{chem_symbols[1]}.png")

    def get_plots(
        self,
        train_bond_length,
        bond_lengths,
        results,
        chem_symbols,
        figsize,
    ):
        self.plots_dir.mkdir(exist_ok=True)

        # energy plot
        e_fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
        axs[0].plot(
            bond_lengths,
            results["energy"],
            "b",
            label="energy",
        )
        if "energy_uncertainty" in results:
            uncertanty = results["energy_uncertainty"]
            max_uncertainty = max(uncertanty)

            axs[0].fill_between(
                bond_lengths,
                results["energy"] - uncertanty,
                results["energy"] + uncertanty,
                color="black",
                alpha=0.2,
                label=f"energy uncertainty / max = {max_uncertainty:.2f} eV",
            )
        axs[0].set_xlabel(r"bond length $r_{i, j}$ / $\AA$")
        axs[0].set_ylabel(r"energy $E$ / eV")

        axs[1].hist(train_bond_length)
        axs[1].set_xlabel(r"bond length $r_{i, j}$ / $\AA$")
        axs[1].set_ylabel(r"frequency density")
        axs[0].set_title(f"{chem_symbols[0]}-{chem_symbols[1]} bond stretch")
        axs[0].legend()

        # plot force
        f_fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
        for i, idx in enumerate(self.idxs):
            ens_force = np.linalg.norm(results["forces"][:, idx], axis=1)
            axs[0].plot(bond_lengths, ens_force, label=f"atomic force {chem_symbols[i]}")

            if "forces_uncertainty" in results:
                uncertanity = results["forces_uncertainty"]
                uncertainty = np.linalg.norm(uncertanity[:, idx], axis=1)
                max_uncertainty = max(uncertainty.flatten())

                axs[0].fill_between(
                    bond_lengths,
                    ens_force - uncertainty,
                    ens_force + uncertainty,
                    color="black",
                    alpha=0.2,
                    label=(
                        f"max uncertainty {chem_symbols[i]}="
                        f" {max_uncertainty:.2f} meV/atom"
                    ),
                )
            axs[0].set_xlabel(r"bond length $r_{i, j}$ / $\AA$")
            axs[0].set_ylabel(r"magnitude of force per atom $|F|$ / eV$ \cdot \AA^{-1}$")

        axs[1].hist(train_bond_length)
        axs[1].set_xlabel(r"bond length $r_{i, j}$ / $\AA$")
        axs[1].set_ylabel(r"frequency density")
        axs[0].set_title(f"{chem_symbols[0]}-{chem_symbols[1]} bond stretch")
        axs[0].legend()

        return e_fig, f_fig
