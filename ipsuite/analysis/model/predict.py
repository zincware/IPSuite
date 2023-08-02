import pathlib

import ase
import numpy as np
import pandas as pd
import tqdm
import zntrack
from ase.calculators.singlepoint import PropertyNotImplementedError

from ipsuite import base, models, utils
from ipsuite.analysis.model.math import decompose_stress_tensor, force_decomposition
from ipsuite.analysis.model.plots import get_figure, get_hist
from ipsuite.geometry import BarycenterMapping
from ipsuite.utils.ase_sim import freeze_copy_atoms


class Prediction(base.ProcessAtoms):
    """Create and Save the predictions from model on atoms.

    Attributes
    ----------
    model: The MLModel node that implements the 'predict' method
    atoms: list[Atoms] to predict properties for

    predictions: list[Atoms] the atoms that have the predicted properties from model
    """

    model: models.MLModel = zntrack.zn.deps()

    def run(self):
        self.atoms = []
        calc = self.model.get_calculator()

        for configuration in tqdm.tqdm(self.get_data(), ncols=70):
            configuration: ase.Atoms
            # Run calculation
            atoms = configuration.copy()
            atoms.calc = calc
            atoms.get_potential_energy()
            if "stress" in calc.implemented_properties:
                atoms.get_stress()

            self.atoms.append(freeze_copy_atoms(atoms))


class PredictionMetrics(base.AnalyseProcessAtoms):
    """Analyse the Models Prediction.

    This Node computes
    - MAE
    - RMSE
    - L4 Error
    - Maximum Error
    """

    energy_df_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "energy_df.csv")
    forces_df_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "forces_df.csv")
    stress_df_file: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "stress_df.csv")
    stress_hydrostatic_df_file: pathlib.Path = zntrack.dvc.outs(
        zntrack.nwd / "stress_hydrostatic_df.csv"
    )
    stress_deviatoric_df_file: pathlib.Path = zntrack.dvc.outs(
        zntrack.nwd / "stress_deviatoric_df.csv"
    )

    energy: dict = zntrack.zn.metrics()
    forces: dict = zntrack.zn.metrics()
    stress: dict = zntrack.zn.metrics()
    hydro_stress: dict = zntrack.zn.metrics()
    deviat_stress: dict = zntrack.zn.metrics()

    plots_dir: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "plots")

    energy_df: pd.DataFrame
    forces_df: pd.DataFrame
    stress_df: pd.DataFrame
    stress_hydro_df: pd.DataFrame
    stress_deviat_df: pd.DataFrame

    def _post_load_(self):
        """Load metrics - if available."""
        try:
            self.energy_df = pd.read_csv(self.energy_df_file)
        except FileNotFoundError:
            self.energy_df = pd.DataFrame({})
        try:
            self.forces_df = pd.read_csv(self.forces_df_file)
        except FileNotFoundError:
            self.forces_df = pd.DataFrame({})
        try:
            self.stress_df = pd.read_csv(self.stress_df_file)
            self.stress_hydro_df = pd.read_csv(self.stress_hydrostatic_df_file)
            self.stress_deviat_df = pd.read_csv(self.stress_deviatoric_df_file)
        except FileNotFoundError:
            self.stress_df = pd.DataFrame({})
            self.stress_hydro_df = pd.DataFrame({})
            self.stress_deviat_df = pd.DataFrame({})

    def get_dataframes(self):
        """Create a pandas dataframe from the given data."""
        true_data, pred_data = self.get_data()

        self.energy_df = pd.DataFrame(
            {
                "true": [x.get_potential_energy() for x in true_data],
                "prediction": [x.get_potential_energy() for x in pred_data],
            }
        )

        try:
            true_forces = np.reshape([x.get_forces() for x in true_data], (-1, 3))
            pred_forces = np.reshape([x.get_forces() for x in pred_data], (-1, 3))

            self.forces_df = pd.DataFrame(
                {
                    "true": np.linalg.norm(true_forces, axis=-1),
                    "true_x": true_forces[:, 0],
                    "true_y": true_forces[:, 1],
                    "true_z": true_forces[:, 2],
                    "prediction": np.linalg.norm(pred_forces, axis=-1),
                    "prediction_x": pred_forces[:, 0],
                    "prediction_y": pred_forces[:, 1],
                    "prediction_z": pred_forces[:, 2],
                }
            )
        except PropertyNotImplementedError:
            self.forces_df = pd.DataFrame({})

        try:
            true_stress = [x.get_stress(voigt=False) for x in true_data]
            pred_stress = [x.get_stress(voigt=False) for x in pred_data]
            hydro_true, deviat_true = decompose_stress_tensor(true_stress)
            hydro_pred, deviat_pred = decompose_stress_tensor(pred_stress)

            true_stress = np.reshape(true_stress, -1)
            pred_stress = np.reshape(pred_stress, -1)
            hydro_true = np.reshape(hydro_true, -1)
            deviat_true = np.reshape(deviat_true, -1)
            hydro_pred = np.reshape(hydro_pred, -1)
            deviat_pred = np.reshape(deviat_pred, -1)

            self.stress_df = pd.DataFrame(
                {
                    "true": true_stress,
                    "prediction": pred_stress,
                }
            )
            self.stress_hydro_df = pd.DataFrame(
                {
                    "true": hydro_true,
                    "prediction": hydro_pred,
                }
            )
            self.stress_deviat_df = pd.DataFrame(
                {
                    "true": deviat_true,
                    "prediction": deviat_pred,
                }
            )
        except PropertyNotImplementedError:
            self.stress_df = pd.DataFrame({})
            self.stress_hydro_df = pd.DataFrame({})
            self.stress_deviat_df = pd.DataFrame({})

    def get_metrics(self):
        """Update the metrics."""
        self.energy = utils.metrics.get_full_metrics(
            np.array(self.energy_df["true"]), np.array(self.energy_df["prediction"])
        )

        if not self.forces_df.empty:
            self.forces = utils.metrics.get_full_metrics(
                np.array(self.forces_df["true"]), np.array(self.forces_df["prediction"])
            )
        else:
            self.forces = {}

        if not self.stress_df.empty:
            self.stress = utils.metrics.get_full_metrics(
                np.array(self.stress_df["true"]), np.array(self.stress_df["prediction"])
            )
            self.hydro_stress = utils.metrics.get_full_metrics(
                np.array(self.stress_hydro_df["true"]),
                np.array(self.stress_hydro_df["prediction"]),
            )
            self.deviat_stress = utils.metrics.get_full_metrics(
                np.array(self.stress_deviat_df["true"]),
                np.array(self.stress_deviat_df["prediction"]),
            )
        else:
            self.stress = {}
            self.hydro_stress = {}
            self.deviat_stress = {}

    def get_plots(self, save=False):
        """Create figures for all available data."""
        self.plots_dir.mkdir(exist_ok=True)

        energy_plot = get_figure(
            self.energy_df["true"],
            self.energy_df["prediction"],
            datalabel=f"MAE: {self.energy['mae']:.4f} meV/atom",
            xlabel=r"$ab~initio$ energy $E$ / eV",
            ylabel=r"predicted energy $E$ / eV",
        )
        if save:
            energy_plot.savefig(self.plots_dir / "energy.png")

        if not self.forces_df.empty:
            forces_plot = get_figure(
                self.forces_df["true"],
                self.forces_df["prediction"],
                datalabel=rf"MAE: {self.forces['mae']:.4f} meV$ / (\AA \cdot $atom)",
                xlabel=(
                    r"$ab~initio$ magnitude of force per atom $|F|$ / eV$ \cdot \AA^{-1}$"
                ),
                ylabel=(
                    r"predicted magnitude of force per atom $|F|$ / eV$ \cdot \AA^{-1}$"
                ),
            )
            if save:
                forces_plot.savefig(self.plots_dir / "forces.png")

        if not self.stress_df.empty:
            stress_plot = get_figure(
                self.stress_df["true"],
                self.stress_df["prediction"],
                datalabel=rf"Max: {self.stress['max']:.4f}",
                xlabel=r"$ab~initio$ stress",
                ylabel=r"predicted stress",
            )

            hydrostatic_stress_plot = get_figure(
                self.stress_hydro_df["true"],
                self.stress_hydro_df["prediction"],
                datalabel=rf"Max: {self.hydro_stress['max']:.4f}",
                xlabel=r"$ab~initio$ hydrostatic stress",
                ylabel=r"predicted hydrostatic stress",
            )
            deviatoric_stress_plot = get_figure(
                self.stress_deviat_df["true"],
                self.stress_deviat_df["prediction"],
                datalabel=rf"Max: {self.deviat_stress['max']:.4f}",
                xlabel=r"$ab~initio$ deviatoric stress",
                ylabel=r"predicted deviatoric stress",
            )
            if save:
                stress_plot.savefig(self.plots_dir / "stress.png")
                hydrostatic_stress_plot.savefig(self.plots_dir / "hydrostatic_stress.png")
                deviatoric_stress_plot.savefig(self.plots_dir / "deviatoric_stress.png")

    def run(self):
        self.nwd.mkdir(exist_ok=True, parents=True)
        self.get_dataframes()
        self.get_metrics()
        self.get_plots(save=True)

        self.energy_df.to_csv(self.energy_df_file)
        self.forces_df.to_csv(self.forces_df_file)
        self.stress_df.to_csv(self.stress_df_file)
        self.stress_hydro_df.to_csv(self.stress_hydrostatic_df_file)
        self.stress_deviat_df.to_csv(self.stress_deviatoric_df_file)


class ForceAngles(base.AnalyseProcessAtoms):
    plot: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "angle.png")
    log_plot: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "angle_ylog.png")

    angles: dict = zntrack.zn.metrics()

    def run(self):
        true_data, pred_data = self.get_data()
        true_forces = np.reshape([x.get_forces() for x in true_data], (-1, 3))
        pred_forces = np.reshape([x.get_forces() for x in pred_data], (-1, 3))

        angles = utils.metrics.get_angles(true_forces, pred_forces)

        self.angles = {
            "rmse": utils.metrics.calculate_l_p_norm(np.zeros_like(angles), angles, p=2),
            "lp4": utils.metrics.calculate_l_p_norm(np.zeros_like(angles), angles, p=4),
            "max": utils.metrics.maximum_error(np.zeros_like(angles), angles),
            "mae": utils.metrics.calculate_l_p_norm(np.zeros_like(angles), angles, p=1),
        }

        fig, ax = get_hist(
            data=angles,
            label=rf"MAE: ${self.angles['mae']:.2f}^\circ$",
            xlabel=r"Angle between true and predicted forces $\theta / ^\circ$",
            ylabel="Probability / %",
        )
        fig.savefig(self.plot)
        ax.set_yscale("log")
        fig.savefig(self.log_plot)


class ForceDecomposition(base.AnalyseProcessAtoms):
    """Node for decomposing forces in a system of molecular units into
    translational, rotational and vibrational components.

    The implementation follows the method described in
    https://doi.org/10.26434/chemrxiv-2022-l4tb9


    Attributes
    ----------
    wasserstein_distance: float
        Compute the wasserstein distance between the distributions of the
        prediced and true forces for each trans, rot, vib component.
    """

    trans_forces: dict = zntrack.zn.metrics()
    rot_forces: dict = zntrack.zn.metrics()
    vib_forces: dict = zntrack.zn.metrics()
    wasserstein_distance = zntrack.zn.metrics()

    rot_force_plt = zntrack.dvc.outs(zntrack.nwd / "rot_force.png")
    trans_force_plt = zntrack.dvc.outs(zntrack.nwd / "trans_force.png")
    vib_force_plt = zntrack.dvc.outs(zntrack.nwd / "vib_force.png")

    histogram_plt = zntrack.dvc.outs(zntrack.nwd / "histogram.png")

    def get_plots(self):
        fig = get_figure(
            np.linalg.norm(self.true_forces["trans"], axis=-1),
            np.linalg.norm(self.pred_forces["trans"], axis=-1),
            datalabel="",
            xlabel=r"$ab~initio$ forces / eV$ \cdot \AA^{-1}$",
            ylabel=r"predicted forces / eV$ \cdot \AA^{-1}$",
        )
        fig.savefig(self.trans_force_plt)

        fig = get_figure(
            np.linalg.norm(self.true_forces["rot"], axis=-1),
            np.linalg.norm(self.pred_forces["rot"], axis=-1),
            datalabel="",
            xlabel=r"$ab~initio$ forces / eV$ \cdot \AA^{-1}$",
            ylabel=r"predicted forces / eV$ \cdot \AA^{-1}$",
        )
        fig.savefig(self.rot_force_plt)

        fig = get_figure(
            np.linalg.norm(self.true_forces["vib"], axis=-1),
            np.linalg.norm(self.pred_forces["vib"], axis=-1),
            datalabel="",
            xlabel=r"$ab~initio$ forces / eV$ \cdot \AA^{-1}$",
            ylabel=r"predicted forces / eV$ \cdot \AA^{-1}$",
        )
        fig.savefig(self.vib_force_plt)

    def get_metrics(self):
        """Update the metrics."""

        self.trans_forces = utils.metrics.get_full_metrics(
            np.array(self.true_forces["trans"]), np.array(self.pred_forces["trans"])
        )

        self.rot_forces = utils.metrics.get_full_metrics(
            np.array(self.true_forces["rot"]), np.array(self.pred_forces["rot"])
        )

        self.vib_forces = utils.metrics.get_full_metrics(
            np.array(self.true_forces["vib"]), np.array(self.pred_forces["vib"])
        )

    def get_histogram(self):
        import matplotlib.pyplot as plt
        from scipy.stats import wasserstein_distance

        def get_rel_scalar_prod(main, relative) -> np.ndarray:
            x = np.einsum("ij,ij->i", main, relative)
            x /= np.linalg.norm(main, axis=-1)
            return x

        fig, axes = plt.subplots(4, 3, figsize=(4 * 5, 3 * 3))
        fig.suptitle(
            (
                r"A fraction $\dfrac{\vec{a} \cdot"
                r" \vec{b}}{\left|\left|\vec{a}\right|\right|_{2}} $ of $\vec{b}$ that"
                r" contributes to $\vec{a}$"
            ),
            fontsize=16,
        )

        self.wasserstein_distance = {}

        for label, ax_ in zip(self.true_forces.keys(), axes):
            self.wasserstein_distance[label] = {}
            for part, ax in zip(["vib", "rot", "trans"], ax_):
                data = get_rel_scalar_prod(
                    self.true_forces[label], self.true_forces[part]
                )
                true_bins = ax.hist(
                    data, bins=50, density=True, label=f"true {label} {part}"
                )

                data = get_rel_scalar_prod(
                    self.pred_forces[label], self.pred_forces[part]
                )
                pred_bins = ax.hist(
                    data,
                    bins=true_bins[1],
                    density=True,
                    alpha=0.5,
                    label=f"pred {label} {part}",
                )
                ax.legend()
                self.wasserstein_distance[label][part] = wasserstein_distance(
                    true_bins[0], pred_bins[0]
                )

        fig.savefig(self.histogram_plt, bbox_inches="tight")

    def run(self):
        true_atoms, pred_atoms = self.get_data()
        mapping = BarycenterMapping(data=None)
        # TODO make the force_decomposition return full forces
        # TODO check if you sum the forces they yield the full forces
        # TODO make mapping a 'zn.nodes' with Mapping(species="BF4")
        #  maybe allow smiles and enumeration 0, 1, ...

        self.true_forces = {"all": [], "trans": [], "rot": [], "vib": []}
        self.pred_forces = {"all": [], "trans": [], "rot": [], "vib": []}

        for atom in tqdm.tqdm(true_atoms, ncols=70):
            atom_trans_forces, atom_rot_forces, atom_vib_forces = force_decomposition(
                atom, mapping
            )
            self.true_forces["all"].append(atom.get_forces())
            self.true_forces["trans"].append(atom_trans_forces)
            self.true_forces["rot"].append(atom_rot_forces)
            self.true_forces["vib"].append(atom_vib_forces)

        self.true_forces["all"] = np.concatenate(self.true_forces["all"])
        self.true_forces["trans"] = np.concatenate(self.true_forces["trans"])
        self.true_forces["rot"] = np.concatenate(self.true_forces["rot"])
        self.true_forces["vib"] = np.concatenate(self.true_forces["vib"])

        for atom in tqdm.tqdm(pred_atoms, ncols=70):
            atom_trans_forces, atom_rot_forces, atom_vib_forces = force_decomposition(
                atom, mapping
            )
            self.pred_forces["all"].append(atom.get_forces())
            self.pred_forces["trans"].append(atom_trans_forces)
            self.pred_forces["rot"].append(atom_rot_forces)
            self.pred_forces["vib"].append(atom_vib_forces)

        self.pred_forces["all"] = np.concatenate(self.pred_forces["all"])
        self.pred_forces["trans"] = np.concatenate(self.pred_forces["trans"])
        self.pred_forces["rot"] = np.concatenate(self.pred_forces["rot"])
        self.pred_forces["vib"] = np.concatenate(self.pred_forces["vib"])

        self.get_metrics()
        self.get_plots()
        self.get_histogram()
