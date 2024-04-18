import pathlib

import ase
import numpy as np
import tqdm
import zntrack
from ase.calculators.singlepoint import PropertyNotImplementedError
from scipy import stats

from ipsuite import base, models, utils
from ipsuite.analysis.model.math import decompose_stress_tensor, force_decomposition
from ipsuite.analysis.model.plots import get_cdf_figure, get_figure, get_hist
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

    model: models.MLModel = zntrack.deps()

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
                try:
                    atoms.get_stress()
                except (
                    PropertyNotImplementedError,
                    ValueError,
                ):  # required for nequip, GAP
                    pass

            self.atoms.append(freeze_copy_atoms(atoms))


class PredictionMetrics(base.AnalyseProcessAtoms):
    """Analyse the Models Prediction on standard metrics.

    Units are given in:
    - energy: meV/atom
    - forces: meV/Å
    - stress: eV/Å^3
    """

    data_file = zntrack.outs_path(zntrack.nwd / "data.npz")

    energy: dict = zntrack.metrics()
    forces: dict = zntrack.metrics()
    stress: dict = zntrack.metrics()
    hydro_stress: dict = zntrack.metrics()
    deviat_stress: dict = zntrack.metrics()

    plots_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plots")

    def _post_init_(self):
        self.content = {}

    def _post_load_(self):
        """Load metrics - if available."""
        try:
            with self.state.fs.open(self.data_file, "rb") as f:
                self.content = dict(np.load(f))
        except FileNotFoundError:
            self.content = {}

    def get_dataframes(self):
        """Create a pandas dataframe from the given data."""
        true_data, pred_data = self.get_data()
        true_keys = true_data[0].calc.results.keys()
        pred_keys = pred_data[0].calc.results.keys()

        energy_true = [x.get_potential_energy() / len(x) for x in true_data]
        energy_true = np.array(energy_true) * 1000
        self.content["energy_true"] = energy_true

        energy_prediction = [x.get_potential_energy() / len(x) for x in pred_data]
        energy_prediction = np.array(energy_prediction) * 1000
        self.content["energy_pred"] = energy_prediction

        if "forces" in true_keys and "forces" in pred_keys:
            true_forces = [x.get_forces() for x in true_data]
            true_forces = np.concatenate(true_forces, axis=0) * 1000
            self.content["forces_true"] = np.reshape(true_forces, (-1,))

            pred_forces = [x.get_forces() for x in pred_data]
            pred_forces = np.concatenate(pred_forces, axis=0) * 1000
            self.content["forces_pred"] = np.reshape(pred_forces, (-1,))

        if "stress" in true_keys and "stress" in pred_keys:
            true_stress = np.array([x.get_stress(voigt=False) for x in true_data])
            pred_stress = np.array([x.get_stress(voigt=False) for x in pred_data])
            hydro_true, deviat_true = decompose_stress_tensor(true_stress)
            hydro_pred, deviat_pred = decompose_stress_tensor(pred_stress)

            self.content["stress_true"] = np.reshape(true_stress, (-1,))
            self.content["stress_pred"] = np.reshape(pred_stress, (-1,))
            self.content["stress_hydro_true"] = np.reshape(hydro_true, (-1,))
            self.content["stress_hydro_pred"] = np.reshape(hydro_pred, (-1,))
            self.content["stress_deviat_true"] = np.reshape(deviat_true, (-1,))
            self.content["stress_deviat_pred"] = np.reshape(deviat_pred, (-1,))

    def get_metrics(self):
        """Update the metrics."""
        self.energy = utils.metrics.get_full_metrics(
            self.content["energy_true"], self.content["energy_pred"]
        )

        if "forces_true" in self.content.keys():
            self.forces = utils.metrics.get_full_metrics(
                self.content["forces_true"], self.content["forces_pred"]
            )
        else:
            self.forces = {}

        if "stress_true" in self.content.keys():
            self.stress = utils.metrics.get_full_metrics(
                self.content["stress_true"], self.content["stress_pred"]
            )
            self.hydro_stress = utils.metrics.get_full_metrics(
                self.content["stress_hydro_true"], self.content["stress_hydro_pred"]
            )
            self.deviat_stress = utils.metrics.get_full_metrics(
                self.content["stress_deviat_true"], self.content["stress_deviat_pred"]
            )
        else:
            self.stress = {}
            self.hydro_stress = {}
            self.deviat_stress = {}

    def get_plots(self, save=False):
        """Create figures for all available data."""
        self.plots_dir.mkdir(exist_ok=True)

        energy_plot = get_figure(
            self.content["energy_true"],
            self.content["energy_pred"],
            datalabel=f"MAE: {self.energy['mae']:.2f} meV/atom",
            xlabel=r"$ab~initio$ energy $E$ / meV/atom",
            ylabel=r"predicted energy $E$ / meV/atom",
        )
        if save:
            energy_plot.savefig(self.plots_dir / "energy.png")

        if "forces_true" in self.content:
            xlabel = r"$ab~initio$ force components per atom $|F|$ / meV$ \cdot \AA^{-1}$"
            ylabel = r"predicted force components per atom $|F|$ / meV$ \cdot \AA^{-1}$"
            forces_plot = get_figure(
                self.content["forces_true"],
                self.content["forces_pred"],
                datalabel=rf"MAE: {self.forces['mae']:.2f} meV$ / \AA$",
                xlabel=xlabel,
                ylabel=ylabel,
            )
            if save:
                forces_plot.savefig(self.plots_dir / "forces.png")

        if "stress_true" in self.content:
            s_true = self.content["stress_true"]
            s_pred = self.content["stress_pred"]
            shydro_true = self.content["stress_hydro_true"]
            shydro_pred = self.content["stress_hydro_pred"]
            sdeviat_true = self.content["stress_deviat_true"]
            sdeviat_pred = self.content["stress_deviat_pred"]

            stress_plot = get_figure(
                s_true,
                s_pred,
                datalabel=rf"Max: {self.stress['max']:.4f}",
                xlabel=r"$ab~initio$ stress",
                ylabel=r"predicted stress",
            )
            hydrostatic_stress_plot = get_figure(
                shydro_true,
                shydro_pred,
                datalabel=rf"Max: {self.hydro_stress['max']:.4f}",
                xlabel=r"$ab~initio$ hydrostatic stress",
                ylabel=r"predicted hydrostatic stress",
            )
            deviatoric_stress_plot = get_figure(
                sdeviat_true,
                sdeviat_pred,
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
        np.savez(self.data_file, **self.content)
        self.get_metrics()
        self.get_plots(save=True)


class CalibrationMetrics(base.AnalyseProcessAtoms):
    """Analyse the Models Prediction on standard metrics.

    Units are given in:
    - energy: meV/atom
    - forces: meV/Å
    - stress: eV/Å^3
    """

    data_file = zntrack.outs_path(zntrack.nwd / "data.npz")
    energy: dict = zntrack.metrics()
    forces: dict = zntrack.metrics()

    plots_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plots")

    def _post_init_(self):
        self.content = {}

    def _post_load_(self):
        """Load metrics - if available."""
        try:
            with self.state.fs.open(self.data_file, "rb") as f:
                self.content = dict(np.load(f))
        except FileNotFoundError:
            self.content = {}

    def get_dataframes(self):
        """Create a pandas dataframe from the given data."""
        true_data, pred_data = self.get_data()
        true_keys = true_data[0].calc.results.keys()
        pred_keys = pred_data[0].calc.results.keys()

        energy_true = [x.get_potential_energy() / len(x) for x in true_data]
        energy_true = np.array(energy_true) * 1000
        energy_pred = [x.get_potential_energy() / len(x) for x in pred_data]
        energy_pred = np.array(energy_pred) * 1000
        self.content["energy_err"] = np.abs(energy_true - energy_pred)

        energy_uncertainty = [
            x.calc.results["energy_uncertainty"] / len(x) for x in pred_data
        ]
        energy_uncertainty = np.array(energy_uncertainty) * 1000
        self.content["energy_unc"] = energy_uncertainty

        if "forces" in true_keys and "forces_uncertainty" in pred_keys:
            true_forces = [x.get_forces() for x in true_data]
            true_forces = np.concatenate(true_forces, axis=0) * 1000
            pred_forces = [x.get_forces() for x in pred_data]
            pred_forces = np.concatenate(pred_forces, axis=0) * 1000
            forces_uncertainty = [x.calc.results["forces_uncertainty"] for x in pred_data]
            forces_uncertainty = np.concatenate(forces_uncertainty, axis=0) * 1000

            self.content["forces_err"] = np.abs(true_forces - pred_forces)
            self.content["forces_unc"] = forces_uncertainty

    def get_metrics(self):
        """Update the metrics."""
        e_err = self.content["energy_err"]
        e_unc = self.content["energy_unc"]
        pearsonr = stats.pearsonr(e_err, e_unc)[0]
        self.energy = {"pearsonr": pearsonr}

        if "forces_err" in self.content.keys():
            f_err = np.reshape(self.content["forces_err"], (-1,))
            f_unc = np.reshape(self.content["forces_unc"], (-1,))
            self.forces = {"pearsonr": stats.pearsonr(f_err, f_unc)[0]}
        else:
            self.forces = {}

    def get_plots(self, save=False):
        """Create figures for all available data."""
        self.plots_dir.mkdir(exist_ok=True)

        energy_plot = get_figure(
            self.content["energy_unc"],
            self.content["energy_err"],
            datalabel=rf"Pearson: {self.energy['pearsonr']:.4f}",
            xlabel=r"energy uncertainty $\sigma$ / meV/atom",
            ylabel=r"energy error $\Delta E$ / meV/atom",
        )
        energy_cdf_plot = get_cdf_figure(
            self.content["energy_err"],
            self.content["energy_unc"],
        )
        if save:
            energy_plot.savefig(self.plots_dir / "energy.png")
            energy_cdf_plot.savefig(self.plots_dir / "energy_cdf.png")

        if "forces_err" in self.content:
            xlabel = r"force uncertainty per atom $\sigma$ / meV$ \cdot \AA^{-1}$"
            ylabel = r"force components error per atom $\Delta F$ / meV$ \cdot \AA^{-1}$"
            f_err = np.reshape(self.content["forces_err"], (-1,))
            f_unc = np.reshape(self.content["forces_unc"], (-1,))
            forces_plot = get_figure(
                f_unc,
                f_err,
                datalabel=rf"Pearson: {self.forces['pearsonr']:.4f}",
                xlabel=xlabel,
                ylabel=ylabel,
            )
            forces_cdf_plot = get_cdf_figure(
                f_err,
                f_unc,
            )
            if save:
                forces_plot.savefig(self.plots_dir / "forces.png")
                forces_cdf_plot.savefig(self.plots_dir / "forces_cdf.png")

    def run(self):
        self.nwd.mkdir(exist_ok=True, parents=True)
        self.get_dataframes()
        np.savez(self.data_file, **self.content)
        self.get_metrics()
        self.get_plots(save=True)


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
            np.reshape(self.true_forces["trans"], -1),
            np.reshape(self.pred_forces["trans"], -1),
            datalabel=rf"Trans. MAE: {self.trans_forces['mae']:.2f} meV$ / \AA$",
            xlabel=r"$ab~initio$ forces / meV$ \cdot \AA^{-1}$",
            ylabel=r"predicted forces / meV$ \cdot \AA^{-1}$",
        )
        fig.savefig(self.trans_force_plt)

        fig = get_figure(
            np.reshape(self.true_forces["rot"], -1),
            np.reshape(self.pred_forces["rot"], -1),
            datalabel=rf"Rot. MAE: {self.rot_forces['mae']:.2f} meV$ / \AA$",
            xlabel=r"$ab~initio$ forces / meV$ \cdot \AA^{-1}$",
            ylabel=r"predicted forces / meV$ \cdot \AA^{-1}$",
        )
        fig.savefig(self.rot_force_plt)

        fig = get_figure(
            np.reshape(self.true_forces["vib"], -1),
            np.reshape(self.pred_forces["vib"], -1),
            datalabel=rf"Vib. MAE: {self.vib_forces['mae']:.2f} meV$ / \AA$",
            xlabel=r"$ab~initio$ forces / meV$ \cdot \AA^{-1}$",
            ylabel=r"predicted forces / meV$ \cdot \AA^{-1}$",
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

        self.true_forces["all"] = np.concatenate(self.true_forces["all"]) * 1000
        self.true_forces["trans"] = np.concatenate(self.true_forces["trans"]) * 1000
        self.true_forces["rot"] = np.concatenate(self.true_forces["rot"]) * 1000
        self.true_forces["vib"] = np.concatenate(self.true_forces["vib"]) * 1000

        for atom in tqdm.tqdm(pred_atoms, ncols=70):
            atom_trans_forces, atom_rot_forces, atom_vib_forces = force_decomposition(
                atom, mapping
            )
            self.pred_forces["all"].append(atom.get_forces())
            self.pred_forces["trans"].append(atom_trans_forces)
            self.pred_forces["rot"].append(atom_rot_forces)
            self.pred_forces["vib"].append(atom_vib_forces)

        self.pred_forces["all"] = np.concatenate(self.pred_forces["all"]) * 1000
        self.pred_forces["trans"] = np.concatenate(self.pred_forces["trans"]) * 1000
        self.pred_forces["rot"] = np.concatenate(self.pred_forces["rot"]) * 1000
        self.pred_forces["vib"] = np.concatenate(self.pred_forces["vib"]) * 1000

        self.get_metrics()
        self.get_plots()
        self.get_histogram()
