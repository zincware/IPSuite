import pathlib
from typing import List, Optional

import ase
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import uncertainty_toolbox as uct
import zntrack
from ase.calculators.singlepoint import PropertyNotImplementedError

from ipsuite import base, models, utils
from ipsuite.analysis.model.math import (
    compute_uncertainty_metrics,
    decompose_stress_tensor,
    force_decomposition,
)
from ipsuite.analysis.model.plots import (  # get_cdf_figure,
    get_calibration_figure,
    get_figure,
    get_gaussianicity_figure,
    get_hist,
    slice_ensemble_uncertainty,
)
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
        self.frames = []
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

            self.frames.append(freeze_copy_atoms(atoms))


class PredictionMetrics(base.ComparePredictions):
    """Analyse the Models Prediction on standard metrics.

    Units are given in:
    - energy: meV/atom
    - forces: meV/Å
    - stress: eV/Å^3

    Attributes
    ----------
    ymax: dict of label key, and figure ylim values.
        Should be set when trying to compare different models.

    """

    # TODO ADD OPTIONAL YMAX PARAMETER

    figure_ymax: dict[str, float] = zntrack.params(default_factory=dict)

    data_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "data.npz")

    energy: dict = zntrack.metrics()
    forces: dict = zntrack.metrics()
    stress: dict = zntrack.metrics()
    stress_hydro: dict = zntrack.metrics()
    stress_deviat: dict = zntrack.metrics()

    plots_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plots")

    def __post_init__(self):
        self.content = {}

    def _post_load_(self):
        """Load metrics - if available."""
        try:
            with self.state.fs.open(self.data_file, "rb") as f:
                self.content = dict(np.load(f))
        except FileNotFoundError:
            self.content = {}

    def get_data(self):
        """Create dict of all data."""
        true_keys = self.x[0].calc.results.keys()
        pred_keys = self.y[0].calc.results.keys()

        energy_true = [x.get_potential_energy() / len(x) for x in self.x]
        energy_true = np.array(energy_true) * 1000
        self.content["energy_true"] = energy_true

        energy_prediction = [x.get_potential_energy() / len(x) for x in self.y]
        energy_prediction = np.array(energy_prediction) * 1000
        self.content["energy_pred"] = energy_prediction
        self.content["energy_error"] = energy_true - energy_prediction

        if "forces" in true_keys and "forces" in pred_keys:
            true_forces = [x.get_forces() for x in self.x]
            true_forces = np.concatenate(true_forces, axis=0) * 1000
            self.content["forces_true"] = np.reshape(true_forces, (-1,))

            pred_forces = [x.get_forces() for x in self.y]
            pred_forces = np.concatenate(pred_forces, axis=0) * 1000
            self.content["forces_pred"] = np.reshape(pred_forces, (-1,))
            self.content["forces_error"] = (
                self.content["forces_true"] - self.content["forces_pred"]
            )

        if "stress" in true_keys and "stress" in pred_keys:
            true_stress = np.array([x.get_stress(voigt=False) for x in self.x])
            pred_stress = np.array([x.get_stress(voigt=False) for x in self.y])
            hydro_true, deviat_true = decompose_stress_tensor(true_stress)
            hydro_pred, deviat_pred = decompose_stress_tensor(pred_stress)

            self.content["stress_true"] = np.reshape(true_stress, (-1,))
            self.content["stress_pred"] = np.reshape(pred_stress, (-1,))
            self.content["stress_error"] = (
                self.content["stress_true"] - self.content["stress_pred"]
            )
            self.content["stress_hydro_true"] = np.reshape(hydro_true, (-1,))
            self.content["stress_hydro_pred"] = np.reshape(hydro_pred, (-1,))
            self.content["stress_hydro_error"] = (
                self.content["stress_hydro_true"] - self.content["stress_hydro_pred"]
            )
            self.content["stress_deviat_true"] = np.reshape(deviat_true, (-1,))
            self.content["stress_deviat_pred"] = np.reshape(deviat_pred, (-1,))
            self.content["stress_deviat_error"] = (
                self.content["stress_deviat_true"] - self.content["stress_deviat_pred"]
            )

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
            self.stress_hydro = utils.metrics.get_full_metrics(
                self.content["stress_hydro_true"], self.content["stress_hydro_pred"]
            )
            self.stress_deviat = utils.metrics.get_full_metrics(
                self.content["stress_deviat_true"], self.content["stress_deviat_pred"]
            )
        else:
            self.stress = {}
            self.stress_hydro = {}
            self.stress_deviat = {}

    def get_plots(self, save=False):
        """Create figures for all available data."""
        self.plots_dir.mkdir(exist_ok=True)

        e_ymax = self.figure_ymax.get("energy", None)
        energy_plot = get_figure(
            self.content["energy_true"],
            self.content["energy_error"],
            datalabel=f"MAE: {self.energy['mae']:.2f} meV/atom",
            xlabel=r"$ab~initio$ energy $E$ / meV/atom",
            ylabel=r"$\Delta E$ / meV/atom",
            ymax=e_ymax,
        )
        if save:
            energy_plot.savefig(self.plots_dir / "energy.png")

        if "forces_true" in self.content:
            xlabel = (
                r"$ab~initio$ force components per atom $F_{alpha,i}$ / meV$ \cdot"
                r" \AA^{-1}$"
            )
            ylabel = r"$\Delta F_{alpha,i}$ / meV$ \cdot \AA^{-1}$"
            f_ymax = self.figure_ymax.get("forces", None)
            forces_plot = get_figure(
                self.content["forces_true"],
                self.content["forces_error"],
                datalabel=rf"MAE: {self.forces['mae']:.2f} meV$ / \AA$",
                xlabel=xlabel,
                ylabel=ylabel,
                ymax=f_ymax,
            )
            if save:
                forces_plot.savefig(self.plots_dir / "forces.png")

        if "stress_true" in self.content:
            s_true = self.content["stress_true"]
            s_error = self.content["stress_error"]
            shydro_true = self.content["stress_hydro_true"]
            shydro_error = self.content["stress_hydro_error"]
            sdeviat_true = self.content["stress_deviat_true"]
            sdeviat_error = self.content["stress_deviat_error"]

            s_ymax = self.figure_ymax.get("stress", None)
            hs_ymax = self.figure_ymax.get("stress_hydro", None)
            ds_ymax = self.figure_ymax.get("stress_deviat", None)

            stress_plot = get_figure(
                s_true,
                s_error,
                datalabel=rf"Max: {self.stress['max']:.4f}",
                xlabel=r"$ab~initio$ stress",
                ylabel=r"$\Delta$ stress",
                ymax=s_ymax,
            )
            hydrostatic_stress_plot = get_figure(
                shydro_true,
                shydro_error,
                datalabel=rf"Max: {self.stress_hydro['max']:.4f}",
                xlabel=r"$ab~initio$ hydrostatic stress",
                ylabel=r"$\Delta$ hydrostatic stress",
                ymax=hs_ymax,
            )
            deviatoric_stress_plot = get_figure(
                sdeviat_true,
                sdeviat_error,
                datalabel=rf"Max: {self.stress_deviat['max']:.4f}",
                xlabel=r"$ab~initio$ deviatoric stress",
                ylabel=r"$\Delta$ deviatoric stress",
                ymax=ds_ymax,
            )
            if save:
                stress_plot.savefig(self.plots_dir / "stress.png")
                hydrostatic_stress_plot.savefig(self.plots_dir / "hydrostatic_stress.png")
                deviatoric_stress_plot.savefig(self.plots_dir / "deviatoric_stress.png")

    def run(self):
        self.nwd.mkdir(exist_ok=True, parents=True)
        self.get_data()
        np.savez(self.data_file, **self.content)
        self.get_metrics()
        self.get_plots(save=True)

    def get_content(self):
        with self.state.fs.open(self.data_file, mode="rb") as f:
            content = dict(np.load(f))
            return content


class CalibrationMetrics(base.ComparePredictions):
    """Analyse the calibration of a models uncertainty estimate.
    Plots the empirical vs predicted error distribution,
    a log-log calibration plot and the miscalibration area.
    Further, various UQ metrics are computed:
    - Mean absolute calibration error
    - Root mean square miscalibration error
    - Miscalibration area
    - NLL
    - RLL

    For more information checkout the uncertainty toolbox or
    the following paper: 10.1088/2632-2153/ad594a

    Parameters
    ----------
    force_dist_slices: List[tuple]
        Interval in which to analyse the gassianity of error distributions.

    """

    force_dist_slices: Optional[List[tuple]] = zntrack.params(None)

    data_file: pathlib.Path = zntrack.outs_path(zntrack.nwd / "data.npz")
    energy: dict = zntrack.metrics()
    forces: dict = zntrack.metrics()

    plots_dir: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plots")

    def __post_init__(self):
        self.content = {}
        self.force_dist_slices = []

    def _post_load_(self):
        """Load metrics - if available."""
        try:
            with self.state.fs.open(self.data_file, "rb") as f:
                self.content = dict(np.load(f))
        except FileNotFoundError:
            self.content = {}

    def get_data(self):
        """Create dict of all data."""
        pred_keys = self.y[0].calc.results.keys()

        energy_true = [a.get_potential_energy() / len(a) for a in self.x]
        energy_true = np.array(energy_true) * 1000
        self.content["energy_true"] = energy_true
        energy_pred = [a.get_potential_energy() / len(a) for a in self.y]
        energy_pred = np.array(energy_pred) * 1000
        self.content["energy_pred"] = energy_pred

        energy_uncertainty = [
            a.calc.results["energy_uncertainty"] / len(a) for a in self.y
        ]
        energy_uncertainty = np.array(energy_uncertainty) * 1000
        self.content["energy_unc"] = energy_uncertainty

        if "forces" in pred_keys:
            true_forces = [a.get_forces() for a in self.x]
            true_forces = np.array(true_forces) * 1000
            pred_forces = [a.get_forces() for a in self.y]
            pred_forces = np.array(pred_forces) * 1000

            forces_uncertainty = [a.calc.results["forces_uncertainty"] for a in self.y]
            forces_uncertainty = np.array(forces_uncertainty) * 1000

            self.content["forces_true"] = np.reshape(true_forces, (-1,))
            self.content["forces_pred"] = np.reshape(pred_forces, (-1,))
            self.content["forces_unc"] = np.reshape(forces_uncertainty, (-1,))

            if "forces_ensemble" in self.y[0].calc.results.keys():
                n_ens = self.y[0].calc.results["forces_ensemble"].shape[2]
                forces_ensemble = [
                    np.reshape(a.calc.results["forces_ensemble"], (n_ens, -1))
                    for a in self.y
                ]
                forces_ensemble = np.array(forces_ensemble) * 1000
                forces_ensemble = np.transpose(forces_ensemble, (0, 2, 1))
                forces_ensemble = np.reshape(forces_ensemble, (-1, n_ens))

                self.content["forces_ensemble"] = forces_ensemble

    def get_metrics(self):
        """Update the metrics."""
        e_pred = self.content["energy_pred"]
        e_std = self.content["energy_unc"]
        e_true = self.content["energy_true"]
        metrics = compute_uncertainty_metrics(e_pred, e_std, e_true)
        self.energy = metrics

        if "forces_unc" in self.content:
            f_pred = self.content["forces_pred"]
            f_std = self.content["forces_unc"]
            f_true = self.content["forces_true"]
            metrics = compute_uncertainty_metrics(f_pred, f_std, f_true)
            self.forces = metrics

    def get_plots(self, save=False):
        """Create figures for all available data."""
        self.plots_dir.mkdir(exist_ok=True)
        e_err = np.abs(self.content["energy_pred"] - self.content["energy_true"])

        energy_plot = get_calibration_figure(
            e_err,
            self.content["energy_unc"],
            markersize=10,
            datalabel=rf"RLL={self.energy['rll']:.1f}",
            forces=False,
        )
        energy_gauss = get_gaussianicity_figure(
            e_err, self.content["energy_unc"], forces=False
        )

        energy_cdf_plot, e_cdf_ax = plt.subplots()
        e_cdf_ax = uct.plot_calibration(
            self.content["energy_pred"],
            self.content["energy_unc"],
            self.content["energy_true"],
            ax=e_cdf_ax,
        )

        if save:
            energy_plot.savefig(self.plots_dir / "energy.png")
            energy_gauss.savefig(self.plots_dir / "energy_gaussianicity.png")
            energy_cdf_plot.savefig(self.plots_dir / "energy_cdf.png")

        if "forces_unc" in self.content:
            f_err = np.abs(self.content["forces_pred"] - self.content["forces_true"])
            f_err = np.reshape(f_err, (-1,))

            forces_plot = get_calibration_figure(
                f_err,
                self.content["forces_unc"],
                datalabel=rf"RLL={self.forces['rll']:.1f}",
                forces=True,
            )
            forces_cdf_plot, f_cdf_ax = plt.subplots()
            f_cdf_ax = uct.plot_calibration(
                self.content["forces_pred"],
                self.content["forces_unc"],
                self.content["forces_true"],
                ax=f_cdf_ax,
            )

            gaussianicy_figures = []
            if "forces_ensemble" in self.content.keys():
                for start, end in self.force_dist_slices:
                    error_true, error_pred = slice_ensemble_uncertainty(
                        self.content["forces_true"],
                        self.content["forces_ensemble"],
                        start,
                        end,
                    )
                    fig = get_gaussianicity_figure(error_true, error_pred, forces=True)
                    gaussianicy_figures.append(fig)
            if save:
                forces_plot.savefig(self.plots_dir / "forces.png")
                forces_cdf_plot.savefig(self.plots_dir / "forces_cdf.png")

                for ii, fig in enumerate(gaussianicy_figures):
                    fig.savefig(self.plots_dir / f"forces_gaussianicity_{ii}.png")

    def run(self):
        self.nwd.mkdir(exist_ok=True, parents=True)
        self.get_data()
        np.savez(self.data_file, **self.content)
        self.get_metrics()
        self.get_plots(save=True)


class ForceAngles(base.ComparePredictions):
    plot: pathlib.Path = zntrack.outs_path(zntrack.nwd / "angle.png")
    log_plot: pathlib.Path = zntrack.outs_path(zntrack.nwd / "angle_ylog.png")

    angles: dict = zntrack.metrics()

    def run(self):
        true_forces = np.reshape([a.get_forces() for a in self.x], (-1, 3))
        pred_forces = np.reshape([a.get_forces() for a in self.y], (-1, 3))

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


class ForceDecomposition(base.ComparePredictions):
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

    trans_forces: dict = zntrack.metrics()
    rot_forces: dict = zntrack.metrics()
    vib_forces: dict = zntrack.metrics()
    wasserstein_distance: dict = zntrack.metrics()

    rot_force_plt: pathlib.Path = zntrack.outs_path(zntrack.nwd / "rot_force.png")
    trans_force_plt: pathlib.Path = zntrack.outs_path(zntrack.nwd / "trans_force.png")
    vib_force_plt: pathlib.Path = zntrack.outs_path(zntrack.nwd / "vib_force.png")

    histogram_plt: pathlib.Path = zntrack.outs_path(zntrack.nwd / "histogram.png")

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
        mapping = BarycenterMapping()
        # TODO make the force_decomposition return full forces
        # TODO check if you sum the forces they yield the full forces
        # TODO make mapping a 'zn.nodes' with Mapping(species="BF4")
        #  maybe allow smiles and enumeration 0, 1, ...

        self.true_forces = {"all": [], "trans": [], "rot": [], "vib": []}
        self.pred_forces = {"all": [], "trans": [], "rot": [], "vib": []}

        for atom in tqdm.tqdm(self.x, ncols=70):
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

        for atom in tqdm.tqdm(self.y, ncols=70):
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
