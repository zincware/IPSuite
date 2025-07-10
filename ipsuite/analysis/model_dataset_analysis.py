import json
import os
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import zntrack
from apax.nodes import Apax

from ipsuite.utils.metrics import get_full_metrics


# TODO: uncertainty for metrics?
class ModelEvaluationAnalysis(zntrack.Node):
    data: list[ase.Atoms] = zntrack.deps()
    models: list[Apax] = zntrack.deps()

    figure: Path = zntrack.outs_path(zntrack.nwd / "figure.png")
    metrics: list = zntrack.metrics()

    def run(self):
        self.metrics = []

        energies = np.array([x.get_potential_energy() for x in self.data])
        forces = np.array([x.get_forces() for x in self.data])

        for model in self.models:
            calc = model.get_calculator()
            if hasattr(calc, "batch_eval"):
                frames = calc.batch_eval(self.data, 1)
            else:
                frames = []
                for atoms in self.data:
                    atoms.calc = calc
                    atoms.get_potential_energy()
                    frames.append(atoms)
            self.metrics.append(
                {
                    "energy": get_full_metrics(
                        energies, np.array([x.get_potential_energy() for x in frames])
                    ),
                    "forces": get_full_metrics(
                        forces, np.array([x.get_forces() for x in frames])
                    ),
                }
            )

        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle("Model evolution evaluation")
        ax[0].plot([x["energy"]["mae"] for x in self.metrics], marker="o")
        ax[0].set_ylabel("MAE Energy")
        ax[1].plot([x["forces"]["mae"] for x in self.metrics], marker="o")
        ax[1].set_xlabel("Model iteration")
        ax[1].set_ylabel("MAE Forces")
        fig.savefig(self.figure, bbox_inches="tight")


class ClusterdDataEvaluation(zntrack.Node):
    data: list[list[ase.Atoms]] = zntrack.deps()
    labels: list[str] = zntrack.params()
    model: Apax = zntrack.deps()

    outs_path: Path = zntrack.outs_path(zntrack.nwd / "outs")

    def run(self):
        os.makedirs(self.outs_path, exist_ok=True)

        calc = self.model.get_calculator()

        all_metrics = []
        all_energies_true = []
        all_energies_pred = []
        all_forces_true = []
        all_forces_pred = []

        for i, (data_cluster, label) in enumerate(zip(self.data, self.labels)):
            # Get true values
            energies_true = np.array([x.get_potential_energy() for x in data_cluster])
            forces_true = np.array([x.get_forces() for x in data_cluster])

            # Run predictions
            if hasattr(calc, "batch_eval"):
                frames = calc.batch_eval(data_cluster, 1)
            else:
                frames = []
                for atoms in data_cluster:
                    atoms.calc = calc
                    atoms.get_potential_energy()
                    frames.append(atoms)

            # Get predicted values
            energies_pred = np.array([x.get_potential_energy() for x in frames])
            forces_pred = np.array([x.get_forces() for x in frames])

            # Calculate metrics
            energy_metrics = get_full_metrics(energies_true, energies_pred)
            forces_metrics = get_full_metrics(forces_true, forces_pred)

            cluster_metrics = {
                "label": label,
                "energy": energy_metrics,
                "forces": forces_metrics,
                "n_structures": len(data_cluster),
            }
            all_metrics.append(cluster_metrics)

            # Store for plotting
            all_energies_true.extend(energies_true)
            all_energies_pred.extend(energies_pred)
            all_forces_true.extend(forces_true.flatten())
            all_forces_pred.extend(forces_pred.flatten())

        # Save all metrics
        with open(self.outs_path / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Model Performance by Data Cluster", fontsize=16)

        # Create color map
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.labels)))

        # Energy relative error plot
        ax1 = axes[0, 0]
        start_idx = 0
        for i, (data_cluster, label) in enumerate(zip(self.data, self.labels)):
            end_idx = start_idx + len(data_cluster)
            true_energies = np.array(all_energies_true[start_idx:end_idx])
            pred_energies = np.array(all_energies_pred[start_idx:end_idx])

            # Calculate per-atom energies
            true_energies_per_atom = true_energies / np.array(
                [len(atoms) for atoms in data_cluster]
            )
            pred_energies_per_atom = pred_energies / np.array(
                [len(atoms) for atoms in data_cluster]
            )

            # Calculate difference: predicted - true
            diff = pred_energies_per_atom - true_energies_per_atom

            ax1.scatter(
                true_energies_per_atom, diff, color=colors[i], label=label, alpha=0.6
            )
            start_idx = end_idx

        ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax1.set_xlabel("True Energy per Atom / eV")
        ax1.set_ylabel("Difference: Pred - True / eV")
        ax1.set_title("Energy Difference Plot")
        ax1.legend()

        # Forces relative error plot
        ax2 = axes[0, 1]
        start_idx = 0
        for i, (data_cluster, label) in enumerate(zip(self.data, self.labels)):
            n_forces = len(data_cluster) * 3 * len(data_cluster[0])
            end_idx = start_idx + n_forces
            true_forces = np.array(all_forces_true[start_idx:end_idx])
            pred_forces = np.array(all_forces_pred[start_idx:end_idx])

            # Calculate difference: predicted - true
            diff = pred_forces - true_forces

            ax2.scatter(true_forces, diff, color=colors[i], label=label, alpha=0.6, s=1)
            start_idx = end_idx

        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax2.set_xlabel("True Forces / eV/Å")
        ax2.set_ylabel("Difference: Pred - True / eV/Å")
        ax2.set_title("Forces Difference Plot")

        # Energy MAE by cluster
        ax3 = axes[1, 0]
        energy_maes = [m["energy"]["mae"] for m in all_metrics]
        ax3.bar(range(len(self.labels)), energy_maes, color=colors)
        ax3.set_xlabel("Data Cluster")
        ax3.set_ylabel("Energy MAE / eV")
        ax3.set_title("Energy MAE by Cluster")
        ax3.set_xticks(range(len(self.labels)))
        ax3.set_xticklabels(self.labels, rotation=45)

        # Forces MAE by cluster
        ax4 = axes[1, 1]
        forces_maes = [m["forces"]["mae"] for m in all_metrics]
        ax4.bar(range(len(self.labels)), forces_maes, color=colors)
        ax4.set_xlabel("Data Cluster")
        ax4.set_ylabel("Forces MAE / eV/Å")
        ax4.set_title("Forces MAE by Cluster")
        ax4.set_xticks(range(len(self.labels)))
        ax4.set_xticklabels(self.labels, rotation=45)

        plt.tight_layout()
        plt.savefig(self.outs_path / "evaluation_plots.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Create jointplots for error distribution by label

        # Combined energy error distribution with all labels
        all_true_energies_per_atom = []
        all_energy_errors_per_atom = []
        all_energy_labels = []

        start_idx = 0
        for i, (data_cluster, label) in enumerate(zip(self.data, self.labels)):
            end_idx = start_idx + len(data_cluster)

            true_energies = np.array(all_energies_true[start_idx:end_idx])
            pred_energies = np.array(all_energies_pred[start_idx:end_idx])

            true_energies_per_atom = true_energies / np.array(
                [len(atoms) for atoms in data_cluster]
            )
            pred_energies_per_atom = pred_energies / np.array(
                [len(atoms) for atoms in data_cluster]
            )
            energy_errors_per_atom = pred_energies_per_atom - true_energies_per_atom

            all_true_energies_per_atom.extend(true_energies_per_atom)
            all_energy_errors_per_atom.extend(energy_errors_per_atom)
            all_energy_labels.extend([label] * len(true_energies_per_atom))

            start_idx = end_idx

        # Create DataFrame for combined energy plot
        energy_df = pd.DataFrame(
            {
                "true_energy": all_true_energies_per_atom,
                "energy_error": all_energy_errors_per_atom,
                "label": all_energy_labels,
            }
        )

        # Create combined energy jointplot with hue
        g = sns.jointplot(
            data=energy_df, x="true_energy", y="energy_error", hue="label", height=7
        )

        g.ax_joint.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        g.set_axis_labels("True Energy per Atom / eV", "Energy Error: Pred - True / eV")
        g.figure.suptitle("Combined Energy Error Distribution - All Labels", y=1.02)

        plt.tight_layout()
        plt.savefig(
            self.outs_path / "energy_error_distribution_combined.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Combined forces error distribution with all labels
        all_true_forces = []
        all_forces_errors = []
        all_forces_labels = []

        start_idx = 0
        for i, (data_cluster, label) in enumerate(zip(self.data, self.labels)):
            n_forces = len(data_cluster) * 3 * len(data_cluster[0])
            end_idx = start_idx + n_forces

            true_forces = np.array(all_forces_true[start_idx:end_idx])
            pred_forces = np.array(all_forces_pred[start_idx:end_idx])
            forces_errors = pred_forces - true_forces

            # Sample for better visualization
            n_sample = min(5000, len(true_forces))  # Smaller sample for combined plot
            if len(true_forces) > n_sample:
                idx = np.random.choice(len(true_forces), n_sample, replace=False)
                true_forces_sample = true_forces[idx]
                forces_errors_sample = forces_errors[idx]
            else:
                true_forces_sample = true_forces
                forces_errors_sample = forces_errors

            all_true_forces.extend(true_forces_sample)
            all_forces_errors.extend(forces_errors_sample)
            all_forces_labels.extend([label] * len(true_forces_sample))

            start_idx = end_idx

        # Create DataFrame for combined forces plot
        forces_df = pd.DataFrame(
            {
                "true_forces": all_true_forces,
                "forces_error": all_forces_errors,
                "label": all_forces_labels,
            }
        )

        # Create combined forces jointplot with hue
        g = sns.jointplot(
            data=forces_df, x="true_forces", y="forces_error", hue="label", height=7
        )

        g.ax_joint.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        g.set_axis_labels("True Forces / eV/Å", "Force Error: Pred - True / eV/Å")
        g.figure.suptitle("Combined Forces Error Distribution - All Labels", y=1.02)

        plt.tight_layout()
        plt.savefig(
            self.outs_path / "forces_error_distribution_combined.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save summary metrics
        summary_metrics = {
            "overall_energy_mae": np.mean(energy_maes),
            "overall_forces_mae": np.mean(forces_maes),
            "cluster_metrics": all_metrics,
        }

        with open(self.outs_path / "summary_metrics.json", "w") as f:
            json.dump(summary_metrics, f, indent=2)
