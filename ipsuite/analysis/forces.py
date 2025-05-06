from ipsuite import base
import numpy as np
import zntrack
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class AnalyseStructureMeanForce(base.AnalyseAtoms):
    """Compute the mean force acting on a single structure.

    The mean force should be zero for periodic structures
    without any external forces acting on them.
    This node can be used to check converges in the DFT
    and find force trends in the system.
    """

    figure_path: Path = zntrack.outs_path(zntrack.nwd / "forces.png")
    forces: dict = zntrack.metrics()


    def run(self):
        self.figure_path.parent.mkdir(parents=True, exist_ok=True)
        force_magnitude_per_atoms = []
        for atoms in self.data:
            total_vector = np.sum(atoms.get_forces(), axis=0)
            magnitude = np.linalg.norm(total_vector)
            force_magnitude_per_atoms.append(magnitude)
        force_magnitude_per_atoms = np.array(force_magnitude_per_atoms)
        self.forces = {
            "mean": np.mean(force_magnitude_per_atoms),
            "std": np.std(force_magnitude_per_atoms),
            "min": np.min(force_magnitude_per_atoms),
            "max": np.max(force_magnitude_per_atoms),
        }

        samples = np.arange(len(force_magnitude_per_atoms))

        # Create a Pandas DataFrame
        df = pd.DataFrame({'Sample': samples, "Mean Force Norm  / eV/Å": force_magnitude_per_atoms})

        # Create the joint plot focusing the marginal on the 'Force Magnitude'
        joint_plot = sns.jointplot(
            data=df,
            x='Sample',
            y="Mean Force Norm  / eV/Å",
            marginal_kws={"bins": 30},
            kind='scatter',
        )

        # Overlay the line plot with the defined color
        joint_plot.ax_joint.plot(df['Sample'], df["Mean Force Norm  / eV/Å"])

        # Add a dashed line for the mean force
        mean_force = self.forces["mean"]
        joint_plot.ax_joint.axhline(mean_force, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_force:.2f} eV/Å')
        joint_plot.ax_joint.legend(loc='upper right')

        # Remove the marginal distribution on the x-axis (Samples)
        joint_plot.ax_marg_x.remove()

        plt.tight_layout()
        plt.savefig(self.figure_path)
