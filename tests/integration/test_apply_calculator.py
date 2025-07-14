
import dataclasses

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

import ipsuite as ips
from ipsuite.abc import NodeWithCalculator


class DummyCalculator(Calculator):
    """A dummy calculator that returns predefined values for testing."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, energy=1.0, forces_value=0.1, **kwargs):
        super().__init__(**kwargs)
        self.energy = energy
        self.forces_value = forces_value

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        if "energy" in properties:
            self.results["energy"] = self.energy

        if "forces" in properties:
            self.results["forces"] = np.full((len(atoms), 3), self.forces_value)


@dataclasses.dataclass
class DummyModel(NodeWithCalculator):
    """A dummy model for testing ApplyCalculator."""

    energy: float = 1.0
    forces_value: float = 0.1

    def get_calculator(self, **kwargs):
        return DummyCalculator(energy=self.energy, forces_value=self.forces_value)


def test_apply_calculator(proj_path):
    project = ips.Project()

    model = ips.MACEMPModel()

    with project:
        water = ips.Smiles2Conformers(
            smiles="O",
            numConfs=10,
        )
        traj = ips.ApplyCalculator(
            data=water.frames,
            model=model,
        )

    project.repro()

    assert len(traj.frames) == 10


def test_apply_calculator_replace_mode(proj_path):
    """Test ApplyCalculator in replace mode (default behavior)."""
    project = ips.Project()

    with project:
        # Create some dummy atoms using Smiles2Atoms
        water = ips.Smiles2Atoms(smiles="O")

        # Apply first calculator
        model1 = DummyModel(energy=5.0, forces_value=0.5)
        traj1 = ips.ApplyCalculator(
            data=water.frames,
            model=model1,
        )

        # Apply second calculator in replace mode (default)
        model2 = DummyModel(energy=3.0, forces_value=0.3)
        traj2 = ips.ApplyCalculator(
            data=traj1.frames,
            model=model2,
            additive=False,  # explicit replace mode
        )

    project.repro()

    # Check that the second calculator replaced the first
    final_atoms = traj2.frames[0]
    assert final_atoms.get_potential_energy() == 3.0  # Only model2's energy
    assert np.allclose(final_atoms.get_forces(), 0.3)  # Only model2's forces


def test_apply_calculator_additive_mode(proj_path):
    """Test ApplyCalculator in additive mode."""
    project = ips.Project()

    with project:
        # Create some dummy atoms using Smiles2Atoms
        water = ips.Smiles2Atoms(smiles="O")

        # Apply first calculator
        model1 = DummyModel(energy=5.0, forces_value=0.5)
        traj1 = ips.ApplyCalculator(
            data=water.frames,
            model=model1,
        )

        # Apply second calculator in additive mode (e.g., D3 correction)
        model2 = DummyModel(energy=2.0, forces_value=0.2)
        traj2 = ips.ApplyCalculator(
            data=traj1.frames,
            model=model2,
            additive=True,  # additive mode
        )

    project.repro()

    # Check that the results were added together
    final_atoms = traj2.frames[0]
    assert final_atoms.get_potential_energy() == 7.0  # 5.0 + 2.0
    assert np.allclose(final_atoms.get_forces(), 0.7)  # 0.5 + 0.2
