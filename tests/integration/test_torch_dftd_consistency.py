"""Tests for TorchDFTD3 calculator consistency between standard and neighbor list implementations."""

import numpy as np
import torch
import ipsuite as ips


def test_torch_dftd3_calculator_consistency(proj_path):
    """Test that TorchDFTD3Calculator and TorchDFTD3CalculatorNL yield consistent results.

    This test verifies that the standard torch-dftd3 calculator and the neighbor list
    version (with vesin) produce results within acceptable tolerances for both
    non-periodic and periodic systems.
    """
    # Create test systems using IPSuite workflow
    project = ips.Project()

    with project:
        # Generate molecular conformers
        water = ips.Smiles2Conformers(smiles="O", numConfs=3, seed=42)
        ethanol = ips.Smiles2Conformers(smiles="CCO", numConfs=2, seed=42)

        # Create a periodic molecular box for testing PBC
        box = ips.MultiPackmol(
            data=[water.frames, ethanol.frames],
            count=[4, 2],  # 4 water + 2 ethanol molecules
            density=800,   # kg/m³
            n_configurations=1,
            seed=42,
        )

    project.repro()

    # Prepare non-periodic test case
    water_mol = water.frames[0].copy()
    water_mol.pbc = False
    water_mol.cell = None

    # Define test cases: (name, atoms_object, is_periodic)
    test_cases = [
        ('water_non_periodic', water_mol, False),
        ('box_periodic', box.frames[0], True),
    ]

    # Common D3 parameters for both calculators
    d3_params = {
        'xc': 'pbe',
        'damping': 'bj',
        'cutoff': 15.0,
        'cnthr': 15.0,
        'abc': False,
        'dtype': torch.float64,
        'device': 'cpu',
    }

    # Test tolerance - adjust based on system size and neighbor list differences
    energy_tol = 2.5e-3  # kcal/mol - acceptable for neighbor list implementations
    forces_tol = 1e-3  # kcal/mol/Angstrom

    for case_name, atoms, is_periodic in test_cases:
        print(f"\nTesting case: {case_name}")
        print(f"  Atoms: {len(atoms)} atoms, PBC: {is_periodic}")

        # Prepare atoms copy for each calculator
        atoms1 = atoms.copy()
        atoms2 = atoms.copy()

        # Import calculators
        from ipsuite.models.torch_d3 import TorchDFTD3Calculator, TorchDFTD3CalculatorNL

        # Create both calculator types
        calc_standard = TorchDFTD3Calculator(**d3_params)
        calc_nl = TorchDFTD3CalculatorNL(**d3_params, skin=3.0)

        # Calculate with standard calculator
        atoms1.calc = calc_standard
        energy1 = atoms1.get_potential_energy()
        forces1 = atoms1.get_forces()

        # Calculate with neighbor list calculator
        atoms2.calc = calc_nl
        energy2 = atoms2.get_potential_energy()
        forces2 = atoms2.get_forces()

        # Compare results
        energy_diff = abs(energy1 - energy2)
        forces_diff = np.max(np.abs(forces1 - forces2))

        print(f"  Energy diff: {energy_diff:.2e} (tol: {energy_tol:.2e})")
        print(f"  Forces diff: {forces_diff:.2e} (tol: {forces_tol:.2e})")

        # Assertions
        assert energy_diff < energy_tol, (
            f"Energy mismatch in {case_name}: "
            f"standard={energy1:.8f}, NL={energy2:.8f}, diff={energy_diff:.2e}"
        )

        assert forces_diff < forces_tol, (
            f"Forces mismatch in {case_name}: max diff={forces_diff:.2e}"
        )

        # Verify PBC settings
        if is_periodic:
            assert atoms1.pbc.any(), f"Expected PBC but got {atoms1.pbc}"
            assert atoms1.cell.volume > 0, "Expected non-zero cell volume"
        else:
            assert not atoms1.pbc.any(), f"Expected no PBC but got {atoms1.pbc}"


def test_torch_dftd3_different_skin_values(proj_path):
    """Test that different skin values in TorchDFTD3CalculatorNL give consistent results.

    This test verifies that the neighbor list calculator produces similar results
    with different skin values, which affects when the neighbor list is updated.
    Small variations are expected due to the neighbor list caching behavior.
    """
    # Create a periodic test system
    project = ips.Project()

    with project:
        water = ips.Smiles2Conformers(smiles="O", numConfs=2, seed=42)
        box = ips.MultiPackmol(
            data=[water.frames],
            count=[6],              # 6 water molecules
            density=1000,           # kg/m³
            n_configurations=1,
            seed=42,
        )

    project.repro()

    atoms_template = box.frames[0]

    # D3 parameters for neighbor list calculator
    d3_params = {
        'xc': 'pbe',
        'damping': 'bj',
        'cutoff': 12.0,             # Smaller cutoff for faster computation
        'cnthr': 12.0,
        'abc': False,
        'dtype': torch.float64,
        'device': 'cpu',
    }

    # Test different skin values
    skin_values = [0.5, 1.0, 2.0]
    energies = []
    forces_list = []

    from ipsuite.models.torch_d3 import TorchDFTD3CalculatorNL

    # Calculate with each skin value
    for skin in skin_values:
        atoms = atoms_template.copy()
        calc = TorchDFTD3CalculatorNL(**d3_params, skin=skin)
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        energies.append(energy)
        forces_list.append(forces)

        print(f"Skin {skin}: Energy = {energy:.8f}")

    # Tolerances for skin value variations
    # Small differences expected due to neighbor list update frequency
    energy_tol = 2e-3  # kcal/mol
    forces_tol = 1e-3  # kcal/mol/Angstrom

    # Compare all skin values against the first one
    for i in range(1, len(skin_values)):
        energy_diff = abs(energies[0] - energies[i])
        forces_diff = np.max(np.abs(forces_list[0] - forces_list[i]))

        assert energy_diff < energy_tol, (
            f"Energy differs between skin={skin_values[0]} and "
            f"skin={skin_values[i]}: diff={energy_diff:.2e}"
        )

        assert forces_diff < forces_tol, (
            f"Forces differ between skin={skin_values[0]} and "
            f"skin={skin_values[i]}: max diff={forces_diff:.2e}"
        )
