import ipsuite as ips
import pytest
import numpy as np


def test_BuildSurface_basic(proj_path):
    """Test basic BuildSurface functionality with default parameters."""
    with ips.Project() as proj:
        surface = ips.BuildSurface(
            lattice="Au",
            indices=(1, 1, 1),
            layers=5
        )

    proj.repro()

    assert len(surface.frames) == 1
    atoms = surface.frames[0]

    # Check that we have a gold surface
    assert all(symbol == "Au" for symbol in atoms.get_chemical_symbols())

    # Check that vacuum was added (cell should be larger in z-direction)
    cell = atoms.get_cell()
    assert cell[2, 2] > 10.0  # Should have vacuum space


def test_BuildSurface_custom_parameters(proj_path):
    """Test BuildSurface with custom lattice constant and crystal structure."""
    with ips.Project() as proj:
        surface = ips.BuildSurface(
            lattice="Cu",
            indices=(1, 0, 0),
            layers=3,
            vacuum=15.0,
            lattice_constant=3.6,
            crystal_structure="fcc"
        )

    proj.repro()

    assert len(surface.frames) == 1
    atoms = surface.frames[0]

    # Check that we have a copper surface
    assert all(symbol == "Cu" for symbol in atoms.get_chemical_symbols())

    # Check that custom vacuum was applied
    cell = atoms.get_cell()
    assert cell[2, 2] > 15.0  # Should have at least 15 Å vacuum


def test_BuildSurface_different_indices(proj_path):
    """Test BuildSurface with different Miller indices."""
    surfaces = []

    with ips.Project() as proj:
        # Test different surface orientations
        surface_100 = ips.BuildSurface(lattice="Pt", indices=(1, 0, 0), layers=4)
        surface_110 = ips.BuildSurface(lattice="Pt", indices=(1, 1, 0), layers=4)
        surface_111 = ips.BuildSurface(lattice="Pt", indices=(1, 1, 1), layers=4)
        surfaces = [surface_100, surface_110, surface_111]

    proj.repro()

    # All should generate valid surfaces
    for surface in surfaces:
        assert len(surface.frames) == 1
        atoms = surface.frames[0]
        assert all(symbol == "Pt" for symbol in atoms.get_chemical_symbols())
        assert len(atoms) > 0


def test_BuildSurface_with_mace_relaxation(proj_path):
    """Test BuildSurface followed by MACE-MP relaxation."""
    with ips.Project() as proj:
        # Build surface
        surface = ips.BuildSurface(
            lattice="Cu",
            indices=(1, 1, 1),
            layers=3,
            vacuum=12.0
        )

        # Set up MACE-MP model
        model = ips.MACEMPModel()

        # Relax the surface using ASEGeoOpt
        relaxed_surface = ips.ASEGeoOpt(
            data=surface.frames,
            model=model,
            optimizer="FIRE",
            run_kwargs={"fmax": 0.1, "steps": 50},
            sampling_rate=10,
            maxstep=50
        )

    proj.repro()

    # Verify the surface was built correctly
    initial_atoms = surface.frames[0]
    assert all(symbol == "Cu" for symbol in initial_atoms.get_chemical_symbols())
    assert len(initial_atoms) > 0

    # Verify relaxation trajectory was generated
    trajectory = relaxed_surface.frames
    assert len(trajectory) > 0

    # Check that final structure still has correct composition
    final_atoms = trajectory[-1]
    assert all(symbol == "Cu" for symbol in final_atoms.get_chemical_symbols())
    assert len(final_atoms) == len(initial_atoms)


def test_AddAdsorbate_basic(proj_path):
    """Test basic AddAdsorbate functionality with single adsorbate."""
    with ips.Project() as proj:
        # Create surface
        surface = ips.BuildSurface(
            lattice="Pt",
            indices=(1, 1, 1),
            layers=4,
            vacuum=15.0
        )

        # Create adsorbate molecules
        water = ips.Smiles2Atoms(smiles="O")  # H2O

        # Add adsorbate to surface (nested structure)
        adsorbed_system = ips.AddAdsorbate(
            slab=surface.frames,
            data=[water.frames],  # Wrap in list for nested structure
            height=[2.0],
            excluded_radius=[3.0]
        )

    proj.repro()

    # Verify surface was created
    surface_atoms = surface.frames[0]
    assert all(symbol == "Pt" for symbol in surface_atoms.get_chemical_symbols())

    # Verify adsorbate was added
    final_system = adsorbed_system.frames[0]
    symbols = final_system.get_chemical_symbols()

    # Should have Pt atoms + water atoms
    pt_count = symbols.count("Pt")
    o_count = symbols.count("O")
    h_count = symbols.count("H")

    assert pt_count > 0  # Original surface atoms
    assert o_count == 1  # One oxygen from water
    assert h_count == 2  # Two hydrogens from water

    # Total atoms should be surface + water
    assert len(final_system) == len(surface_atoms) + 3  # Pt atoms + H2O


def test_AddAdsorbate_multiple_adsorbates(proj_path):
    """Test AddAdsorbate with multiple adsorbates and collision detection."""
    with ips.Project() as proj:
        # Create surface
        surface = ips.BuildSurface(
            lattice="Cu",
            indices=(1, 0, 0),
            layers=3,
            vacuum=12.0
        )

        # Create different adsorbate molecules
        co = ips.Smiles2Atoms(smiles="[C-]#[O+]")  # CO
        h2 = ips.Smiles2Atoms(smiles="[H][H]")     # H2

        # Add multiple adsorbates with different heights and exclusion radii
        adsorbed_system = ips.AddAdsorbate(
            slab=surface.frames,
            data=[co.frames, h2.frames],  # Nested structure
            height=[1.5, 3.0],
            excluded_radius=[2.5, 2.0]
        )

    proj.repro()

    # Verify multiple adsorbates were added
    final_system = adsorbed_system.frames[0]
    symbols = final_system.get_chemical_symbols()

    cu_count = symbols.count("Cu")
    c_count = symbols.count("C")
    o_count = symbols.count("O")
    h_count = symbols.count("H")

    assert cu_count > 0   # Original surface
    assert c_count == 1   # One carbon from CO
    assert o_count == 1   # One oxygen from CO
    assert h_count == 2   # Two hydrogens from H2

    # Verify positioning: CO should be closer to surface than H2 (1.5 vs 3.0 Å)
    positions = final_system.get_positions()
    z_coords = positions[:, 2]

    # Find CO and H2 z-coordinates (approximate check)
    surface_top_z = np.max([z for i, z in enumerate(z_coords)
                           if symbols[i] == "Cu"])

    # Both adsorbates should be above the surface
    co_z = z_coords[symbols.index("C")]  # CO carbon position
    h2_z = np.mean([z_coords[i] for i, s in enumerate(symbols) if s == "H"])  # H2 average

    assert co_z > surface_top_z
    assert h2_z > surface_top_z
    assert co_z < h2_z  # CO should be closer to surface


def test_AddAdsorbate_collision_detection_error(proj_path):
    """Test that AddAdsorbate raises error when adsorbates can't be placed without collision."""
    with ips.Project() as proj:
        # Create small surface
        surface = ips.BuildSurface(
            lattice="Au",
            indices=(1, 1, 1),
            layers=2,
            vacuum=10.0,
            lattice_constant=3.0  # Small lattice for small surface
        )

        # Try to place too many large adsorbates
        water1 = ips.Smiles2Atoms(smiles="O")
        water2 = ips.Smiles2Atoms(smiles="O")
        water3 = ips.Smiles2Atoms(smiles="O")

        # Very large exclusion radii that should cause collision
        adsorbed_system = ips.AddAdsorbate(
            slab=surface.frames,
            data=[water1.frames, water2.frames, water3.frames],  # Nested structure
            height=[2.0, 2.0, 2.0],
            excluded_radius=[10.0, 10.0, 10.0]  # Unrealistically large radii
        )

    # This should raise an error during repro due to collision detection
    with pytest.raises(ValueError, match="Could not find valid position"):
        proj.repro()


def test_AddAdsorbate_data_index_functionality(proj_path):
    """Test AddAdsorbate with data_index parameter for selecting specific conformers."""
    with ips.Project() as proj:
        # Create surface
        surface = ips.BuildSurface(
            lattice="Pt",
            indices=(1, 1, 1),
            layers=3,
            vacuum=12.0
        )

        # Create multiple conformers of the same molecule
        water_conformers = ips.Smiles2Conformers(smiles="O", numConfs=5)
        co_molecule = ips.Smiles2Atoms(smiles="[C-]#[O+]")

        # Test with data_index=None (default, uses -1 index)
        adsorbed_system_default = ips.AddAdsorbate(
            slab=surface.frames,
            data=[water_conformers.frames, co_molecule.frames],
            data_index=None,  # Use last conformer of water, last (only) CO
            height=[2.0, 2.5],
            excluded_radius=[2.0, 2.0]
        )

        # Test with specific data_index
        adsorbed_system_custom = ips.AddAdsorbate(
            slab=surface.frames,
            data=[water_conformers.frames, co_molecule.frames],
            data_index=[0, None],  # Use first water conformer, default CO
            height=[2.0, 2.5],
            excluded_radius=[2.0, 2.0]
        )

    proj.repro()

    # Both should have same atom counts but potentially different structures
    system_default = adsorbed_system_default.frames[0]
    system_custom = adsorbed_system_custom.frames[0]

    # Same composition in both cases
    assert len(system_default) == len(system_custom)

    symbols_default = system_default.get_chemical_symbols()
    symbols_custom = system_custom.get_chemical_symbols()

    # Should have same element counts
    for element in set(symbols_default + symbols_custom):
        assert symbols_default.count(element) == symbols_custom.count(element)

    # Should have Pt + water + CO
    pt_count = symbols_default.count("Pt")
    h_count = symbols_default.count("H")
    o_count = symbols_default.count("O")
    c_count = symbols_default.count("C")

    assert pt_count > 0  # Surface atoms
    assert h_count == 2  # Water hydrogens
    assert o_count == 2  # Water oxygen + CO oxygen
    assert c_count == 1  # CO carbon


def test_AddAdsorbate_slab_idx_functionality(proj_path):
    """Test AddAdsorbate with slab_idx parameter for selecting specific slab."""
    with ips.Project() as proj:
        # Create multiple surfaces
        surface1 = ips.BuildSurface(
            lattice="Au",
            indices=(1, 1, 1),
            layers=3,
            vacuum=10.0
        )

        surface2 = ips.BuildSurface(
            lattice="Pt",
            indices=(1, 0, 0),
            layers=4,
            vacuum=12.0
        )

        # Combine surfaces into a list
        combined_surfaces = surface1.frames + surface2.frames

        # Create adsorbate
        water = ips.Smiles2Atoms(smiles="O")

        # Test with slab_idx=0 (first slab - Au)
        adsorbed_au = ips.AddAdsorbate(
            slab=combined_surfaces,
            slab_idx=0,
            data=[water.frames],
            height=[2.0],
            excluded_radius=[2.0]
        )

        # Test with slab_idx=-1 (default, last slab - Pt)
        adsorbed_pt = ips.AddAdsorbate(
            slab=combined_surfaces,
            slab_idx=-1,
            data=[water.frames],
            height=[2.0],
            excluded_radius=[2.0]
        )

    proj.repro()

    # Check that correct surfaces were used
    au_system = adsorbed_au.frames[0]
    pt_system = adsorbed_pt.frames[0]

    au_symbols = au_system.get_chemical_symbols()
    pt_symbols = pt_system.get_chemical_symbols()

    # Au system should have Au atoms
    assert "Au" in au_symbols
    assert "Pt" not in au_symbols

    # Pt system should have Pt atoms
    assert "Pt" in pt_symbols
    assert "Au" not in pt_symbols

    # Both should have water
    assert au_symbols.count("H") == 2
    assert au_symbols.count("O") == 1
    assert pt_symbols.count("H") == 2
    assert pt_symbols.count("O") == 1
