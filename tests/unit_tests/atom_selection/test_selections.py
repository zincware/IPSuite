"""Tests for atom selection implementations."""

import numpy as np
import pytest
from ase import Atoms

import ipsuite as ips


def test_element_type_selection():
    """Test ElementTypeSelection with different element types."""
    # Create test atoms with mixed elements
    atoms = Atoms(['H', 'H', 'O', 'C', 'C', 'N'], 
                  positions=np.random.rand(6, 3) * 10)
    
    # Test single element selection
    h_selector = ips.ElementTypeSelection(elements=['H'])
    h_indices = h_selector.select(atoms)
    assert h_indices == [0, 1]
    
    # Test multiple element selection
    multi_selector = ips.ElementTypeSelection(elements=['O', 'N'])
    multi_indices = multi_selector.select(atoms)
    assert set(multi_indices) == {2, 5}
    
    # Test no matches
    empty_selector = ips.ElementTypeSelection(elements=['Pt'])
    empty_indices = empty_selector.select(atoms)
    assert empty_indices == []


def test_z_position_selection():
    """Test ZPositionSelection with various Z-coordinate ranges."""
    # Create atoms at different Z positions
    positions = np.array([
        [0, 0, 1.0],
        [0, 0, 2.5], 
        [0, 0, 4.0],
        [0, 0, 5.5],
        [0, 0, 7.0]
    ])
    atoms = Atoms(['H'] * 5, positions=positions)
    
    # Test Z range selection
    mid_selector = ips.ZPositionSelection(z_min=2.0, z_max=5.0)
    mid_indices = mid_selector.select(atoms)
    assert set(mid_indices) == {1, 2}
    
    # Test only minimum
    min_selector = ips.ZPositionSelection(z_min=5.0)
    min_indices = min_selector.select(atoms)
    assert set(min_indices) == {3, 4}
    
    # Test only maximum  
    max_selector = ips.ZPositionSelection(z_max=3.0)
    max_indices = max_selector.select(atoms)
    assert set(max_indices) == {0, 1}


def test_radial_selection():
    """Test RadialSelection with different center types."""
    # Create atoms in a simple pattern
    positions = np.array([
        [0, 0, 0],     # at origin
        [1, 0, 0],     # 1 unit away
        [0, 2, 0],     # 2 units away
        [3, 0, 0],     # 3 units away
        [0, 0, 4]      # 4 units away
    ])
    atoms = Atoms(['C'] * 5, positions=positions)
    
    # Test radial selection from specific center
    center_selector = ips.RadialSelection(center=(0, 0, 0), radius=2.5)
    center_indices = center_selector.select(atoms)
    assert set(center_indices) == {0, 1, 2}  # Within 2.5 units of origin
    
    # Test center of mass (should be at average position)
    com_selector = ips.RadialSelection(center='com', radius=1.0)
    com_indices = com_selector.select(atoms)
    # COM is at (0.8, 0.4, 0.8), so check which atoms are within 1.0 unit
    com = atoms.get_center_of_mass()
    expected = []
    for i, pos in enumerate(positions):
        if np.linalg.norm(pos - com) <= 1.0:
            expected.append(i)
    assert set(com_indices) == set(expected)


def test_layer_selection():
    """Test LayerSelection for identifying atomic layers."""
    # Create a simple 3-layer slab
    positions = np.array([
        # Bottom layer (z=0)
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        # Middle layer (z=2) 
        [0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2],
        # Top layer (z=4)
        [0, 0, 4], [1, 0, 4], [0, 1, 4], [1, 1, 4]
    ])
    atoms = Atoms(['Pt'] * 12, positions=positions)
    
    # Test bottom layer selection
    bottom_selector = ips.LayerSelection(layer_indices=[0])
    bottom_indices = bottom_selector.select(atoms)
    assert set(bottom_indices) == {0, 1, 2, 3}
    
    # Test top layer selection
    top_selector = ips.LayerSelection(layer_indices=[-1])
    top_indices = top_selector.select(atoms)
    assert set(top_indices) == {8, 9, 10, 11}
    
    # Test multiple layers
    multi_selector = ips.LayerSelection(layer_indices=[0, -1])
    multi_indices = multi_selector.select(atoms)
    assert set(multi_indices) == {0, 1, 2, 3, 8, 9, 10, 11}


def test_surface_selection():
    """Test SurfaceSelection for identifying surface atoms."""
    # Create a simple bulk-like structure with surface atoms
    # 2x2x2 cube with one atom removed to create a surface
    positions = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],  # bottom layer
        [0, 0, 1], [1, 0, 1], [0, 1, 1],             # top layer (missing one atom)
    ])
    atoms = Atoms(['Cu'] * 7, positions=positions)
    
    # All atoms should be surface atoms due to low coordination
    surface_selector = ips.SurfaceSelection(cutoff=1.5, min_neighbors=6)  
    surface_indices = surface_selector.select(atoms)
    
    # In this small cluster, all atoms have < 6 neighbors within 1.5 Å
    assert len(surface_indices) == 7  # All atoms are surface atoms


def test_fix_atoms_constraint():
    """Test FixAtomsConstraint integration with element selection."""
    atoms = Atoms(['Pt', 'Pt', 'H', 'O'], 
                  positions=np.random.rand(4, 3) * 5)
    
    # Create constraint to fix Pt atoms
    pt_selector = ips.ElementTypeSelection(elements=['Pt'])
    constraint = ips.FixAtomsConstraint(selection=pt_selector)
    
    # Get ASE constraint
    ase_constraint = constraint.get_constraint(atoms)
    
    # Check that correct atoms are fixed
    assert set(ase_constraint.index) == {0, 1}


def test_fix_atoms_constraint_empty_selection():
    """Test FixAtomsConstraint with empty selection raises error."""
    atoms = Atoms(['H', 'O'], positions=np.random.rand(2, 3) * 5)
    
    # Try to fix non-existent atoms
    empty_selector = ips.ElementTypeSelection(elements=['Pt'])
    constraint = ips.FixAtomsConstraint(selection=empty_selector)
    
    with pytest.raises(ValueError, match="No atoms selected"):
        constraint.get_constraint(atoms)


def test_selection_combination_workflow():
    """Test realistic workflow combining surface building with atom selection."""
    # This would be an integration test showing the full workflow
    # Create a surface, select atoms, and apply constraints
    
    # Create test surface-like structure
    positions = np.array([
        # Bottom layer (bulk-like)
        [0, 0, 0], [2, 0, 0], [1, 1.7, 0],
        # Top layer (surface)  
        [0, 0, 2], [2, 0, 2], [1, 1.7, 2]
    ])
    atoms = Atoms(['Pt'] * 6, positions=positions)
    
    # Select bottom layer atoms
    bottom_selector = ips.LayerSelection(layer_indices=[0], tolerance=0.5)
    bottom_indices = bottom_selector.select(atoms)
    assert set(bottom_indices) == {0, 1, 2}
    
    # Create constraint to fix bottom layer
    fix_constraint = ips.FixAtomsConstraint(selection=bottom_selector)
    ase_constraint = fix_constraint.get_constraint(atoms)
    
    # Verify constraint fixes the right atoms
    assert set(ase_constraint.index) == {0, 1, 2}
    
    # This constraint could now be used with ASEGeoOpt for surface relaxation
    # where only the surface layer atoms are allowed to move