import ipsuite as ips


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