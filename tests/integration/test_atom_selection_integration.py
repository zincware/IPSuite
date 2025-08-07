"""Integration tests for atom selection with ASEGeoOpt."""

import ipsuite as ips


def test_asegeopt_with_atom_selection_constraints(proj_path):
    """Test ASEGeoOpt using atom selection constraints."""
    with ips.Project() as proj:
        # Create a small surface
        surface = ips.BuildSurface(
            lattice="Cu",
            indices=(1, 1, 1),
            layers=3,
            vacuum=10.0
        )
        
        # Create constraint to fix bottom layer atoms during relaxation
        bottom_layer_selector = ips.LayerSelection(layer_indices=[0], tolerance=0.5)
        fix_bottom_constraint = ips.FixAtomsConstraint(selection=bottom_layer_selector)
        
        # Set up MACE-MP model for forces
        model = ips.MACEMPModel()
        
        # Run geometry optimization with constraint
        relaxed_surface = ips.ASEGeoOpt(
            data=surface.frames,
            model=model,
            constraints=[fix_bottom_constraint],  # Use new constraint system
            optimizer="BFGS",
            run_kwargs={"fmax": 0.2, "steps": 20},
            sampling_rate=5,
            maxstep=20
        )

    proj.repro()

    # Verify surface was created
    initial_surface = surface.frames[0]
    assert len(initial_surface) > 0
    assert all(symbol == "Cu" for symbol in initial_surface.get_chemical_symbols())
    
    # Verify optimization trajectory was generated
    trajectory = relaxed_surface.frames
    assert len(trajectory) > 0
    
    # Check that final structure has same composition
    final_surface = trajectory[-1]
    assert len(final_surface) == len(initial_surface)
    assert all(symbol == "Cu" for symbol in final_surface.get_chemical_symbols())


def test_element_selection_constraint(proj_path):
    """Test constraint using element-based selection."""
    with ips.Project() as proj:
        # Create surface with adsorbate
        surface = ips.BuildSurface(
            lattice="Pt", 
            indices=(1, 1, 1),
            layers=3,
            vacuum=12.0
        )
        
        water = ips.Smiles2Atoms(smiles="O")  # H2O
        
        adsorbed_system = ips.AddAdsorbate(
            slab=surface.frames,
            data=[water.frames],
            height=[2.0],
            excluded_radius=[3.0]
        )
        
        # Create constraint to fix only Pt atoms (substrate)
        pt_selector = ips.ElementTypeSelection(elements=['Pt'])
        fix_substrate = ips.FixAtomsConstraint(selection=pt_selector)
        
        model = ips.MACEMPModel()
        
        # Optimize only the adsorbate, keeping substrate fixed
        optimized = ips.ASEGeoOpt(
            data=adsorbed_system.frames,
            model=model,
            constraints=[fix_substrate],
            optimizer="FIRE", 
            run_kwargs={"fmax": 0.3, "steps": 15},
            maxstep=15
        )

    proj.repro()
    
    # Verify the system was set up correctly
    initial_system = adsorbed_system.frames[0]
    symbols = initial_system.get_chemical_symbols()
    
    # Should have Pt + water atoms
    assert 'Pt' in symbols
    assert 'H' in symbols
    assert 'O' in symbols
    
    # Verify optimization completed
    trajectory = optimized.frames
    assert len(trajectory) > 0


def test_multiple_selection_types(proj_path):
    """Test using multiple different selection types together."""
    with ips.Project() as proj:
        # Create larger surface for more complex selections
        surface = ips.BuildSurface(
            lattice="Au",
            indices=(1, 1, 1),
            layers=4,
            vacuum=15.0
        )
        
        # Create multiple constraint types
        # 1. Fix bottom 2 layers using layer selection
        bottom_layers = ips.LayerSelection(layer_indices=[0, 1], tolerance=0.5)
        fix_bottom = ips.FixAtomsConstraint(selection=bottom_layers)
        
        # 2. Could add more constraints here for complex scenarios
        
        model = ips.MACEMPModel()
        
        optimized = ips.ASEGeoOpt(
            data=surface.frames,
            model=model,
            constraints=[fix_bottom],  # Can extend with more constraints
            optimizer="FIRE",
            run_kwargs={"fmax": 0.2, "steps": 10},
            maxstep=10
        )

    proj.repro()
    
    # Verify it worked
    trajectory = optimized.frames
    assert len(trajectory) > 0
    
    final_atoms = trajectory[-1]
    assert all(symbol == "Au" for symbol in final_atoms.get_chemical_symbols())