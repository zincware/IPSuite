from pathlib import Path
import typing as t

import ase
import h5py
import znh5md
import zntrack
from ase.build import bulk, surface

from ipsuite import base


class BuildSurface(base.IPSNode):
    """Build crystal surfaces using ase.build.surface functionality.
    
    This node creates crystal surfaces from bulk structures using Miller indices
    to define the surface orientation and adds vacuum layers for surface calculations.
    
    Parameters
    ----------
    lattice : str
        Chemical symbol for the bulk lattice (e.g., 'Au', 'Pt', 'Cu').
    indices : tuple[int, int, int]
        Miller indices defining the surface orientation (e.g., (1, 1, 1), (1, 0, 0)).
    layers : int
        Number of equivalent atomic layers in the slab.
    vacuum : float, optional
        Thickness of vacuum layer in Angstroms, by default 10.0.
    lattice_constant : float | None, optional
        Custom lattice constant in Angstroms, by default None (uses ASE default).
    crystal_structure : str | None, optional
        Crystal structure type ('fcc', 'bcc', 'hcp', 'diamond', 'zincblende'), 
        by default None (uses ASE default).
    
    Attributes
    ----------
    frames : list[ase.Atoms]
        List containing the generated surface structure.
    """
    
    # Surface parameters
    lattice: str = zntrack.params()
    indices: tuple[int, int, int] = zntrack.params()
    layers: int = zntrack.params()
    
    # Optional parameters
    vacuum: float = zntrack.params(10.0)
    lattice_constant: float | None = zntrack.params(None)
    crystal_structure: t.Literal['fcc', 'bcc', 'hcp', 'diamond', 'zincblende'] | None = zntrack.params(None)
    
    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def run(self) -> None:
        """Generate the surface structure and save to frames file."""
        # Create bulk structure
        if self.crystal_structure and self.lattice_constant:
            # Create bulk structure with custom parameters
            bulk_atoms = bulk(
                self.lattice, 
                self.crystal_structure, 
                a=self.lattice_constant,
                cubic=True
            )
        else:
            # Use default bulk structure
            bulk_atoms = self.lattice
        
        # Create surface
        surface_atoms = surface(
            lattice=bulk_atoms,
            indices=self.indices,
            layers=self.layers
        )
        
        # Add vacuum
        surface_atoms.center(vacuum=self.vacuum, axis=2)
        
        # Save to frames file
        io = znh5md.IO(filename=self.frames_path)
        io.append(surface_atoms)

    @property
    def frames(self) -> list[ase.Atoms]:
        """Get the generated surface structure.
        
        Returns
        -------
        list[ase.Atoms]
            List containing the surface structure.
        """
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]