import typing as t
from pathlib import Path

import ase
import h5py
import numpy as np
import znh5md
import zntrack
from ase.build import add_adsorbate, bulk, surface

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
    crystal_structure: t.Literal["fcc", "bcc", "hcp", "diamond", "zincblende"] | None = (
        zntrack.params(None)
    )

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def run(self) -> None:
        """Generate the surface structure and save to frames file."""
        # Create bulk structure
        if self.crystal_structure and self.lattice_constant:
            # Create bulk structure with custom parameters
            bulk_atoms = bulk(
                self.lattice, self.crystal_structure, a=self.lattice_constant, cubic=True
            )
        else:
            # Use default bulk structure
            bulk_atoms = self.lattice

        # Create surface
        surface_atoms = surface(
            lattice=bulk_atoms, indices=self.indices, layers=self.layers
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


class AddAdsorbate(base.IPSNode):
    """Add adsorbate molecules to a surface slab with collision detection.

    This node places adsorbate molecules on a surface at specified heights with
    automatic positioning to avoid collisions between adsorbates.

    Parameters
    ----------
    slab : list[ase.Atoms]
        List containing surface slab structures.
    slab_idx : int, optional
        Index to select which slab to use from the slab list. Defaults to -1 (last).
    data : list[list[ase.Atoms]]
        List of lists of adsorbate molecules. Each inner list represents different
        conformations/structures of the same adsorbate species.
    data_index : list[int | None] | None, optional
        Indices to select from each adsorbate list in data. If None (default),
        the last element (-1 index) is selected from each list. If provided,
        must have same length as data list.
    height : list[float]
        Heights (in Angstroms) at which to place each adsorbate above the surface.
        Length must match the number of adsorbates to be placed.
    excluded_radius : list[float]
        Exclusion radius (in Angstroms) around each adsorbate position where
        no other adsorbate can be placed. Length must match height list.

    Attributes
    ----------
    frames : list[ase.Atoms]
        List containing the surface with adsorbed molecules.
    """

    slab: list[ase.Atoms] = zntrack.deps()
    slab_idx: int = zntrack.params(-1)
    data: list[list[ase.Atoms]] = zntrack.deps()
    data_index: list[int | None] | None = zntrack.params(None)
    height: list[float] = zntrack.params()
    excluded_radius: list[float] = zntrack.params()

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def _get_com_position(self, atoms: ase.Atoms) -> np.ndarray:
        """Calculate center of mass for adsorbate positioning.

        Parameters
        ----------
        atoms : ase.Atoms
            Adsorbate molecule.

        Returns
        -------
        np.ndarray
            Center of mass coordinates [x, y, z].
        """
        masses = atoms.get_masses()
        positions = atoms.get_positions()
        return np.average(positions, weights=masses, axis=0)

    def _get_surface_center(self, slab: ase.Atoms) -> tuple[float, float]:
        """Get the center position of the surface in x-y plane.

        Parameters
        ----------
        slab : ase.Atoms
            Surface slab structure.

        Returns
        -------
        tuple[float, float]
            Center coordinates (x, y) of the surface.
        """
        cell = slab.get_cell()
        return (cell[0, 0] / 2, cell[1, 1] / 2)

    def _check_collision(
        self,
        new_pos: tuple[float, float],
        existing_positions: list[tuple[float, float]],
        new_radius: float,
        existing_radii: list[float],
    ) -> bool:
        """Check if new adsorbate position collides with existing ones.

        Parameters
        ----------
        new_pos : tuple[float, float]
            Proposed (x, y) position for new adsorbate.
        existing_positions : list[tuple[float, float]]
            List of (x, y) positions of already placed adsorbates.
        new_radius : float
            Exclusion radius for the new adsorbate.
        existing_radii : list[float]
            Exclusion radii for existing adsorbates.

        Returns
        -------
        bool
            True if collision detected, False otherwise.
        """
        for i, (ex_x, ex_y) in enumerate(existing_positions):
            distance = np.sqrt((new_pos[0] - ex_x) ** 2 + (new_pos[1] - ex_y) ** 2)
            min_distance = max(new_radius, existing_radii[i])
            if distance < min_distance:
                return True
        return False

    def _find_valid_position(
        self,
        slab: ase.Atoms,
        radius: float,
        existing_positions: list[tuple[float, float]],
        existing_radii: list[float],
    ) -> tuple[float, float]:
        """Find a valid position for adsorbate that doesn't collide with existing ones.

        Parameters
        ----------
        slab : ase.Atoms
            Surface slab structure.
        radius : float
            Exclusion radius for the new adsorbate.
        existing_positions : list[tuple[float, float]]
            Positions of existing adsorbates.
        existing_radii : list[float]
            Exclusion radii of existing adsorbates.

        Returns
        -------
        tuple[float, float]
            Valid (x, y) position for the adsorbate.

        Raises
        ------
        ValueError
            If no valid position can be found.
        """
        cell = slab.get_cell()
        max_x, max_y = cell[0, 0], cell[1, 1]

        # Try different positions with increasing distance from existing adsorbates
        for attempt in range(1000):  # Limit attempts to prevent infinite loops
            # Use systematic grid search with some randomness
            if attempt < 100:
                # Systematic grid search
                grid_size = int(np.sqrt(attempt + 1))
                i, j = attempt % grid_size, attempt // grid_size
                x = (i + 0.5) * max_x / (grid_size + 1)
                y = (j + 0.5) * max_y / (grid_size + 1)
            else:
                # Random positions for remaining attempts
                x = np.random.uniform(radius, max_x - radius)
                y = np.random.uniform(radius, max_y - radius)

            if not self._check_collision(
                (x, y), existing_positions, radius, existing_radii
            ):
                return (x, y)

        raise ValueError(
            f"Could not find valid position for adsorbate with radius {radius}. "
            "Consider reducing excluded_radius values or using a larger surface."
        )

    def _get_selected_adsorbates(self) -> list[ase.Atoms]:
        """Extract the selected adsorbates from the nested data structure.

        Returns
        -------
        list[ase.Atoms]
            List of selected adsorbate molecules.

        Raises
        ------
        ValueError
            If data_index is provided but has wrong length.
        """
        if self.data_index is None:
            # Use -1 index (last element) from each list
            return [adsorbate_list[-1] for adsorbate_list in self.data]

        if len(self.data_index) != len(self.data):
            raise ValueError(
                f"Length of data_index ({len(self.data_index)}) must match "
                f"length of data ({len(self.data)})"
            )

        selected_adsorbates = []
        for i, (adsorbate_list, index) in enumerate(zip(self.data, self.data_index)):
            if index is None:
                # Use -1 index for this specific adsorbate
                selected_adsorbates.append(adsorbate_list[-1])
            else:
                # Use specified index
                try:
                    selected_adsorbates.append(adsorbate_list[index])
                except IndexError:
                    raise ValueError(
                        f"Index {index} is out of bounds for adsorbate list {i} "
                        f"with length {len(adsorbate_list)}"
                    )

        return selected_adsorbates

    def run(self) -> None:
        """Add adsorbates to the surface slab."""
        if len(self.height) != len(self.excluded_radius):
            raise ValueError("Length of height and excluded_radius lists must match")

        # Get selected adsorbates from nested structure
        adsorbates = self._get_selected_adsorbates()

        n_adsorbates = min(len(adsorbates), len(self.height))
        if n_adsorbates == 0:
            raise ValueError("No adsorbates to place")

        # Start with the selected surface slab
        slab = self.slab[self.slab_idx].copy()

        # Track positions and radii of placed adsorbates
        placed_positions = []
        placed_radii = []

        for i in range(n_adsorbates):
            adsorbate = adsorbates[i]
            height = self.height[i]
            radius = self.excluded_radius[i]

            if i == 0:
                # Place first adsorbate at surface center
                position = self._get_surface_center(slab)
            else:
                # Find valid position for subsequent adsorbates
                position = self._find_valid_position(
                    slab, radius, placed_positions, placed_radii
                )

            # Add adsorbate to slab
            # Use mol_index=0 instead of None to avoid array broadcasting issues
            add_adsorbate(
                slab=slab,
                adsorbate=adsorbate,
                height=height,
                position=position,
                offset=None,
                mol_index=0,  # Use first atom as reference instead of COM
            )

            # Clean up info dict to ensure JSON serializability
            if "adsorbate_info" in slab.info:
                # Convert numpy int64 to regular int for JSON compatibility
                adsorbate_info = slab.info["adsorbate_info"]
                if isinstance(adsorbate_info, dict):
                    for key, value in adsorbate_info.items():
                        if (
                            hasattr(value, "item") and value.size == 1
                        ):  # numpy scalar with single element
                            adsorbate_info[key] = value.item()
                        elif isinstance(value, np.integer):
                            adsorbate_info[key] = int(value)
                        elif isinstance(value, np.ndarray):
                            adsorbate_info[key] = (
                                value.tolist()
                            )  # Convert arrays to lists

            # Track this adsorbate's position and radius
            placed_positions.append(position)
            placed_radii.append(radius)

        # Save result
        io = znh5md.IO(filename=self.frames_path)
        io.append(slab)

    @property
    def frames(self) -> list[ase.Atoms]:
        """Get the surface with adsorbed molecules.

        Returns
        -------
        list[ase.Atoms]
            List containing the surface with adsorbates.
        """
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]
