"""Concrete atom selection implementations."""

import dataclasses
import typing as t

import ase
import numpy as np

from ipsuite.interfaces import AtomSelector


@dataclasses.dataclass
class ElementTypeSelection(AtomSelector):
    """Select atoms based on element types.

    Parameters
    ----------
    elements : list[str]
        List of element symbols to select (e.g., ['H', 'O', 'C']).
    """

    elements: list[str]

    def select(self, atoms: ase.Atoms) -> list[int]:
        """Select atoms based on the provided element types.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure to select from.

        Returns
        -------
        list[int]
            Indices of atoms with the specified elements.
        """
        symbols = atoms.get_chemical_symbols()
        return [i for i, symbol in enumerate(symbols) if symbol in self.elements]


@dataclasses.dataclass
class ZPositionSelection(AtomSelector):
    """Select atoms based on their Z-coordinate position.

    Parameters
    ----------
    z_min : float | None, optional
        Minimum Z-coordinate (inclusive). If None, no lower bound.
    z_max : float | None, optional
        Maximum Z-coordinate (inclusive). If None, no upper bound.
    """

    z_min: float | None = None
    z_max: float | None = None

    def select(self, atoms: ase.Atoms) -> list[int]:
        """Select atoms within the Z-coordinate range.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure to select from.

        Returns
        -------
        list[int]
            Indices of atoms within the specified Z range.
        """
        positions = atoms.get_positions()
        z_coords = positions[:, 2]

        mask = np.ones(len(atoms), dtype=bool)

        if self.z_min is not None:
            mask &= z_coords >= self.z_min
        if self.z_max is not None:
            mask &= z_coords <= self.z_max

        return np.where(mask)[0].tolist()


@dataclasses.dataclass
class RadialSelection(AtomSelector):
    """Select atoms within a radial distance from a center point.

    Parameters
    ----------
    center : tuple[float, float, float] | str
        Center point for radial selection. Can be coordinates (x, y, z)
        or 'com' for center of mass, 'geometric' for geometric center.
    radius : float
        Selection radius in Angstroms.
    """

    center: tuple[float, float, float] | t.Literal["com", "geometric"]
    radius: float

    def _get_center_position(self, atoms: ase.Atoms) -> np.ndarray:
        """Get the center position for radial selection.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure.

        Returns
        -------
        np.ndarray
            Center position coordinates.
        """
        if isinstance(self.center, tuple):
            return np.array(self.center)
        elif self.center == "com":
            return atoms.get_center_of_mass()
        elif self.center == "geometric":
            return atoms.get_positions().mean(axis=0)
        else:
            raise ValueError(f"Unknown center type: {self.center}")

    def select(self, atoms: ase.Atoms) -> list[int]:
        """Select atoms within radial distance from center.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure to select from.

        Returns
        -------
        list[int]
            Indices of atoms within the specified radius.
        """
        center_pos = self._get_center_position(atoms)
        positions = atoms.get_positions()

        distances = np.linalg.norm(positions - center_pos, axis=1)
        mask = distances <= self.radius

        return np.where(mask)[0].tolist()


@dataclasses.dataclass
class LayerSelection(AtomSelector):
    """Select atoms from specific layers in a slab.

    Parameters
    ----------
    layer_indices : list[int]
        Layer indices to select (0 = bottom layer, -1 = top layer).
    tolerance : float, optional
        Z-coordinate tolerance for grouping atoms into layers, by default 0.5.
    """

    layer_indices: list[int]
    tolerance: float = 0.5

    def _identify_layers(self, atoms: ase.Atoms) -> list[list[int]]:
        """Identify atomic layers based on Z-coordinates.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure.

        Returns
        -------
        list[list[int]]
            List of lists, where each inner list contains atom indices
            for atoms in that layer (sorted by Z-coordinate).
        """
        positions = atoms.get_positions()
        z_coords = positions[:, 2]

        # Sort atoms by Z-coordinate
        sorted_indices = np.argsort(z_coords)
        sorted_z = z_coords[sorted_indices]

        # Group atoms into layers based on tolerance
        layers = []
        current_layer = [sorted_indices[0]]
        current_z = sorted_z[0]

        for i in range(1, len(sorted_indices)):
            if sorted_z[i] - current_z <= self.tolerance:
                current_layer.append(sorted_indices[i])
            else:
                layers.append(current_layer)
                current_layer = [sorted_indices[i]]
                current_z = sorted_z[i]

        layers.append(current_layer)  # Add final layer
        return layers

    def select(self, atoms: ase.Atoms) -> list[int]:
        """Select atoms from specified layers.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure to select from.

        Returns
        -------
        list[int]
            Indices of atoms in the specified layers.
        """
        layers = self._identify_layers(atoms)
        selected_indices = []

        for layer_idx in self.layer_indices:
            if -len(layers) <= layer_idx < len(layers):
                selected_indices.extend(layers[layer_idx])

        return sorted(selected_indices)


@dataclasses.dataclass
class SurfaceSelection(AtomSelector):
    """Select surface atoms (atoms with fewer neighbors than bulk).

    Parameters
    ----------
    cutoff : float, optional
        Cutoff distance for neighbor counting, by default 3.0.
    min_neighbors : int, optional
        Minimum number of neighbors for an atom to be considered bulk,
        by default 8. Atoms with fewer neighbors are considered surface atoms.
    """

    cutoff: float = 3.0
    min_neighbors: int = 8

    def select(self, atoms: ase.Atoms) -> list[int]:
        """Select surface atoms based on neighbor count.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure to select from.

        Returns
        -------
        list[int]
            Indices of surface atoms.
        """
        positions = atoms.get_positions()
        surface_indices = []

        for i in range(len(atoms)):
            # Calculate distances to all other atoms
            distances = np.linalg.norm(positions - positions[i], axis=1)
            # Count neighbors within cutoff (excluding self)
            neighbor_count = np.sum((distances > 0) & (distances <= self.cutoff))

            if neighbor_count < self.min_neighbors:
                surface_indices.append(i)

        return surface_indices
