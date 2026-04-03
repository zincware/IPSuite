import logging
import typing
import warnings
from pathlib import Path

import h5py
import MDAnalysis as mda
import numpy as np
import tqdm
import znh5md
import zntrack
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import kJ, mol
from MDAnalysis.auxiliary.EDR import EDRReader

logger = logging.getLogger(__name__)

_TYPE_TO_ELEMENT = {
    "CL": "Cl",
    "NA": "Na",
    "MG": "Mg",
    "ZN": "Zn",
    "FE": "Fe",
    "CA": "Ca",
    "MN": "Mn",
    "CU": "Cu",
    "LI": "Li",
    "AL": "Al",
    "SI": "Si",
    "BR": "Br",
    "SE": "Se",
}


def _get_symbols(u: mda.Universe) -> list[str]:
    """Extract element symbols from a Universe, trying multiple strategies."""
    # 1. Use elements attribute if available
    try:
        return list(u.atoms.elements)
    except (mda.exceptions.NoDataError, AttributeError):
        pass

    # 2. Use atom types (usually cleaner than names for CHARMM-GUI)
    types = u.atoms.types
    symbols = []
    for t in types:
        t_upper = t.upper()
        if t_upper in _TYPE_TO_ELEMENT:
            symbols.append(_TYPE_TO_ELEMENT[t_upper])
        elif len(t) <= 2 and t[0].isalpha():
            # Capitalize properly: first letter upper, rest lower
            symbols.append(t[0].upper() + t[1:].lower() if len(t) > 1 else t.upper())
        else:
            # Last resort: take leading alphabetic characters from atom name
            symbols.append(t[0].upper())
    return symbols


def _match_edr_frame(
    edr_times: np.ndarray, traj_time: float, tolerance: float = 0.1
) -> int:
    """Find the EDR index closest to a trajectory time, warning on large gaps."""
    idx = int(np.argmin(np.abs(edr_times - traj_time)))
    time_diff = abs(edr_times[idx] - traj_time)
    if time_diff > tolerance:
        logger.warning(
            "EDR time %.3f ps does not match trajectory time %.3f ps (diff=%.3f ps)",
            edr_times[idx],
            traj_time,
            time_diff,
        )
    return idx


def gmx_to_ase(
    topology: str,
    trajectory: str | None = None,
    edr: str | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> list[Atoms]:
    """Convert a GROMACS trajectory to a list of ASE Atoms objects.

    Extracts all available information: positions, velocities, forces,
    and (via the .edr file) energies and stress.

    Parameters
    ----------
    topology : str
        Path to a GROMACS topology/structure file (.gro, .tpr).
    trajectory : str | None
        Path to a trajectory file (.xtc, .trr). If None, only the single
        structure from the topology file is returned.
    edr : str | None
        Path to a GROMACS energy file (.edr). If given, per-frame energies
        and stress tensors are attached via SinglePointCalculator.
    start, stop, step : int | None
        Slice parameters for selecting a subset of frames.

    Returns
    -------
    list[Atoms]
        One ASE Atoms object per frame. Each Atoms has:
        - positions (always)
        - cell and pbc (always)
        - velocities (if present in trajectory)
        - forces (if present in trajectory, e.g. .trr)
        - calculator with energy/stress/forces (if .edr provided or forces
          present), plus all EDR terms stored in calc.results
    """
    if trajectory is not None:
        u = mda.Universe(topology, trajectory)
    else:
        u = mda.Universe(topology)

    symbols = _get_symbols(u)

    # Load EDR data if provided
    edr_data = None
    if edr is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = EDRReader(edr)
            edr_all = reader.get_data(list(reader.terms))
            edr_times = edr_all.pop("Time")
            edr_data = dict(edr_all)
            edr_terms = list(edr_data.keys())

    frames = []
    for ts in tqdm.tqdm(u.trajectory[start:stop:step]):
        positions = ts.positions.copy()
        box = ts.dimensions

        atoms = Atoms(symbols=symbols, positions=positions, pbc=True)

        if box is not None and all(box[:3] > 0):
            atoms.set_cell(box, scale_atoms=False)

        # Velocities (e.g. from .gro or .trr)
        if ts.has_velocities:
            # MDAnalysis: Å/ps, ASE: Å/fs -> divide by 1000
            atoms.set_velocities(ts.velocities / 1000.0)

        # Forces and energies via SinglePointCalculator
        forces = ts.forces.copy() if ts.has_forces else None
        energy = None
        stress = None
        extra_results = {}

        if edr_data is not None:
            idx = _match_edr_frame(edr_times, ts.time)
            energy = float(edr_data["Potential"][idx]) * (kJ / mol)  # convert to eV

            # Build Voigt stress from pressure tensor if available
            try:
                pxx = edr_data["Pres-XX"][idx]
                pyy = edr_data["Pres-YY"][idx]
                pzz = edr_data["Pres-ZZ"][idx]
                pyz = edr_data["Pres-YZ"][idx]
                pxz = edr_data["Pres-XZ"][idx]
                pxy = edr_data["Pres-XY"][idx]
                # GROMACS pressure in bar -> store as-is (not ASE native eV/Å³)
                stress = np.array([pxx, pyy, pzz, pyz, pxz, pxy])
            except KeyError:
                pass

            # Store all EDR terms for this frame
            for term in edr_terms:
                extra_results[term] = float(edr_data[term][idx])

        if energy is not None or forces is not None:
            calc = SinglePointCalculator(
                atoms,
                energy=energy,
                forces=forces,
                stress=stress,
            )
            calc.results.update(extra_results)
            atoms.calc = calc

        frames.append(atoms)

    return frames


class Gmx2Frames(zntrack.Node):
    """Convert GROMACS output files to ASE Atoms frames.

    Reads topology, trajectory, and optionally energy (.edr) files
    to produce a list of ASE Atoms with positions, velocities, forces,
    energies, and stress where available.

    Parameters
    ----------
    topology : Path
        Path to a GROMACS topology/structure file (.gro, .tpr).
    trajectory : Path, optional
        Path to a trajectory file (.xtc, .trr).
    edr : Path, optional
        Path to a GROMACS energy file (.edr).
    start : int, optional
        First frame index to read.
    stop : int, optional
        Last frame index (exclusive) to read.
    step : int, optional
        Step size for frame selection.

    Examples
    --------
    >>> with project:
    ...     md = ips.Gmx2Frames(
    ...         topology="gromacs/system.gro",
    ...         trajectory="gromacs/production.xtc",
    ...         edr="gromacs/production.edr",
    ...         start=1,
    ...     )
    """

    topology: Path = zntrack.deps_path()
    trajectory: Path | None = zntrack.deps_path(None)
    edr: Path | None = zntrack.deps_path(None)
    start: int | None = zntrack.params(None)
    stop: int | None = zntrack.params(None)
    step: int | None = zntrack.params(None)

    frames_path: Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def run(self) -> None:
        data = gmx_to_ase(
            topology=str(self.topology),
            trajectory=str(self.trajectory) if self.trajectory else None,
            edr=str(self.edr) if self.edr else None,
            start=self.start,
            stop=self.stop,
            step=self.step,
        )
        frame_io = znh5md.IO(self.frames_path)
        frame_io.extend(data)

    @property
    def frames(self) -> typing.List[Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]


if __name__ == "__main__":
    # Example: load the production trajectory with energies
    frames = gmx_to_ase(
        "gromacs/system.gro",
        "gromacs/production.xtc",
        edr="gromacs/production.edr",
    )
    print(f"Loaded {len(frames)} frames, {len(frames[0])} atoms per frame")
    print(f"Cell: {frames[0].cell.cellpar()}")
    print(f"Potential energy (frame 0): {frames[0].get_potential_energy()} eV")
    if len(frames) >= 2:
        print(f"All EDR terms on frame 1: {list(frames[1].calc.results.keys())}")
