"""ipsuite data loading with ASE."""

import logging
import pathlib
import typing

import ase.io
import h5py
import tqdm
import znh5md
import zntrack

from ipsuite import base

log = logging.getLogger(__name__)


def load_data(
    file: typing.Union[str, pathlib.Path],
    lines_to_read: int = None,
):
    """Add data to the database.

    Parameters
    ----------
    database: Database
        instance of a database to add data to
    file: str|Path
        path to the file that should be added to the database
    lines_to_read: int, optional
        maximal number of lines/configurations to read, None for read all
    """
    if isinstance(file, str):
        file = pathlib.Path(file)

    frames = []
    for config, atoms in enumerate(
        tqdm.tqdm(ase.io.iread(file.as_posix()), desc="Reading File", ncols=70)
    ):
        if lines_to_read is not None and config >= lines_to_read:
            break
        frames.append(atoms)
    return frames


class AddData(base.IPSNode):
    """Load atomic configurations from files using ASE.

    Parameters
    ----------
    file : str or Path
        Path to the file containing atomic configurations.
    lines_to_read : int, optional
        Maximum number of configurations to read. If None, reads all.

    Attributes
    ----------
    frames : List[ase.Atoms]
        List of loaded atomic configurations.

    Examples
    --------
    >>> with project:
    ...     data = ips.AddData(file="ethanol.xyz", lines_to_read=50)
    >>> project.repro()
    >>> print(f"Loaded {len(data.frames)} configurations.")
    Loaded 50 configurations.
    """

    file: typing.Union[str, pathlib.Path] = zntrack.deps_path()
    lines_to_read: int | None = zntrack.params(None)
    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.h5")

    def __post_init__(self):
        if not pathlib.Path(pathlib.Path(self.file).name + ".dvc").exists():
            log.warning(
                f"Please run 'dvc add {self.file}' to track the file with DVC. Otherwise,"
                " it might end up being git tracked."
            )

    def run(self):
        """ZnTrack run method."""
        if self.lines_to_read == -1:  # backwards compatibility
            self.lines_to_read = None
        frames = load_data(file=self.file, lines_to_read=self.lines_to_read)
        io = znh5md.IO(self.frames_path)
        io.extend(frames)

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]

    @classmethod
    def save_atoms_to_file(cls, atoms: typing.List[ase.Atoms], file: str):
        """Save atoms to a file.

        Parameters
        ----------
        atoms: list
            list of atoms objects
        file: str
            path to the file to save to
        """
        ase.io.write(file, atoms, format="extxyz")
