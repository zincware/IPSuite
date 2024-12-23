"""ipsuite data loading with ASE."""

import logging
import pathlib
import typing

import ase.io
import tqdm
import zntrack

from ipsuite import base, fields

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
    """Add data using ASE.

    Attributes
    ----------
    use_dvc: bool
        Don't use the filename as a parameter but rather use dvc add <file>
        to track the file with DVC.
    """

    frames: typing.List[ase.Atoms] = fields.Atoms()
    file: typing.Union[str, pathlib.Path] = zntrack.deps_path()
    lines_to_read: int = zntrack.params(None)

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
        self.frames = load_data(file=self.file, lines_to_read=self.lines_to_read)

    def __iter__(self) -> ase.Atoms:
        """Get iterable object."""
        yield from self.frames

    def __len__(self) -> int:
        """Get the number of atoms."""
        return len(self.frames)

    def __getitem__(self, item):
        """Access atoms objects directly and via advanced slicing."""
        if isinstance(item, list):
            return [self.frames[idx] for idx in item]
        return self.frames[item]

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
