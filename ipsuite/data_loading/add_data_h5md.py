"""Load Data directly from a H5MD trajectory file."""

import typing
from pathlib import Path

import ase
import h5py
import znh5md
import zntrack

from ipsuite import base


class AddDataH5MD(base.IPSNode):
    """Load Data directly from a H5MD trajectory file."""

    file: str | Path = zntrack.deps_path()

    def run(self):
        pass

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.file, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]
