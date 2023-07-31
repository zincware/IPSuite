"""Load Data directly from a H5MD trajectory file."""
import functools
import typing

import ase
import h5py
import znh5md
import zntrack

from ipsuite import base


class AddDataH5MD(base.IPSNode):
    """Load Data directly from a H5MD trajectory file."""

    file = zntrack.dvc.deps()
    _atoms = None

    def run(self):
        pass

    @property
    def atoms(self) -> typing.List[ase.Atoms]:
        def file_handle(filename):
            file = self.state.fs.open(filename, "rb")
            return h5py.File(file)

        return znh5md.ASEH5MD(
            self.file,
            format_handler=functools.partial(
                znh5md.FormatHandler, file_handle=file_handle
            ),
        ).get_atoms_list()
