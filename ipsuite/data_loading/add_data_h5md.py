"""Load Data directly from a H5MD trajectory file."""
import functools
import typing
import uuid

import ase
import h5py
import znh5md
import zntrack

from ipsuite import base


class AddDataH5MD(base.IPSNode):
    """Load Data directly from a H5MD trajectory file."""

    file = zntrack.dvc.deps()
    _hash = zntrack.zn.outs()
    _atoms = None

    def run(self):
        self._hash = str(uuid.uuid4())  # we must have an output

    @property
    def atoms(self) -> typing.List[ase.Atoms]:
        def file_handle(filename):
            file = self.state.fs.open(filename, "rb")
            return h5py.File(file)

        return znh5md.ASEH5MD(
            self.traj_file,
            format_handler=functools.partial(
                znh5md.FormatHandler, file_handle=file_handle
            ),
        ).get_atoms_list()
