"""Load Data directly from a H5MD trajectory file."""
import uuid

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
    def atoms(self):
        if self._atoms is None:
            self._atoms = znh5md.ASEH5MD(self.file).get_atoms_list()
        return self._atoms
