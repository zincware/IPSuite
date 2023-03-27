"""Load Data directly from a H5MD trajectory file."""
import uuid

import znh5md
import zntrack


class AddDataH5MD(zntrack.Node):
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
