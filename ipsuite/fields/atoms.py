"""Lazy ASE Atoms loading."""

import typing

import h5py
import znh5md
import zntrack

from ipsuite import base


class Atoms(zntrack.Field):
    """Store list[ase.Atoms] in an ASE database."""

    dvc_option = "--outs"
    group = zntrack.FieldGroup.RESULT

    def __init__(self):
        super().__init__(use_repr=False)

    def get_files(self, instance: zntrack.Node) -> list:
        return [(instance.nwd / f"{self.name}.h5").as_posix()]

    def get_stage_add_argument(self, instance: zntrack.Node) -> typing.List[tuple]:
        return [(self.dvc_option, file) for file in self.get_files(instance)]

    def save(self, instance: zntrack.Node):
        """Save value with ase.db.connect."""
        atoms: base.ATOMS_LST = getattr(instance, self.name)
        instance.nwd.mkdir(exist_ok=True, parents=True)
        file = self.get_files(instance)[0]

        db = znh5md.IO(filename=file)
        db.extend(atoms)

    def get_data(self, instance: zntrack.Node) -> base.protocol.ATOMS_LST:
        """Get data from znh5md File."""
        file_name = self.get_files(instance)[0]

        with instance.state.fs.open(file_name, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]
