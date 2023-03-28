"""Lazy ASE Atoms loading."""
import functools
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

        db = znh5md.io.DataWriter(filename=file)
        db.initialize_database_groups()
        db.add(znh5md.io.AtomsReader(atoms, frames_per_chunk=100000, use_pbc_group=True))

    def get_data(self, instance: zntrack.Node) -> base.protocol.ATOMS_LST:
        """Get data from znh5md File."""
        file = self.get_files(instance)[0]

        def file_handle(filename):
            file = instance.state.get_file_system().open(filename, "rb")
            return h5py.File(file)

        data = znh5md.ASEH5MD(
            file,
            format_handler=functools.partial(
                znh5md.FormatHandler, file_handle=file_handle
            ),
        )
        return data[:]
        # if instance.state.rev is None and instance.state.remote is None::
        #     # it is slightly faster
        #     return znh5md.ASEH5MD(file)[:]
