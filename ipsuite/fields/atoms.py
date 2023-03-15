"""Lazy ASE Atoms loading."""
import collections.abc
import functools
import typing

import ase.calculators.singlepoint
import ase.db
import h5py
import znh5md
import znslice
import zntrack

from ipsuite import base


class ASEAtomsFromDB(collections.abc.Sequence):
    """ASE Atoms from ASE DB loading."""

    def __init__(self, database: znh5md.ASEH5MD, threshold: int = 100):
        """Construct ASEAtomsFromDB.

        Parameters
        ----------
        database: file
            The database to read from
        threshold: int
            Minimum number of atoms to read at once to print tqdm loading bars.
        """
        self.database = database
        self.threshold = threshold
        self._len = None

    @znslice.znslice(lazy=True, advanced_slicing=True)
    def __getitem__(
        self, item: typing.Union[int, list]
    ) -> typing.Union[ase.Atoms, znslice.LazySequence]:
        """Get Atoms from the database."""
        return self.database[item]

    def __len__(self):
        """Get the len based on the db.

        This value is cached because the db is not expected to
        change during the lifetime of this class.
        """
        return len(self.database.position)

    def __repr__(self):
        """Repr."""
        db_short = "/".join(self._database.parts[-3:])
        return f"{self.__class__.__name__}(db='{db_short}')"


class Atoms(zntrack.Field):
    """Store list[ase.Atoms] in an ASE database."""

    dvc_option = "--outs"
    group = zntrack.FieldGroup.RESULT

    def __init__(self):
        super().__init__(use_repr=False)

    def get_affected_files(self, instance: zntrack.Node) -> list:
        # TODO: remove on new ZnTrack release
        return self.get_files(instance)

    def get_files(self, instance: zntrack.Node) -> list:
        return [(instance.nwd / f"{self.name}.h5").as_posix()]

    def get_stage_add_argument(self, instance: zntrack.Node) -> typing.List[tuple]:
        return [(self.dvc_option, file) for file in self.get_affected_files(instance)]

    def save(self, instance: zntrack.Node):
        """Save value with ase.db.connect."""
        atoms: base.ATOMS_LST = getattr(instance, self.name)
        instance.nwd.mkdir(exist_ok=True, parents=True)
        file = self.get_affected_files(instance)[0]

        db = znh5md.io.DataWriter(filename=file)
        db.initialize_database_groups()
        db.add(znh5md.io.AtomsReader(atoms))

    def get_data(self, instance: zntrack.Node) -> any:
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
        return ASEAtomsFromDB(database=data)[:]
