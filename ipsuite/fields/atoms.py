"""Lazy ASE Atoms loading."""
import collections.abc
import pathlib
import typing

import ase.calculators.singlepoint
import ase.db
import tqdm
import znslice
import zntrack

from ipsuite import base


class ASEAtomsFromDB(collections.abc.Sequence):
    """ASE Atoms from ASE DB loading."""

    def __init__(self, database: str, threshold: int = 100):
        """Construct ASEAtomsFromDB.

        Parameters
        ----------
        database: file
            The database to read from
        threshold: int
            Minimum number of atoms to read at once to print tqdm loading bars.
        """
        self._database = pathlib.Path(database)
        self._threshold = threshold
        self._len = None

    @znslice.znslice(lazy=True, advanced_slicing=True)
    def __getitem__(
        self, item: typing.Union[int, list]
    ) -> typing.Union[ase.Atoms, znslice.LazySequence]:
        """Get atoms.

        Parameters
        ----------
        item: int | list | slice
            The identifier of the requested atoms to return
        Returns
        -------
        Atoms | list[Atoms].
        """
        atoms = []
        single_item = isinstance(item, int)
        if single_item:
            item = [item]

        with ase.db.connect(self._database) as database:
            atoms.extend(
                database[key + 1].toatoms()
                for key in tqdm.tqdm(
                    item,
                    disable=len(item) < self._threshold,
                    ncols=120,
                    desc=f"Loading atoms from {self._database}",
                )
            )
        return atoms[0] if single_item else atoms

    def __len__(self):
        """Get the len based on the db.

        This value is cached because the db is not expected to
        change during the lifetime of this class.
        """
        if self._len is None:
            with ase.db.connect(self._database) as db:
                self._len = len(db)
        return self._len

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
        return [(instance.nwd / f"{self.name}.db").as_posix()]

    def get_stage_add_argument(self, instance: zntrack.Node) -> typing.List[tuple]:
        return [(self.dvc_option, file) for file in self.get_affected_files(instance)]

    def save(self, instance: zntrack.Node):
        """Save value with ase.db.connect."""
        atoms: base.ATOMS_LST = getattr(instance, self.name)
        instance.nwd.mkdir(exist_ok=True, parents=True)
        file = self.get_affected_files(instance)[0]

        with ase.db.connect(file, append=False) as db:
            for atom in tqdm.tqdm(atoms, desc=f"Writing atoms to {file}"):
                db.write(atom, group=instance.name)

    def get_data(self, instance: zntrack.Node) -> any:
        if all([pathlib.Path(x).exists() for x in self.get_affected_files(instance)]):
            return ASEAtomsFromDB(database=self.get_affected_files(instance)[0])[:]
        else:
            raise FileNotFoundError
