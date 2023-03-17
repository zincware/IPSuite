import collections.abc
import typing

import ase
import zntrack

from ipsuite import fields, utils

# TODO raise error if both data and data_file are given


class ProcessAtoms(zntrack.Node):
    """Protocol for objects that process atoms.

    Attributes
    ----------
    data: list[ase.Atoms]
        The atoms data to process. This must be an input to the Node
    data_file: str | None
        The path to the file containing the atoms data. This is an
        alternative to 'data' and can be used to load the data from
        a file. If both are given, 'data' is used. Set 'data' to None
        if you want to use 'data_file'.
    atoms: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
        It musn't be a 'field.Atoms' but can also be e.g. a 'property'.
    """

    data: list[ase.Atoms] = zntrack.zn.deps()
    data_file: str = zntrack.dvc.deps(None)
    atoms: list[ase.Atoms] = fields.Atoms()

    def _post_init_(self):
        if self.data is not None:
            self.data = utils.helpers.get_deps_if_node(self.data, "atoms")

    def update_data(self):
        """Update the data attribute."""
        if self.data is None:
            self.data = self.get_data()

    def get_data(self) -> list[ase.Atoms]:
        """Get the atoms data to process."""
        if self.data is not None:
            return self.data
        elif self.data_file is not None:
            return list(ase.io.iread(self.data_file))
        else:
            raise ValueError("No data given.")


class ProcessSingleAtom(zntrack.Node):
    """Protocol for objects that process a single atom.

    Attributes
    ----------
    data: ase.Atoms | list[ase.Atoms]
        The atoms data to process. This must be an input to the Node.
        It can either a single atoms object or a list of atoms objects
        with a given 'data_id'.
    data_id: int | None
        The id of the atoms object to process. If None, the first
        atoms object is used. Only relevant if 'data' is a list.
    data_file: str | None
        The path to the file containing the atoms data. This is an
        alternative to 'data' and can be used to load the data from
        a file. If both are given, 'data' is used. Set 'data' to None
        if you want to use 'data_file'.
    atoms: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
        It musn't be a 'field.Atoms' but can also be e.g. a 'property'.
        Altough we only process a single atoms object, we return a list.
        This could e.g. be the case when we want to create a trajectory
        starting from a single atoms object.
    """

    data: typing.Union[ase.Atoms, typing.List[ase.Atoms]] = zntrack.zn.deps()
    data_file: str = zntrack.dvc.deps(None)
    data_id: typing.Optional[int] = zntrack.zn.params(0)

    atoms: typing.List[ase.Atoms] = fields.Atoms()

    def get_data(self) -> ase.Atoms:
        """Get the atoms object to process given the 'data' and 'data_id'.

        Returns
        -------
        ase.Atoms
            The atoms object to process
        """
        if self.data is not None:
            if isinstance(self.data, (list, collections.abc.Sequence)):
                atoms = self.data[self.data_id].copy()
            else:
                atoms = self.data.copy()
        elif self.data_file is not None:
            atoms = list(ase.io.iread(self.data_file))[self.data_id]
        else:
            raise ValueError("No data given.")
        return atoms


class AnalyseAtoms(zntrack.Node):
    """Protocol for objects that analyse atoms.

    Attributes
    ----------
    data: list[ase.Atoms]
        The atoms data to analyse. This must be an input to the Node
    """

    data: list[ase.Atoms] = zntrack.zn.deps()


class AnalyseProcessAtoms(zntrack.Node):
    """Analyse the output of a ProcessAtoms Node."""

    data: ProcessAtoms = zntrack.zn.deps()

    def get_data(self) -> typing.Tuple[list[ase.Atoms], list[ase.Atoms]]:
        self.data.update_data()  # otherwise, data might not be available
        return self.data.data, self.data.atoms
