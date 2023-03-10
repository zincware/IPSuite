import collections.abc
import typing

import ase
import zntrack

from ipsuite import fields


class ProcessAtoms(zntrack.Node):
    """Protocol for objects that process atoms.

    Attributes
    ----------
    data: list[ase.Atoms]
        The atoms data to process. This must be an input to the Node
    atoms: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
        It musn't be a 'field.Atoms' but can also be e.g. a 'property'.
    """

    data: list[ase.Atoms] = zntrack.zn.deps()
    atoms: list[ase.Atoms] = fields.Atoms()


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
    atoms: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
        It musn't be a 'field.Atoms' but can also be e.g. a 'property'.
        Altough we only process a single atoms object, we return a list.
        This could e.g. be the case when we want to create a trajectory
        starting from a single atoms object.
    """

    data: typing.Union[ase.Atoms, typing.List[ase.Atoms]] = zntrack.zn.deps()
    data_id: typing.Optional[int] = zntrack.zn.params(None)

    atoms: typing.List[ase.Atoms] = fields.Atoms()

    def get_data(self) -> ase.Atoms:
        """Get the atoms object to process given the 'data' and 'data_id'.

        Returns
        -------
        ase.Atoms
            The atoms object to process
        """
        if isinstance(self.data, (list, collections.abc.Sequence)):
            atoms = self.data[self.data_id if self.data_id else 0].copy()
        else:
            atoms = self.data.copy()
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

    def get_data(self):
        return self.data.data, self.data.atoms
