import abc
import dataclasses
import typing

import ase
import zntrack

from ipsuite import fields


class IPSNode(zntrack.Node):
    """Base class for all IPSuite nodes."""


class ProcessAtoms(IPSNode):
    """Protocol for objects that process atoms.

    Attributes
    ----------
    data: list[ase.Atoms]
        The atoms data to process. This must be an input to the Node
    frames: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
        It does not have to be 'field.Atoms' but can also be e.g. a 'property'.
    """

    data: list[ase.Atoms] = zntrack.deps()
    frames: list[ase.Atoms] = fields.Atoms()

    def get_data(self) -> list[ase.Atoms]:
        """Get the atoms data to process."""
        if self.data is not None:
            return self.data
        else:
            raise ValueError("No data given.")


class ProcessSingleAtom(IPSNode):
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
    frames: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
        It does not have to be 'field.Atoms' but can also be e.g. a 'property'.
        Although, we only process a single atoms object, we return a list.
        This could e.g. be the case when we want to create a trajectory
        starting from a single atoms object.
    """

    data: typing.List[ase.Atoms] = zntrack.deps()
    data_id: int = zntrack.params(-1)

    frames: typing.List[ase.Atoms] = fields.Atoms()

    def get_data(self) -> ase.Atoms:
        """Get the atoms object to process given the 'data' and 'data_id'.

        Returns
        -------
        ase.Atoms
            The atoms object to process
        """
        return self.data[self.data_id]


class AnalyseAtoms(IPSNode):
    """Protocol for objects that analyse atoms.

    Attributes
    ----------
    data: list[ase.Atoms]
        The atoms data to analyse. This must be an input to the Node
    """

    data: list[ase.Atoms] = zntrack.deps()


class ComparePredictions(IPSNode):
    """Compare the predictions of two models."""

    x: list[ase.Atoms] = zntrack.deps()
    y: list[ase.Atoms] = zntrack.deps()


@dataclasses.dataclass
class Check:
    """Base class for check nodes.
    These are callbacks that can be used to preemptively terminate
    a molecular dynamics simulation if a vertain condition is met.
    """

    status: str = None

    def initialize(self, atoms: ase.Atoms) -> None:
        """Stores some reference property to compare the current property
        against and see whether the simulation should be stopped.
        Derived classes do not need to override this if they consider
        absolute values and not comparisons.
        """
        self.status = False
        pass

    @abc.abstractmethod
    def check(self, atoms: ase.Atoms) -> bool:
        """Method to check whether a simulation should be stopped."""
        ...

    @abc.abstractmethod
    def get_value(self, atoms: ase.Atoms):
        """Returns the metric that is tracked for stopping."""
        ...

    @abc.abstractmethod
    def get_quantity(self) -> str: ...

    @abc.abstractmethod
    def mod_atoms(self, atoms: ase.Atoms):
        """Returns the metric that is tracked for stopping."""
        ...
    
    def __str__(self):
        return self.status


class Flatten(ProcessAtoms):
    """Flattens list[list[ase.Atoms]] to list[ase.Atoms]"""

    def run(self):
        self.frames = sum(self.data, [])
