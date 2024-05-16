import abc
import collections.abc
import pathlib
import typing

import ase
import tqdm
import znflow
import zntrack

from ipsuite import fields

# TODO raise error if both data and data_file are given


class IPSNode(zntrack.Node):
    _module_ = "ipsuite.nodes"


class ProcessAtoms(IPSNode):
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
        It does not have to be 'field.Atoms' but can also be e.g. a 'property'.
    """

    data: list[ase.Atoms] = zntrack.deps()
    data_file: str = zntrack.dvc.deps(None)
    atoms: list[ase.Atoms] = fields.Atoms()

    def _post_init_(self):
        if self.data is not None:
            self.data = znflow.combine(self.data, attribute="atoms")

    def update_data(self):
        """Update the data attribute."""
        if self.data is None:
            self.data = self.get_data()

    def get_data(self) -> list[ase.Atoms]:
        """Get the atoms data to process."""
        if self.data is not None:
            return self.data
        elif self.data_file is not None:
            try:
                with self.state.fs.open(pathlib.Path(self.data_file).as_posix()) as f:
                    return list(ase.io.iread(f))
            except FileNotFoundError:
                # File can not be opened with DVCFileSystem, try normal open
                return list(ase.io.iread(self.data_file))
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
    data_file: str | None
        The path to the file containing the atoms data. This is an
        alternative to 'data' and can be used to load the data from
        a file. If both are given, 'data' is used. Set 'data' to None
        if you want to use 'data_file'.
    atoms: list[ase.Atoms]
        The processed atoms data. This is an output of the Node.
        It does not have to be 'field.Atoms' but can also be e.g. a 'property'.
        Although, we only process a single atoms object, we return a list.
        This could e.g. be the case when we want to create a trajectory
        starting from a single atoms object.
    """

    data: typing.Union[ase.Atoms, typing.List[ase.Atoms]] = zntrack.deps()
    data_file: str = zntrack.deps_path(None)
    data_id: typing.Optional[int] = zntrack.params(0)

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
            try:
                with self.state.fs.open(pathlib.Path(self.data_file).as_posix()) as f:
                    atoms = list(ase.io.iread(f))[self.data_id]
            except FileNotFoundError:
                # File can not be opened with DVCFileSystem, try normal open
                atoms = list(ase.io.iread(self.data_file))[self.data_id]
        else:
            raise ValueError("No data given.")
        return atoms


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


class Mapping(ProcessAtoms):
    """Base class for transforming ASE `Atoms`.

    `Mapping` nodes can be used in a more functional manner when initialized
    with `data=None` outside the project graph.
    In that case, one can use the mapping methods but the Node itself does not store
    the transformed configurations.

    Attributes
    ----------
    molecules: list[ase.Atoms]
        A flat list of all molecules in the system.

    Parameters
    ----------
    frozen: bool
        If True, the neighbor list is only constructed for the first configuration.
        The indices of the molecules will be frozen for all configurations.
    """

    molecules: list[ase.Atoms] = zntrack.outs()
    frozen: bool = zntrack.params(False)

    # TODO, should we allow to transfer the frozen mapping to another node?
    #  mapping = Mapping(frozen=True, reference=mapping)

    def run(self):
        self.atoms = []
        self.molecules = []
        for atoms in tqdm.tqdm(self.get_data(), ncols=70):
            cg_atoms, molecules = self.forward_mapping(atoms)
            self.atoms.append(cg_atoms)
            self.molecules.extend(molecules)

    def get_molecules_per_configuration(self) -> typing.List[typing.List[ase.Atoms]]:
        """Get a list of lists of molecules per configuration."""
        molecules_per_configuration = []
        start = 0
        for atoms in self.atoms:
            molecules_per_configuration.append(self.molecules[start : start + len(atoms)])
            start += len(atoms)
        return molecules_per_configuration

    def forward_mapping(
        self, atoms: ase.Atoms
    ) -> typing.Tuple[ase.Atoms, list[ase.Atoms]]:
        raise NotImplementedError

    def backward_mapping(
        self, cg_atoms: ase.Atoms, molecules: list[ase.Atoms]
    ) -> list[ase.Atoms]:
        raise NotImplementedError


class Check(IPSNode):
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

    def __str__(self):
        return self.status


class Modifier(IPSNode):
    """Base class for modifier nodes.
    These are callbacks that can be used to alter the dynamics of an MD run.
    This can be achieved by modifying the thermostat state or atoms in the system.
    """

    @abc.abstractmethod
    def modify(self, thermostat: IPSNode, step: int, total_steps: int) -> None: ...


class Flatten(ProcessAtoms):
    """Flattens list[list[ase.Atoms]] to list[ase.Atoms]"""

    def run(self):
        atoms = sum(self.data, [])
        self.atoms = atoms
