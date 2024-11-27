import dataclasses

import tqdm
from ase.calculators.calculator import all_changes
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

from ipsuite import base


@dataclasses.dataclass
class LJSinglePoint:
    """This is a testing Node!
    It uses ASE'S Lennard-Jones calculator with default arguments.
    The calculator accept all elements and implements energy, forces and stress,
    making it very useful for creating dummy data.
    """

    epsilon: float = 1.0
    sigma: float = 1.0
    rc: float = 10.0

    def get_calculator(self, **kwargs):
        """Get an LJ ase calculator."""

        return LennardJones()
@dataclasses.dataclass
class EMTSinglePoint:
    """This is a testing Node!
    It uses ASE'S EMT calculator with default arguments.
    The calculator accept all elements and implements energy, forces,
    making it very useful for creating dummy data.
    """

    def get_calculator(self, **kwargs):
        """Get an EMT ase calculator."""
        return EMTCalculator()


class EMTCalculator(EMT):
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy"]
        elif "stress" not in properties and atoms.cell.rank == 3:
            properties.append("stress")
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )
