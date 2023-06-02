import tqdm
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

from ipsuite import base


class LJSinglePoint(base.ProcessAtoms):
    """This is a testing Node!
    It uses ASE'S Lennard-Jones calculator with default arguments.
    The calculator accept all elements and implements energy, forces and stress,
    making it very useful for creating dummy data.
    """

    def run(self):
        self.atoms = self.get_data()

        calculator = self.get_calculator()

        for atom in tqdm.tqdm(self.atoms):
            atom.calc = calculator
            atom.get_potential_energy()
            atom.get_stress()

    def get_calculator(self, **kwargs):
        """Get an LJ ase calculator."""

        return LennardJones()


class EMTSinglePoint(base.ProcessAtoms):
    """This is a testing Node!
    It uses ASE'S EMT calculator with default arguments.
    The calculator accept all elements and implements energy, forces,
    making it very useful for creating dummy data.
    """

    def run(self):
        self.atoms = self.get_data()

        calculator = self.get_calculator()

        for atom in tqdm.tqdm(self.atoms):
            atom.calc = calculator
            atom.get_potential_energy()

    def get_calculator(self, **kwargs):
        """Get an EMT ase calculator."""
        return EMT()
