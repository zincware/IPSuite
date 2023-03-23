import tqdm
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

        calculator = self.calc

        for atom in tqdm.tqdm(self.atoms):
            atom.calc = calculator
            atom.get_potential_energy()
            atom.get_stress()

    @property
    def calc(self):
        """Get an LJ ase calculator."""

        calculator = LennardJones()
        return calculator
