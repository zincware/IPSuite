import typing
from uuid import uuid4

import ase
import numpy as np
import zntrack
from ase.calculators.calculator import Calculator, all_changes
from tqdm import tqdm

from ipsuite import base
from ipsuite.models.base import MLModel
from ipsuite.utils.ase_sim import freeze_copy_atoms


class EnsembleCalculator(Calculator):

    def __init__(self, calculators: typing.List[Calculator], **kwargs):
        Calculator.__init__(self, **kwargs)
        self.calculators = calculators
        self.implemented_properties = self.calculators[0].implemented_properties
    
    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        results = []
        for calc in self.calculators:
            _atoms = atoms.copy()
            _atoms.calc = calc
            results.append(_atoms)

        self.results["energy"] = np.mean([x.get_potential_energy() for x in results])
        self.results["forces"] = np.mean([x.get_forces() for x in results], axis=0)
        self.results["energy_uncertainty"] = np.std(
            [x.get_potential_energy() for x in results]
        )
        self.results["forces_uncertainty"] = np.std(
            [x.get_forces() for x in results], axis=0
        )

        if "stress" in self.implemented_properties:
            self.results["stress"] = np.mean([x.get_stress() for x in results], axis=0)
            self.results["stress_uncertainty"] = np.std(
                [x.get_stress() for x in results], axis=0
            )


class EnsembleModel(base.IPSNode):
    models: typing.List[MLModel] = zntrack.zn.deps()

    uuid = zntrack.zn.outs()  # to connect this Node to other Nodes it requires an output.

    def run(self) -> None:
        self.uuid = str(uuid4())

    def get_calculator(self, **kwargs) -> ase.calculators.calculator.Calculator:
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """
        return EnsembleCalculator(
            calculators=[x.get_calculator(**kwargs) for x in self.models]
        )

    def predict(self, atoms_list: typing.List[ase.Atoms]) -> typing.List[ase.Atoms]:
        """Predict energy, forces and stresses.

        based on what was used to train for given atoms objects.

        Parameters
        ----------
        atoms_list: typing.List[ase.Atoms]
            list of atoms objects to predict on

        Returns
        -------
        typing.List[ase.Atoms]
            Atoms with updated calculators
        """
        calc = self.get_calculator()
        result = []
        for atoms in tqdm(atoms_list, ncols=120):
            atoms.calc = calc
            atoms.get_potential_energy()
            result.append(freeze_copy_atoms(atoms))
        return result
