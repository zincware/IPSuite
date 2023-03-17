import typing
from uuid import uuid4

import ase
import numpy as np
import zntrack
from ase.calculators.calculator import Calculator, all_changes
from tqdm import tqdm

from ipsuite.models.base import MLModel, Prediction


class EnsembleCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, calculators: typing.List[Calculator], **kwargs):
        Calculator.__init__(self, **kwargs)
        self.calculators = calculators

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


class EnsembleModel(zntrack.Node):
    models: typing.List[MLModel] = zntrack.zn.deps()

    uuid = zntrack.zn.outs()  # to connect this Node to other Nodes it requires an output.

    def run(self) -> None:
        self.uuid = str(uuid4())

    @property
    def calc(self) -> ase.calculators.calculator.Calculator:
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """
        return EnsembleCalculator(calculators=[x.calc for x in self.models])

    def predict(self, atoms: typing.List[ase.Atoms]) -> Prediction:
        """Predict energy, forces and stresses.

        based on what was used to train for given atoms objects.

        Parameters
        ----------
        atoms: typing.List[ase.Atoms]
            list of atoms objects to predict on

        Returns
        -------
        Prediction: Prediction
            dataclass which contains the predicted energy, forces and stresses
        """
        potential = self.calc
        validation_energy = []
        validation_forces = []
        for configuration in tqdm(atoms, ncols=70):
            configuration.calc = potential
            if all(x.use_energy for x in self.models):
                validation_energy.append(configuration.get_potential_energy())
            if all(x.use_forces for x in self.models):
                validation_forces.append(configuration.get_forces())

        return Prediction(
            energy=np.array(validation_energy),
            forces=np.array(validation_forces),
            n_atoms=len(atoms[0]),
        )
