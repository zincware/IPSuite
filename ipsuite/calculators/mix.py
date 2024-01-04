import contextlib
import typing

import tqdm
import zntrack
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from ipsuite import base
from ipsuite.utils.ase_sim import freeze_copy_atoms


class _MixCalculator(Calculator):
    def __init__(self, calculators: typing.List[Calculator], methods: list, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.calculators = calculators
        self.implemented_properties = self.calculators[0].implemented_properties
        self.methods = methods

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        mean_results = []
        sum_results = []

        for i, calc in enumerate(self.calculators):
            _atoms = atoms.copy()
            _atoms.calc = calc
            if self.methods[i] == "mean":
                mean_results.append(_atoms)
            elif self.methods[i] == "sum":
                sum_results.append(_atoms)
            else:
                raise NotImplementedError

        for atoms in mean_results:
            if "energy" in self.results:
                self.results["energy"] += atoms.get_potential_energy()
            else:
                self.results["energy"] = atoms.get_potential_energy()

            if "forces" in self.results:
                self.results["forces"] += atoms.get_forces()
            else:
                self.results["forces"] = atoms.get_forces()

            with contextlib.suppress(PropertyNotImplementedError):
                if "stress" in self.results:
                    self.results["stress"] += atoms.get_stress()
                else:
                    self.results["stress"] = atoms.get_stress()

        if "energy" in self.results:
            self.results["energy"] /= len(mean_results)
        if "forces" in self.results:
            self.results["forces"] /= len(mean_results)
        if "stress" in self.results:
            self.results["stress"] /= len(mean_results)

        for atoms in sum_results:
            if "energy" in self.results:
                self.results["energy"] += atoms.get_potential_energy()
            else:
                self.results["energy"] = atoms.get_potential_energy()

            if "forces" in self.results:
                self.results["forces"] += atoms.get_forces()
            else:
                self.results["forces"] = atoms.get_forces()

            with contextlib.suppress(PropertyNotImplementedError):
                if "stress" in self.results:
                    self.results["stress"] += atoms.get_stress()
                else:
                    self.results["stress"] = atoms.get_stress()


class CalculatorNode(typing.Protocol):
    def get_calculator(self) -> typing.Type[Calculator]: ...


class MixCalculator(base.ProcessAtoms):
    """Combine multiple models or calculators into one.

    Attributes:
        calculators: list[CalculatorNode]
            List of calculators to combine.
        methods: str|list[str]
            choose from ['mean', 'sum'] either for all calculators
            as a string or for each calculator individually as a list.
            All calculators that are assigned with 'mean' will be
            computed first, then the calculators assigned with 'sum'
            will be added.
    """

    calculators: typing.List[CalculatorNode] = zntrack.deps()
    methods: str | typing.List[str] = zntrack.params("sum")

    def run(self) -> None:
        calc = self.get_calculator()
        self.atoms = []
        for atoms in tqdm.tqdm(self.get_data(), ncols=70):
            atoms.calc = calc
            atoms.get_potential_energy()
            self.atoms.append(freeze_copy_atoms(atoms))

    def get_calculator(self, **kwargs) -> Calculator:
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """
        if isinstance(self.methods, str):
            methods = [self.methods] * len(self.calculators)
        else:
            methods = self.methods
        return _MixCalculator(
            calculators=[x.get_calculator(**kwargs) for x in self.calculators],
            methods=methods,
        )
