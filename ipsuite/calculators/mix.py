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


def _update_is_exists(results, key, atoms_list, func, mean: bool):
    with contextlib.suppress(PropertyNotImplementedError):
        value = sum(func(x) for x in atoms_list)
        if mean and len(atoms_list) > 0:
            value /= len(atoms_list)

        if key in results:
            results[key] += value
        else:
            results[key] = value


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
        
        _update_is_exists(self.results, "energy", mean_results, lambda x: x.get_potential_energy(), True)
        _update_is_exists(self.results, "forces", mean_results, lambda x: x.get_forces(), True)
        _update_is_exists(self.results, "stress", mean_results, lambda x: x.get_stress(), True)

        _update_is_exists(self.results, "energy", sum_results, lambda x: x.get_potential_energy(), False)
        _update_is_exists(self.results, "forces", sum_results, lambda x: x.get_forces(), False)
        _update_is_exists(self.results, "stress", sum_results, lambda x: x.get_stress(), False)


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
    # weights: list = zntrack.params(None) ?

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
