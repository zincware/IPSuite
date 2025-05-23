import zntrack
from ase.calculators import mixing
from ase.calculators.calculator import Calculator

from ipsuite import base
from ipsuite.abc import NodeWithCalculator


class MixCalculator(base.IPSNode):
    """Combine multiple models or calculators into one.

    Attributes:
        calculators: list[NodeWithCalculator]
            List of calculators to combine.
        method: str
            choose from "mean" or "sum" to combine the calculators.
    """

    calculators: list[NodeWithCalculator] = zntrack.deps()
    method: str = zntrack.params("sum")

    def run(self) -> None:
        pass

    def get_calculator(self, **kwargs) -> Calculator:
        """Property to return a model specific ase calculator object.

        Returns
        -------
        calc:
            ase calculator object
        """
        if self.method == "mean":
            return mixing.AverageCalculator(
                [calc.get_calculator() for calc in self.calculators]
            )
        elif self.method == "sum":
            return mixing.SumCalculator(
                [calc.get_calculator() for calc in self.calculators]
            )
        else:
            raise ValueError(f"method {self.method} not supported")
