import dataclasses
from pathlib import Path

from ase.calculators.calculator import Calculator, all_changes

from ipsuite.interfaces import NodeWithCalculator

@dataclasses.dataclass
class VASPModel(NodeWithCalculator):
    """
    Docstring for VASPModel
    """
    xc: str = "hse06"
    hfscreen: float = 0.4
#    config: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, directory: str | Path, **kwargs) -> Calculator:
        from ase.calculators.vasp import Vasp

        return Vasp(xc='hse06', hfscreen=0.4)
        raise NotImplementedError()