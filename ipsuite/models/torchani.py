import ase.calculators.calculator
import torchani
import zntrack

from ipsuite.models import MLModel

_models = {
    "ANI2x": torchani.models.ANI2x,
    "ANI1ccx": torchani.models.ANI1ccx,
    "ANI1x": torchani.models.ANI1x,
}


class TorchAni(MLModel):
    """TorchAni model implementation."""

    model_name: str = zntrack.zn.params("ANI2x")

    def run(self) -> None:
        pass

    def get_calculator(self, **kwargs) -> ase.calculators.calculator.Calculator:
        return _models[self.model_name].ase()
