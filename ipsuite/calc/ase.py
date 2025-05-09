import dataclasses
import importlib
import typing as t
from pathlib import Path

import zntrack
from ase.calculators.calculator import Calculator


class Device:
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"

    @staticmethod
    def resolve_auto() -> t.Literal["cpu", "cuda"]:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


# TODO: add files as dependencies somehow!


@dataclasses.dataclass
class GenericASECalculator:
    """Generic ASE calculator.

    Load any ASE calculator from a module and class name.

    Parameters
    ----------
    module : str
        Module name containing the calculator class.
        For LJ this would be 'ase.calculators.lj'.
    class_name : str
        Class name of the calculator.
        For LJ this would be 'LennardJones'.
    kwargs : dict, default=None
        Additional keyword arguments to pass to the calculator.
        For LJ this could be {'epsilon': 1.0, 'sigma': 1.0}.
    parameter_paths : str, Path, list[str|Path], default=None
        Path to configuration files for the calculator, e.g. `cp2k.yaml`.
    file_paths : str, Path, list[str|Path], default=None
        Path to files needed by the calculator, e.g. `GTH_BASIS_SETS`.
    """

    module: str
    class_name: str
    kwargs: dict[str, t.Any] | None = None
    device: t.Literal["auto", "cpu", "cuda"] | None = None
    parameter_paths: str | Path | list[str | Path] | None = zntrack.params_path(None)
    file_paths: str | Path | list[str | Path] | None = zntrack.deps_path(None)

    def get_calculator(self, **kwargs) -> Calculator:
        if self.kwargs is not None:
            kwargs.update(self.kwargs)
        module = importlib.import_module(self.module)
        cls = getattr(module, self.class_name)
        if self.device is None:
            return cls(**kwargs)
        elif self.device == "auto":
            return cls(**kwargs, device=Device.resolve_auto())
        else:
            return cls(**kwargs, device=self.device)

    @property
    def available(self) -> bool:
        try:
            importlib.import_module(self.module)
            return True
        except ImportError:
            return False
