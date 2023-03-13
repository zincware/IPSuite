import logging

from ipsuite.models.base import MLModel, Prediction
from ipsuite.models.ensemble import EnsembleModel

log = logging.getLogger(__name__)

__all__ = ["Prediction", "MLModel", "EnsembleModel"]
try:
    from ipsuite.models.gap import GAP

    __all__ += ["GAP"]
except ModuleNotFoundError:
    log.warning(
        "No GAP installation was found. You can install GAP using 'pip install"
        " ipsuite[gap]'"
    )

try:
    from ipsuite.models.nequip import Nequip

    __all__ += ["Nequip"]
except ModuleNotFoundError:
    log.warning(
        "No Nequip installation was found. You can install GAP using 'pip install"
        " ipsuite[nequip]'"
    )

try:
    from ipsuite.models.mace_model import MACE

    __all__ += ["MACE"]
except ModuleNotFoundError:
    log.warning(
        "No MACE installation was found. The installation is described at"
        " 'https://github.com/ACEsuit/mace'"
    )
