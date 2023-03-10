import logging

from ipsuite.models.base import MLModel, Prediction
from ipsuite.models.ensemble import EnsembleModel
from ipsuite.models.gap import GAP
from ipsuite.models.nequip import Nequip

log = logging.getLogger(__name__)

__all__ = ["Prediction", "MLModel", "GAP", "Nequip", "EnsembleModel"]
try:
    from ipsuite.models.mace_model import MACE

    __all__ += ["MACE"]
except ImportError:
    log.warning(
        "The MACE doesn't seem to be installed. If you want to use MACE, please instal"
        " it. The installation of MACE is described here"
        " 'https://github.com/ACEsuit/mace' ."
    )
