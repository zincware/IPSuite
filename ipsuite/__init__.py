"""The ipsuite package."""

import lazy_loader as lazy
from ipsuite.utils.logs import setup_logging
from ipsuite.utils.helpers import setup_ase

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

setup_logging(__name__)
setup_ase()
