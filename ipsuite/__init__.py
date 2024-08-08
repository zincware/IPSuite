"""The ipsuite package."""

import importlib.metadata

from ipsuite.utils.logs import setup_logging
from ipsuite.utils.helpers import setup_ase

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

__version__ = importlib.metadata.version(__name__)
setup_logging(__name__)
setup_ase()
