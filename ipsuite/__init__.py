"""The ipsuite package."""
import importlib.metadata

from ipsuite import (
    analysis,
    base,
    calculators,
    configuration_comparison,
    configuration_selection,
    fields,
    models,
    utils,
)
from ipsuite.data_loading.add_data_ase import AddData
from ipsuite.project import Project
from ipsuite.utils.logs import setup_logging

__all__ = [
    "base",
    "utils",
    "AddData",
    "Project",
    "configuration_comparison",
    "configuration_selection",
    "models",
    "analysis",
    "fields",
    "calculators",
]

__version__ = importlib.metadata.version(__name__)
setup_logging(__name__)
