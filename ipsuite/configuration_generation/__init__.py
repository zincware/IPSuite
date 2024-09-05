"""This module provides Nodes for creating molecular dynamics starting structures.
Workflows can be quickly created from just knowing the SMILES strings of the molecules within the simulation box.
The PACKMOL interface can then be used to create the actual starting configuration."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
