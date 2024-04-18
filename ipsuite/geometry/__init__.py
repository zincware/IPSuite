"""This module contains Nodes and utilities for molecule mapping.
The Nodes provide `forward_mapping` and `backward_mapping` methods
for applying and reversing the transformations.
Note that they do not implement `run` methods and are thus intended
to be used via `zntrack.deps` only.
"""

from ipsuite.geometry.mapping import BarycenterMapping

__all__ = ["BarycenterMapping"]
