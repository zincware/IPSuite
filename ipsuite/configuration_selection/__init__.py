"""The configuration selection submodule contains Nodes for splitting datasets.
A simple 80:20 random split can be achieved with a single Node, but multiple Nodes can be chained to create complex splittings.."""

from ipsuite.configuration_selection.base import ConfigurationSelection
from ipsuite.configuration_selection.filter import FilterOutlier
from ipsuite.configuration_selection.index import IndexSelection
from ipsuite.configuration_selection.kernel import KernelSelection
from ipsuite.configuration_selection.random import RandomSelection
from ipsuite.configuration_selection.split import SplitSelection
from ipsuite.configuration_selection.threshold import ThresholdSelection
from ipsuite.configuration_selection.uniform_arange import UniformArangeSelection
from ipsuite.configuration_selection.uniform_energetic import UniformEnergeticSelection
from ipsuite.configuration_selection.uniform_temporal import UniformTemporalSelection

__all__ = [
    "ConfigurationSelection",
    "RandomSelection",
    "UniformEnergeticSelection",
    "UniformTemporalSelection",
    "UniformArangeSelection",
    "KernelSelection",
    "IndexSelection",
    "ThresholdSelection",
    "SplitSelection",
    "FilterOutlier",
]
