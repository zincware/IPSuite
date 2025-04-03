"""Test 'IPSuite' version."""

import dataclasses

import ipsuite as ips


def test_node_imports():
    """Test that all nodes are imported correctly."""
    for node in ips.__all__:
        if node in ["__version__", "Project", "base"]:
            continue
        subclass = issubclass(getattr(ips, node), ips.base.IPSNode)
        dataclass = dataclasses.is_dataclass(getattr(ips, node))
        assert subclass or dataclass
