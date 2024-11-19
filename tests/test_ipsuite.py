"""Test 'IPSuite' version."""

import ipsuite as ips


def test_version():
    """Test 'IPSuite' version."""
    assert ips.__version__ == "0.2.0"


def test_node_imports():
    """Test that all nodes are imported correctly."""
    for node in ips.nodes.__all__:
        assert issubclass(getattr(ips.nodes, node), ips.base.IPSNode)
