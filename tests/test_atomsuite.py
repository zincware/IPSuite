"""Test 'IPSuite' version."""
from ipsuite import __version__


def test_version():
    """Test 'IPSuite' version."""
    assert __version__ == "0.1.0a2"
