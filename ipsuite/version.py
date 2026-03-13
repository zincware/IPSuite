try:
    from ipsuite._version import __version__, __version_tuple__  # noqa: F401
except ImportError:
    __version__ = "0.0.0+unknown"
    __version_tuple__ = (0, 0, 0, "+unknown")
