"""Configuration Comparison Module."""
try:
    from ipsuite.configuration_comparison.base import ConfigurationComparison
    from ipsuite.configuration_comparison.MMKernel import MMKernel
    from ipsuite.configuration_comparison.REMatch import REMatch

    __all__ = ["MMKernel", "ConfigurationComparison", "REMatch"]
except ImportError as err:
    raise ImportError(
        "Using configuration comparison requires 'pip install ipsuite[comparison]'"
    ) from err
