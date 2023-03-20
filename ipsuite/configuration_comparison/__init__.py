"""Configuration Comparison Module."""
import logging

log = logging.getLogger(__name__)

try:
    from ipsuite.configuration_comparison.base import ConfigurationComparison
    from ipsuite.configuration_comparison.MMKernel import MMKernel
    from ipsuite.configuration_comparison.REMatch import REMatch

    __all__ = ["MMKernel", "ConfigurationComparison", "REMatch"]
except ImportError:
    log.warning(
        "Using configuration comparison requires 'pip install ipsuite[comparison]'"
    )
