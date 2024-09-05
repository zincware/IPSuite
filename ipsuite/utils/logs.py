"""Logging Setup for the ipsuite package."""

import logging
import sys


def setup_logging(name) -> None:
    """Configure logging for the ipsuite package."""
    logger = logging.getLogger(name)

    # Set the default log level here
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s (%(levelname)s): %(message)s")

    channel = logging.StreamHandler(sys.stdout)
    channel.setLevel(logging.DEBUG)
    channel.setFormatter(formatter)

    logger.addHandler(channel)

    logger.debug("Welcome to IPS - the Interatomic Potential Suite!")
