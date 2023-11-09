"""The IPSuite project module."""

import logging

from zntrack import Project

log = logging.getLogger(__name__)


class Project(Project):
    """Project class for MLSuite interfacing."""
