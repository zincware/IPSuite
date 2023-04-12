import abc


class LogPathCalculator:
    """Abstract base class for calculators that produce output files."""

    @property
    @abc.abstractmethod
    def log_path(self):
        """Define the path to the log file.

        Some calculators produce log files.
        E.g. DFT calculators, such as CP2k store the SCF convergence
        in a log file.
        """
