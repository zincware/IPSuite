import dataclasses
import logging

log = logging.getLogger(__name__)

@dataclasses.dataclass
class TBLiteModel:
    """A model for the TBLite calculator [1]_.

    Parameters
    ----------
    method : str
        The method to use for the calculator.
    verbosity : int
        The verbosity level of the calculator.

    .. [1] https://tblite.readthedocs.io/en/latest/

    Examples
    --------
    >>> import ipsuite as ips
    >>> project = ips.Project()
    >>> tblite = ips.TBLiteModel(method="GFN2-xTB")
    >>> with project:
    ...     water = ips.Smiles2Conformers(smiles="O", numConfs=100)
    ...     box = ips.MultiPackmol(
    ...         data=[water.frames], count=[16], density=1000, n_configurations=11,
    ...     )
    ...     ips.ApplyCalculator(
    ...         data=box.frames,
    ...         model=tblite,
    ...     )
    >>> project.build()
    """
    method:  str = "GFN2-xTB"
    verbosity:  int = 0

    def get_calculator(self, **kwargs):
        """Get an xtb ase calculator."""
        try:
            from tblite.ase import TBLite
        except ImportError:
            log.warning(
                "No xtb-python installation found. "
                "See https://tblite.readthedocs.io/ for more information."
            )
            raise

        calc = TBLite(method=self.method, verbosity=self.verbosity)
        return calc
