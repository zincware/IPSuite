import dataclasses
import logging
from pathlib import Path
from typing import Union, Optional, Literal
from ase import units
import zntrack

log = logging.getLogger(__name__)


@dataclasses.dataclass
class MACEMPModel:
    """Interface for the MACE model.
    
    For more information, see:
    - https://github.com/ACEsuit/mace
    """
    model: Optional[Union[str, Path]] = None
    device: str = ""
    default_dtype: str = "float32"
    dispersion: bool = False
    damping: Literal["zero", "bj", "zerom", "bjm"] = "bj"
    dispersion_xc: str = "pbe"
    dispersion_cutoff: float = 40.0 * units.Bohr

    model_path: Optional[Path] = zntrack.deps_path(None)
    

    def get_calculator(self, **kwargs):
        """Get an xtb ase calculator."""
        try:
            from mace.calculators import mace_mp
        except ImportError:
            log.warning(
                "No `mace-torch` installation found. "
                "See https://github.com/ACEsuit/mace for more information."
            )
            raise

        return mace_mp(
            model=self.model,
            device=self.device,
            default_dtype=self.default_dtype,
            dispersion=self.dispersion,
            damping=self.damping,
            dispersion_xc=self.dispersion_xc,
            dispersion_cutoff=self.dispersion_cutoff,
        )
