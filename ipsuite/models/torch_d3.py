import dataclasses
from typing import Dict, Optional, Tuple

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr

try:
    import torch
    from torch import Tensor
    from torch_dftd.functions.edge_extraction import calc_edge_index
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
except ImportError as e:
    raise ImportError(
        "torch_dftd is not installed. You can install it using the"
        " extra 'pip install ipsuite[d3]' command."
    ) from e


class TorchDFTD3CalculatorNL(TorchDFTD3Calculator):
    def __init__(
        self,
        dft: Optional[Calculator] = None,
        atoms: Optional[Atoms] = None,
        damping: str = "zero",
        xc: str = "pbe",
        old: bool = False,
        device: str = "cpu",
        cutoff: float = 95.0 * Bohr,
        cnthr: float = 40.0 * Bohr,
        abc: bool = False,
        # --- torch dftd3 specific params ---
        dtype: torch.dtype = torch.float32,
        bidirectional: bool = True,
        cutoff_smoothing: str = "none",
        skin=0.5,
        **kwargs,
    ):
        self.skin = skin
        self.pbc = torch.tensor([False, False, False], device=device)
        self.Z = None
        self.pos0 = None
        self.edge_index = None
        self.S = None
        super().__init__(
            dft=dft,
            atoms=atoms,
            damping=damping,
            xc=xc,
            old=old,
            device=device,
            cutoff=cutoff,
            cnthr=cnthr,
            abc=abc,
            dtype=dtype,
            bidirectional=bidirectional,
            cutoff_smoothing=cutoff_smoothing,
            **kwargs,
        )

    def _calc_edge_index(
        self,
        pos: Tensor,
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        edge_index = calc_edge_index(
            pos,
            cell,
            pbc,
            cutoff=self.cutoff + self.skin,
            bidirectional=self.bidirectional,
        )
        return edge_index

    def _preprocess_atoms(self, atoms: Atoms) -> Dict[str, Optional[Tensor]]:
        pos = torch.tensor(atoms.get_positions(), device=self.device, dtype=self.dtype)
        Z = torch.tensor(atoms.get_atomic_numbers(), device=self.device)

        if self.pos0 is None:
            self.pos0 = torch.zeros_like(pos)
        if self.Z is None:
            self.Z = Z.clone()

        if any(atoms.pbc):
            cell: Optional[Tensor] = torch.tensor(
                atoms.get_cell(), device=self.device, dtype=self.dtype
            )
        else:
            cell = None
        pbc = torch.tensor(atoms.pbc, device=self.device)
        condition = (
            self.edge_index is None
            or torch.any(self.pbc != pbc)
            or len(self.Z) != len(Z)
            or ((self.pos0 - pos) ** 2).sum(1).max() > self.skin**2 / 4.0
        )

        if condition:
            self.edge_index, self.S = self._calc_edge_index(pos, cell, pbc)
            self.pos0 = pos
            self.pbc = pbc

        if cell is None:
            shift_pos = self.S
        else:
            shift_pos = torch.mm(self.S, cell.detach())

        input_dicts = {
            "pos": pos,
            "Z": Z,
            "cell": cell,
            "pbc": pbc,
            "edge_index": self.edge_index,
            "shift_pos": shift_pos,
        }
        return input_dicts


@dataclasses.dataclass
class TorchDFTD3:
    """Compute D3 correction terms using torch-dftd.

    Attributes
    ----------
    xc : str
    damping : str
    cutoff : float
    abc : bool
        ATM 3-body interaction
    cnthr : float
        Coordination number cutoff distance in angstrom
    dtype : str
        Data type used for the calculation.
    device : str
        Device used for the calculation. Defaults to "cuda" if available, otherwise "cpu".
    skin : float
        If > 0, switches to a D3 implementation that reuses neighborlists.
        This can significantly improve performance.
    """

    xc: str
    damping: str
    cutoff: float
    abc: bool
    cnthr: float
    dtype: str
    device: str | None = None
    skin: float = 0.0

    def get_calculator(self, **kwargs):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dtype == "float64":
            dtype = torch.float64
        elif self.dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError("dtype must be float64 or float32")

        if self.skin < 1e-5:
            calc = TorchDFTD3Calculator(
                xc=self.xc,
                damping=self.damping,
                cutoff=self.cutoff,
                abc=self.abc,
                cnthr=self.cnthr,
                dtype=dtype,
                atoms=None,
                device=self.device,
            )
        else:
            calc = TorchDFTD3CalculatorNL(
                xc=self.xc,
                damping=self.damping,
                cutoff=self.cutoff,
                abc=self.abc,
                cnthr=self.cnthr,
                dtype=dtype,
                atoms=None,
                device=self.device,
                skin=self.skin,
            )

        return calc
