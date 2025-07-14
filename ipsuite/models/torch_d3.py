import dataclasses
from typing import Dict, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Bohr

try:
    import torch
    from torch import Tensor
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    from vesin import NeighborList
except ImportError as e:
    raise ImportError(
        "torch_dftd and vesin are required. You can install them using the"
        " extra 'pip install ipsuite[d3]' command."
    ) from e


def calc_neighbor_by_vesin(
    pos: Tensor, cell: Optional[Tensor], pbc: Tensor, cutoff: float
) -> Tuple[Tensor, Tensor]:
    """Calculate neighbors using vesin for better performance."""
    # Convert to numpy for vesin
    pos_np = pos.detach().cpu().numpy()

    # Create vesin neighbor list calculator
    nl_calc = NeighborList(cutoff=cutoff, full_list=True)

    # Check if we have periodic boundary conditions
    if cell is not None and torch.any(pbc):
        cell_np = cell.detach().cpu().numpy()
        # Compute neighbor list with periodic boundaries
        idx_i, idx_j, S = nl_calc.compute(
            points=pos_np, box=cell_np, periodic=True, quantities="ijS"
        )
    else:
        # Non-periodic case - provide empty box and False for periodic
        idx_i, idx_j, S = nl_calc.compute(
            points=pos_np, box=np.zeros((3, 3)), periodic=False, quantities="ijS"
        )

    # Convert back to tensors
    edge_index = torch.tensor(np.stack([idx_i, idx_j], axis=0), device=pos.device)
    S = torch.tensor(S, dtype=pos.dtype, device=pos.device)
    return edge_index, S


def calc_edge_index(
    pos: Tensor,
    cell: Optional[Tensor] = None,
    pbc: Optional[Tensor] = None,
    cutoff: float = 95.0 * Bohr,
    bidirectional: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Calculate atom pair as `edge_index`, and shift vector `S`.

    Args:
        pos (Tensor): atom positions in angstrom
        cell (Tensor): cell size in angstrom, None for non periodic system.
        pbc (Tensor): pbc condition, None for non periodic system.
        cutoff (float): cutoff distance in angstrom
        bidirectional (bool): calculated `edge_index` is bidirectional or not.

    Returns:
        edge_index (Tensor): (2, n_edges)
        S (Tensor): (n_edges, 3) dtype is same with `pos`
    """
    if pbc is None or torch.all(~pbc):
        assert cell is None
        # Calculate distance brute force way
        distances = torch.sum((pos.unsqueeze(0) - pos.unsqueeze(1)).pow_(2), dim=2)
        right_ind, left_ind = torch.where(distances < cutoff**2)
        if bidirectional:
            edge_index = torch.stack(
                (left_ind[left_ind != right_ind], right_ind[left_ind != right_ind])
            )
        else:
            edge_index = torch.stack(
                (left_ind[left_ind < right_ind], right_ind[left_ind < right_ind])
            )
        n_edges = edge_index.shape[1]
        S = pos.new_zeros((n_edges, 3))
    else:
        if not bidirectional:
            raise NotImplementedError("bidirectional=False is not supported")
        if pos.shape[0] == 0:
            edge_index = torch.zeros([2, 0], dtype=torch.long, device=pos.device)
            S = torch.zeros_like(pos)
        else:
            edge_index, S = calc_neighbor_by_vesin(pos, cell, pbc, cutoff)

    return edge_index, S


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
            torch.any(self.pbc != pbc)
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
    Parameters
    ----------
    xc : str
        Exchange-correlation functional (e.g., "pbe", "b3lyp").
    damping : str
        Damping function type (e.g., "bj", "zero").
    cutoff : float
        Cutoff distance in angstrom for D3 interactions.
    abc : bool
        Enable ATM 3-body interaction terms.
    cnthr : float
        Coordination number cutoff distance in angstrom.
    dtype : str
        Data type used for the calculation ("float32" or "float64").
    device : str, optional
        Device used for the calculation. Defaults to "cuda" if available, otherwise "cpu".
    skin : float, default 0.0
        Neighbor list skin distance for efficient neighbor list reuse.
        Uses vesin-based neighbor list implementation for better performance.

    Examples
    --------

    Combined with MACE-MP-0 model:

    >>> import ipsuite as ips
    >>> project = ips.Project()
    >>> d3 = ips.TorchDFTD3(
    ...     xc="pbe",
    ...     damping="bj",
    ...     cutoff=3,
    ...     cnthr=3,
    ...     abc=False,
    ...     dtype="float32",
    ... )
    >>> mp0 = ips.MACEMPModel()
    >>> with project:
    ...     data = ips.Smiles2Atoms(smiles="CCO")
    ...     mix_calc = ips.MixCalculator(calculators=[d3, mp0])
    ...     data_with_d3 = ips.ApplyCalculator(data=data.frames, model=mix_calc)
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
