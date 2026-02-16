from unittest.mock import MagicMock, patch

import ase
import pytest

from ipsuite import MultiPackmol, Packmol

# -----------------------------
# Helpers
# -----------------------------


def dummy_atoms():
    return ase.Atoms("H2")


# -----------------------------
# Basic init tests
# -----------------------------


def test_packmol_ratio_default():
    node = Packmol(
        data=[[dummy_atoms()]],
        count=[1],
        density=1000,
    )
    assert node.ratio == [1.0, 1.0, 1.0]


def test_packmol_ratio_custom():
    node = Packmol(
        data=[[dummy_atoms()]],
        count=[1],
        density=1000,
        ratio=[1.0, 2.0, 3.0],
    )
    assert node.ratio == [1.0, 2.0, 3.0]


def test_packmol_data_count_mismatch():
    with pytest.raises(ValueError):
        Packmol(
            data=[[dummy_atoms()], [dummy_atoms()]],
            count=[1],  # mismatch
            density=1000,
        )


# -----------------------------
# Mutable default safety
# -----------------------------


def test_ratio_not_shared_between_instances():
    a = Packmol(
        data=[[dummy_atoms()]],
        count=[1],
        density=1000,
    )
    b = Packmol(
        data=[[dummy_atoms()]],
        count=[1],
        density=1000,
    )

    a.ratio[0] = 99.0
    assert b.ratio == [1.0, 1.0, 1.0]


# -----------------------------
# run() call test (mock pack)
# -----------------------------


@patch("ipsuite.pack.pack")
@patch("ipsuite.pack.znh5md.IO")
def test_packmol_run_calls_pack(mock_io, mock_pack):
    mock_pack.return_value = dummy_atoms()
    mock_io.return_value = MagicMock()

    node = Packmol(
        data=[[dummy_atoms()]],
        count=[1],
        density=1000,
        ratio=[1.0, 2.0, 1.0],
    )

    node.run()

    mock_pack.assert_called_once()
    kwargs = mock_pack.call_args.kwargs
    assert kwargs["ratio"] == [1.0, 2.0, 1.0]


# -----------------------------
# MultiPackmol inheritance test
# -----------------------------


@patch("ipsuite.pack.pack")
@patch("ipsuite.pack.znh5md.IO")
def test_multipackmol_multiple_configs(mock_io, mock_pack):
    mock_pack.return_value = dummy_atoms()
    mock_io.return_value = MagicMock()

    node = MultiPackmol(
        data=[[dummy_atoms()]],
        count=[1],
        density=1000,
        n_configurations=3,
        ratio=[1.0, 1.0, 2.0],
    )

    node.run()

    assert mock_pack.call_count == 3
    for call in mock_pack.call_args_list:
        assert call.kwargs["ratio"] == [1.0, 1.0, 2.0]
