import os
import pathlib

from ipsuite.configuration_comparison import ConfigurationComparison


def test_save_representation(tmp_path, atoms_list):
    os.chdir(tmp_path)
    atoms = atoms_list
    comparison = ConfigurationComparison(
        analyte=atoms, reference=None, soap={"periodic": False}
    )
    comparison.save_representation()
    assert pathlib.Path(comparison.soap_file).is_file()
