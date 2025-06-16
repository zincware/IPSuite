from ipsuite.models.cp2k import CP2KOutput
from pathlib import Path
import pytest
import numpy as np

DATA_PATH = Path(__file__).parent / "data"

def test_cp2k_output():
    file_content = (DATA_PATH / "cp2k.out").read_text()
    output = CP2KOutput.from_file_content(file_content)
    assert output.energy == pytest.approx(-3744, abs=1.0)
    # using linalg.norm as some collective variable to represent the dataset.
    assert np.linalg.norm(output.forces) == pytest.approx(7.4088829, abs=1e-5)
    assert np.linalg.norm(output.stress) == pytest.approx(0.077688, abs=1e-5)
    assert np.linalg.norm(output.hirshfeld_charges) == pytest.approx(3.89524, abs=1e-5)
