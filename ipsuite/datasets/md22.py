import typing
import urllib
import zipfile
from pathlib import Path
import tempfile

import ase
from ase import units
import zntrack

import ipsuite as ips
from ipsuite import fields


def modify_xyz_file(file_path, target_string, replacement_string):
    new_file_path = file_path.with_name(file_path.stem + "_mod" + file_path.suffix)

    with file_path.open("r") as input_file, new_file_path.open("w") as output_file:
        for line in input_file:
            # Replace all occurrences of the target string with the replacement string
            modified_line = line.replace(target_string, replacement_string)
            output_file.write(modified_line)
    return new_file_path


def download_data(url: str, data_path: Path):
    url_path = Path(urllib.parse.urlparse(url).path)
    zip_path = data_path / url_path.stem
    file_path = zip_path.with_suffix(".xyz")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    file_path = modify_xyz_file(
        file_path, target_string="Energy", replacement_string="energy"
    )
    return file_path


class MD22Dataset(ips.base.IPSNode):
    dataset: str = zntrack.params()

    atoms: typing.List[ase.Atoms] = fields.Atoms()

    raw_data_dir: Path = zntrack.outs_path(zntrack.nwd / "raw_data")

    datasets = {
        "Ac-Ala3-NHMe": (
            "http://www.quantum-machine.org/gdml/repo/static/md22_Ac-Ala3-NHMe.zip"
        ),
        "DHA": "http://www.quantum-machine.org/gdml/repo/static/md22_DHA.zip",
        "stachyose": "http://www.quantum-machine.org/gdml/repo/static/md22_stachyose.zip",
        "AT-AT": "http://www.quantum-machine.org/gdml/repo/static/md22_AT-AT.zip",
        "AT-AT-CG-CG": (
            "http://www.quantum-machine.org/gdml/repo/static/md22_AT-AT-CG-CG.zip"
        ),
        "buckyball-catcher": (
            "http://www.quantum-machine.org/gdml/repo/static/md22_buckyball-catcher.zip"
        ),
        "double-walled_nanotube": "http://www.quantum-machine.org/gdml/repo/static/md22_double-walled_nanotube.zip",
    }

    def run(self):
        tmpdir = tempfile.TemporaryDirectory()
        raw_data_dir = Path(tmpdir.name) / "raw_data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        if self.dataset not in self.datasets.keys():
            raise FileNotFoundError(
                f"Dataset {self.dataset} is not known. Key has top be in {self.datasets}"
            )

        url = self.datasets[self.dataset]

        file_path = download_data(url, raw_data_dir)

        self.atoms = ase.io.read(file_path, ":")
        for atoms in self.atoms:
            atoms.calc.results["energy"] *= units.kcal / units.mol
            atoms.calc.results["forces"] *= units.kcal / units.mol
        tmpdir.cleanup()
