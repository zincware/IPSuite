import os
import pathlib
import typing

import ase.io
import dvc.cli
import git
import numpy as np
import zntrack
from ase.calculators.singlepoint import SinglePointCalculator

from ipsuite import AddData, base, fields


class UpdateCalculator(base.ProcessSingleAtom):
    """Update the calculator of an atoms object.

    Set energy, forces to zero.
    """

    def run(self) -> None:
        self.atoms = self.get_data()

        self.atoms.calc = SinglePointCalculator(
            self.atoms, energy=0, forces=np.zeros((len(self.atoms), 3))
        )
        self.atoms = [self.atoms]


class MockAtoms(zntrack.Node):
    """Create Atoms objects with random data."""

    atoms: typing.List[ase.Atoms] = fields.Atoms()
    seed: int = zntrack.zn.params(0)

    n_configurations: int = zntrack.zn.params(10)
    n_atoms: int = zntrack.zn.params(10)

    calculator: bool = zntrack.zn.params(True)

    def run(self) -> None:
        self.atoms = []
        np.random.seed(self.seed)
        for _ in range(self.n_configurations):
            atoms = ase.Atoms(
                symbols="C" * self.n_atoms,
                positions=np.random.random((self.n_atoms, 3)),
            )
            if self.calculator:
                atoms.calc = SinglePointCalculator(
                    atoms,
                    energy=np.random.random(),
                    forces=np.random.random((self.n_atoms, 3)),
                )
            self.atoms.append(atoms)


class AtomsToXYZ(base.AnalyseAtoms):
    """Convert Atoms objects to XYZ files."""

    output: pathlib.Path = zntrack.dvc.outs(zntrack.nwd / "atoms")

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        for idx, atom in enumerate(self.data):
            ase.io.write(self.output / f"{idx:05d}.xyz", atom)

    @property
    def files(self) -> typing.List[pathlib.Path]:
        return [x.resolve() for x in self.output.glob("*.xyz")]


class NodesPerAtoms(base.ProcessAtoms):
    # processor: base.ProcessSingleAtom = zntrack.zn.nodes()
    repo: str = zntrack.meta.Text(None)

    def run(self):
        _ = self.data  # lazy loading: load now
        repo = git.Repo.init(self.repo or self.name)
        os.chdir(repo.working_dir)
        dvc.cli.main(["init"])
        project = zntrack.Project()

        with project:
            data = AddData(file="atoms.xyz")
            processor = UpdateCalculator(data=data.atoms, data_id=0)
            # we replace processor with a zn.nodes and
            # then we want to update the parameters

        project.run(repro=False)

        # TODO use some parallelization here, e.g. dvc exp + dask4dvc
        commits = []
        for idx, atom in enumerate(self.data):
            ase.io.write("atoms.xyz", atom)
            dvc.cli.main(["add", "atoms.xyz"])
            dvc.cli.main(["repro"])
            repo.git.add(all=True)
            # do not use repo.index.add("*"); it will add atoms.xyz
            commit_message = f"repro {self.name}_{idx}"
            commits.append(repo.index.commit(commit_message))

        self.atoms = [processor.from_rev(rev=x.hexsha).atoms[0] for x in commits]
        os.chdir("..")  # we need to go back to save
