import os
import pathlib
import typing

import ase.io
import dvc.cli
import git
import numpy as np
import zntrack
from ase.calculators.singlepoint import SinglePointCalculator
from zntrack.tools import timeit

from ipsuite import AddData, Project, base, fields


class UpdateCalculator(base.ProcessSingleAtom):
    """Update the calculator of an atoms object.

    Set energy, forces to zero.
    """

    energy = zntrack.zn.params(0.0)
    forces = zntrack.zn.params((0, 0, 0))

    time: float = zntrack.zn.metrics()

    @timeit(field="time")
    def run(self) -> None:
        self.atoms = self.get_data()

        self.atoms.calc = SinglePointCalculator(
            self.atoms,
            energy=self.energy,
            forces=np.stack([self.forces] * len(self.atoms)),
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
    processor: base.ProcessSingleAtom = zntrack.zn.nodes()
    repo: str = zntrack.meta.Text(None)
    commit: bool = zntrack.meta.Text(True)
    clean_exp: bool = zntrack.meta.Text(True)

    def run(self):
        # lazy loading: load now
        _ = self.data
        processor = self.processor
        processor.name = processor.__class__.__name__

        repo = git.Repo.init(self.repo or self.name)

        gitignore = pathlib.Path(".gitignore")
        # TODO: move this into a function
        if not gitignore.exists():
            gitignore.write_text(f"{repo.working_dir}\n")
        elif repo.working_dir not in gitignore.read_text().split(" "):
            gitignore.write_text(f"{repo.working_dir}\n")

        os.chdir(repo.working_dir)
        dvc.cli.main(["init"])
        project = Project()

        with project:
            data = AddData(file="atoms.xyz")
        project.run(repro=False)

        processor.data = data @ "atoms"
        processor.write_graph()

        repo.git.add(all=True)
        repo.index.commit("Build graph")

        if self.clean_exp:
            dvc.cli.main(["exp", "gc", "-w", "-f"])

        self.run_exp(project, processor)
        if self.commit:
            self.run_commits(repo)

        os.chdir("..")  # we need to go back to save

    def run_exp(self, project, processor):
        exp_lst = []
        for atom in self.data:
            with project.create_experiment() as exp:
                ase.io.write("atoms.xyz", atom)
            exp_lst.append(exp)
        project.run_exp()

        self.atoms = [
            processor.from_rev(name=processor.name, rev=x.name).atoms[0] for x in exp_lst
        ]

    def run_commits(self, repo):
        commits = []
        for idx, atom in enumerate(self.data):
            ase.io.write("atoms.xyz", atom)
            dvc.cli.main(["add", "atoms.xyz"])
            dvc.cli.main(["repro"])
            repo.git.add(all=True)
            # do not use repo.index.add("*"); it will add atoms.xyz
            commit_message = f"repro {self.name}_{idx}"
            commits.append(repo.index.commit(commit_message))
