import pathlib
import typing

import ase
import h5py
import numpy as np
import tqdm
import znh5md
import zntrack
from ase.calculators.calculator import Calculator, all_changes

from ipsuite.analysis.model.math import force_decomposition
from ipsuite.geometry import BarycenterMapping
from ipsuite.utils.ase_sim import freeze_copy_atoms


class InterIntraMD(zntrack.Node):
    model: typing.Any = zntrack.deps()
    data: list[ase.Atoms] = zntrack.deps()
    data_id: int = zntrack.params(-1)

    # Weights
    wt: float = zntrack.params(1)
    wr: float = zntrack.params(1)
    wv: float = zntrack.params(1)

    # MD Parameter
    thermostat: typing.Any = zntrack.deps()
    steps: int = zntrack.params(1_000_000)
    sampling_rate: int = zntrack.params(5)
    dump_rate: int = zntrack.params(1000)

    frames_path: pathlib.Path = zntrack.outs_path(zntrack.nwd / "frames.h5")
    model_outs: pathlib.Path = zntrack.outs_path(zntrack.nwd / "model/")

    def run(self):
        self.model_outs.mkdir(parents=True, exist_ok=True)
        (self.model_outs / "outs.txt").write_text("Lorem Ipsum")
        frames = self.data[self.data_id]
        calculator = self.model.get_calculator(directory=self.model_outs)
        calc = BiasedCalc(calc=calculator, weights=[self.wt, self.wr, self.wv])
        frames.calc = calc

        dyn = self.thermostat.get_thermostat(atoms=frames)

        buffer = []
        i = 0
        io = znh5md.IO(self.frames_path)
        for _ in tqdm.tqdm(dyn.irun(self.steps), total=self.steps):
            if i % self.sampling_rate == 0:
                buffer.append(freeze_copy_atoms(frames))
                if len(buffer) >= self.dump_rate:
                    io.extend(buffer)
                    buffer = []
            i += 1
        io.extend(buffer)

    @property
    def frames(self) -> typing.List[ase.Atoms]:
        with self.state.fs.open(self.frames_path, "rb") as f:
            with h5py.File(f) as file:
                return znh5md.IO(file_handle=file)[:]


class BiasedCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, calc, weights, atoms=None):
        Calculator.__init__(self, atoms=atoms)
        self.calc = calc
        self.weights = weights
        self.counter: int = 0
        self.mols_index = None
        self.mol_map = None

    def calculate(
        self,
        atoms: ase.Atoms = None,
        properties=None,
        system_changes=all_changes,
    ):
        properties = ["energy", "forces"]
        Calculator.calculate(self, atoms, properties, system_changes)

        energy = self.calc.get_potential_energy(atoms)
        forces = self.calc.get_forces(atoms)

        if np.any(np.abs(forces) > 1e7):
            raise Exception("System Broken")
        mapping = BarycenterMapping(frozen=True)

        atom_trans_forces, atom_rot_forces, atom_vib_forces, self.mol_map = (
            force_decomposition(atoms, mapping, forces.copy(), mol_map=self.mol_map)
        )
        f_ges = (
            self.weights[0] * atom_trans_forces
            + self.weights[1] * atom_rot_forces
            + self.weights[2] * atom_vib_forces
        )

        self.results = {
            "energy": energy,
            "forces": f_ges,
            "f_trans": self.weights[0] * atom_trans_forces,
            "f_rot": self.weights[1] * atom_rot_forces,
            "f_vib": self.weights[2] * atom_vib_forces,
            "full_forces": forces,
        }
