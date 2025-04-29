from ase import units
from ase.calculators.plumed import Plumed
import zntrack
from ipsuite import base
import typing
import pathlib
import ase

def get_plumed_setup(sigma, height, pace, biasfactor, plumed_path, temp):
    ps = 1000 * units.fs

    setup = [
        f"UNITS LENGTH=A TIME={1/ps} ENERGY={units.mol/units.kJ}",
        "O_add: POSITION ATOM=121 SCALED_COMPONENTS",
        "LOWER_WALLS ARG=O_add.a,O_add.b,O_add.c AT=-0.5,-0.5,+0.17 KAPPA=150.0,150.0,150.0 EXP=2,2,2 EPS=1,1,1 OFFSET=0,0,0",
        "UPPER_WALLS ARG=O_add.a,O_add.b,O_add.c AT=+0.5,+0.5,+0.4 KAPPA=150.0,150.0,150.0 EXP=2,2,2 EPS=1,1,1 OFFSET=0,0,0",
        f"mtd: METAD ARG=O_add.a,O_add.b,O_add.c SIGMA={sigma},{sigma},{sigma} HEIGHT={height} PACE={pace} BIASFACTOR={biasfactor} TEMP={temp} " +
        f"GRID_MIN=-0.5,-0.5,-0.5 GRID_MAX=+0.5,+0.5,+0.5 GRID_BIN=100,100,100 CALC_MAX_BIAS CALC_RCT FILE={plumed_path}/HILLS",
        f"PRINT ARG=O_add.a,O_add.b,O_add.c,mtd.bias STRIDE=10 FILE={plumed_path}/COLLVAR ", 
    ]
    return setup

class PlumedCalc(base.IPSNode):

    data: list[ase.Atoms] = zntrack.deps()
    model: typing.Any = zntrack.deps()
    
    data_id: typing.Optional[int] = zntrack.params(-1)
    timestep: float = zntrack.params()
    kT: float = zntrack.params()
    sigma: float = zntrack.params()
    height: float = zntrack.params()
    pace: float = zntrack.params()
    biasfactor: float = zntrack.params()
    temp: float = zntrack.params()
    
    directory: pathlib.Path = zntrack.outs_path(zntrack.nwd / "plumed")
            
    def run(self):
        pathlib.Path(self.directory).mkdir(parents=True, exist_ok=True)
        (self.directory / "outs.txt").write_text("Lorem Ipsum")

    def get_calculator(self, directory: str = None, **kwargs):
        if directory is None:
            directory = self.directory
            
        input = get_plumed_setup(
            self.sigma,
            self.height,
            self.pace,
            self.biasfactor,
            directory,
            self.temp,
        )
        
        calc = Plumed(
            calc=self.model.get_calculator(),
            input=input,
            atoms=self.data[self.data_id],
            kT=self.kT,
            timestep=self.timestep,
            )
        
        return calc