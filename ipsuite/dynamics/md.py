import dataclasses
import logging
import typing

import ase
import ase.constraints
import ase.geometry
import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.verlet import VelocityVerlet

from ipsuite.calculators.integrators import StochasticVelocityCellRescaling

log = logging.getLogger(__name__)


@dataclasses.dataclass
class LangevinThermostat:
    """Initialize the langevin thermostat

    Attributes
    ----------
    time_step: float
        The simulation time step should be adjust for the system.
        To properly resolve C-H vibrations, a time step of 0.5 fs is recommended.
        For systems without significant C-H vibrations, larger time steps might be used.

    temperature: float
        temperature in Kelvin to simulate at

    friction: float
        friction of the Langevin simulator
    """

    time_step: float
    temperature: float
    friction: float

    def get_thermostat(self, atoms: ase.Atoms) -> Langevin:
        thermostat = Langevin(
            atoms=atoms,
            timestep=self.time_step * units.fs,
            temperature_K=self.temperature,
            friction=self.friction,
        )
        return thermostat


@dataclasses.dataclass
class VelocityVerletDynamic:
    """Initialize the Velocity Verlet dynamics

    Attributes
    ----------
    time_step: float
        time step of simulation
    """

    time_step: int

    def get_thermostat(self, atoms):
        dyn = VelocityVerlet(
            atoms=atoms,
            timestep=self.time_step * units.fs,
        )
        return dyn


@dataclasses.dataclass
class NPTThermostat:
    """Initialize the ASE NPT barostat
    (Nose Hoover temperature coupling + Parrinello Rahman pressure coupling).

    Attributes
    ----------
    time_step: float
        time step of simulation

    temperature: float
        temperature in K to simulate at

    pressure: float
        pressure in ASE units

    ttime: float
        characteristic temperature coupling time in ASE units

    pfactor: float
        characteristic pressure coupling time in ASE units

    tetragonal_strain: bool
        if True allows only the diagonal elements of the box to change,
        i.e. box angles are constant

    fraction_traceless: Union[int, float]
        How much of the traceless part of the virial to keep.
        If set to 0, the volume of the cell can change, but the shape cannot.
    """

    time_step: float
    temperature: float
    pressure: float
    ttime: float
    pfactor: float
    tetragonal_strain: bool = True
    fraction_traceless: typing.Union[int, float] = 1

    def get_thermostat(self, atoms):
        if self.tetragonal_strain:
            mask = np.array(
                [
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                ]
            )
        else:
            mask = None
        self.time_step *= units.fs
        thermostat = NPT(
            atoms,
            self.time_step,
            temperature_K=self.temperature,
            externalstress=self.pressure,
            ttime=self.ttime,
            pfactor=self.pfactor,
            mask=mask,
        )
        thermostat.set_fraction_traceless(self.fraction_traceless)
        return thermostat


@dataclasses.dataclass
class SVCRBarostat:
    """Initialize the CSVR thermostat

    Attributes
    ----------
    time_step: float
        time step of simulation

    temperature: float
        temperature in K to simulate at
    betaT: float
        Very approximate compressibility of the system.
    pressure_au: float
        Pressure in atomic units.
    taut: float
        Temperature coupling time scale.
    taup: float
        Pressure coupling time scale.
    """

    time_step: int
    temperature: float
    betaT: float = 4.57e-5
    pressure_au: float = 1.01325
    taut: float = 100
    taup: typing.Optional[float] = None

    def get_thermostat(self, atoms):
        if self.taup:
            taup = self.taup * units.fs
        else:
            taup = self.taup

        thermostat = StochasticVelocityCellRescaling(
            atoms=atoms,
            timestep=self.time_step * units.fs,
            temperature_K=self.temperature,
            betaT=self.betaT / units.bar,
            pressure_au=self.pressure_au * units.bar,
            taut=self.taut * units.fs,
            taup=taup,
        )
        return thermostat


@dataclasses.dataclass
class Berendsen:
    """Initialize the Berendsen thermostat

    Attributes
    ----------
    time_step: float
        time step of simulation
    temperature: float
        temperature in K to simulate at
    taut: float
        Temperature coupling time scale.
    """

    time_step: float
    temperature: float
    taut: float = 100

    def get_thermostat(self, atoms):
        thermostat = NVTBerendsen(
            atoms=atoms,
            timestep=self.time_step * units.fs,
            temperature_K=self.temperature,
            taut=self.taut * units.fs,
        )
        return thermostat
