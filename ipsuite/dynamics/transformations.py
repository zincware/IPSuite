import dataclasses
import logging
import typing

import numpy as np
from ase import units

from ipsuite.utils.ase_sim import get_box_from_density

log = logging.getLogger(__name__)


@dataclasses.dataclass
class WrapModifier:
    """Wrap atoms to the into the cell."""

    def modify(self, thermostat, step, total_steps) -> None:
        thermostat.atoms.wrap()


@dataclasses.dataclass
class RescaleBoxModifier:
    cell: int | None = None
    density: float | None = None
    _initial_cell = None

    def __post_init__(self):
        if self.density is not None and self.cell is not None:
            raise ValueError("Only one of density or cell can be given.")
        if self.density is None and self.cell is None:
            raise ValueError("Either density or cell has to be given.")

    # Currently not possible due to a ZnTrack bug

    def modify(self, thermostat, step, total_steps):
        # we use the thermostat, so we can also modify e.g. temperature
        if self.cell is None:
            self.cell = get_box_from_density([[thermostat.atoms]], [1], self.density)
        if isinstance(self.cell, int):
            self.cell = np.array(
                [[self.cell, 0, 0], [0, self.cell, 0], [0, 0, self.cell]]
            )
        elif isinstance(self.cell, list):
            self.cell = np.array(
                [[self.cell[0], 0, 0], [0, self.cell[1], 0], [0, 0, self.cell[2]]]
            )

        if self._initial_cell is None:
            self._initial_cell = thermostat.atoms.get_cell()
        percentage = step / (total_steps)
        new_cell = (1 - percentage) * self._initial_cell + percentage * self.cell
        thermostat.atoms.set_cell(new_cell, scale_atoms=True)


@dataclasses.dataclass
class BoxOscillatingRampModifier:
    """Ramp the simulation cell to a specified end cell with some oscillations.

    Attributes
    ----------
    end_cell: float, list[float], optional
        cell to ramp to, cubic or tetragonal. If None, the cell will oscillate
        around the initial cell.
    cell_amplitude: float
        amplitude in oscillations of the diagonal cell elements
    num_oscillations: float
        number of oscillations. No oscillations will occur if set to 0.
    interval: int, default 1
        interval in which the box size is changed.
    num_ramp_oscillations: float, optional
        number of oscillations to ramp the box size to the end cell.
        This value has to be smaller than num_oscillations.
        For LotF applications, this can prevent a loop of ever decreasing cell sizes.
        To ensure this use a value of 0.5.
    """

    def __post_init__(self):
        if self.num_ramp_oscillations is not None:
            if self.num_ramp_oscillations > self.num_oscillations:
                raise ValueError(
                    "num_ramp_oscillations has to be smaller than num_oscillations."
                )

    cell_amplitude: typing.Union[float, list[float]]
    num_oscillations: float
    end_cell: float | None = None
    num_ramp_oscillations: float | None = None
    interval: int = 1
    _initial_cell = None

    def modify(self, thermostat, step, total_steps):
        if self.end_cell is None:
            self.end_cell = thermostat.atoms.get_cell()
        if self._initial_cell is None:
            self._initial_cell = thermostat.atoms.get_cell()
            if isinstance(self.end_cell, (float, int)):
                self.end_cell = np.array(
                    [
                        [self.end_cell, 0, 0],
                        [0, self.end_cell, 0],
                        [0, 0, self.end_cell],
                    ]
                )
            elif isinstance(self.end_cell, list):
                self.end_cell = np.array(
                    [
                        [self.end_cell[0], 0, 0],
                        [0, self.end_cell[1], 0],
                        [0, 0, self.end_cell[2]],
                    ]
                )

        percentage = step / (total_steps)
        # if num_ramp_oscillations is set, the cell size is ramped to end_cell within
        # num_ramp_oscillations instead of num_oscillations. This can prevent a loop of
        # ever decreasing cell sizes in LoTF applications where simulations
        # can be aborted at small cell sizes.
        if self.num_ramp_oscillations is not None:
            percentage_per_oscillation = (
                percentage * self.num_oscillations / self.num_ramp_oscillations
            )
            percentage_per_oscillation = min(percentage_per_oscillation, 1)
        else:
            # ramp over all oscillations
            percentage_per_oscillation = percentage

        ramp = percentage_per_oscillation * (self.end_cell - self._initial_cell)
        oscillation = self.cell_amplitude * np.sin(
            2 * np.pi * percentage * self.num_oscillations
        )
        oscillation = np.eye(3) * oscillation
        new_cell = self._initial_cell + ramp + oscillation

        if step % self.interval == 0:
            thermostat.atoms.set_cell(new_cell, scale_atoms=True)


@dataclasses.dataclass
class TemperatureRampModifier:
    """Ramp the temperature from start_temperature to temperature.

    Attributes
    ----------
    start_temperature: float, optional
        temperature to start from, if None, the temperature of the thermostat is used.
    temperature: float
        temperature to ramp to.
    interval: int, default 1
        interval in which the temperature is changed.
    """

    temperature: float
    start_temperature: float | None = None
    interval: int = 1

    def modify(self, thermostat, step, total_steps):
        # we use the thermostat, so we can also modify e.g. temperature
        if self.start_temperature is None:
            # different thermostats call the temperature attribute differently
            if hasattr(thermostat, "temp"):
                start_temperature = thermostat.temp
            elif hasattr(thermostat, "temperature"):
                start_temperature = thermostat.temperature
            self.start_temperature = start_temperature / units.kB

        percentage = step / (total_steps - 1)
        new_temperature = (
            1 - percentage
        ) * self.start_temperature + percentage * self.temperature
        if step % self.interval == 0:
            thermostat.set_temperature(temperature_K=new_temperature)


@dataclasses.dataclass
class TemperatureOscillatingRampModifier:
    """Ramp the temperature from start_temperature to temperature with some oscillations.

    Attributes
    ----------
    start_temperature: float, optional
        temperature to start from, if None, the temperature of the thermostat is used.
    end_temperature: float
        temperature to ramp to.
    temperature_amplitude: float
        amplitude of temperature oscillations.
    num_oscillations: float
        number of oscillations. No oscillations will occur if set to 0.
    interval: int, default 1
        interval in which the temperature is changed.
    """

    end_temperature: float
    temperature_amplitude: float
    num_oscillations: float
    start_temperature: float | None = None
    interval: int = 1

    def modify(self, thermostat, step, total_steps):
        # we use the thermostat, so we can also modify e.g. temperature
        if self.start_temperature is None:
            # different thermostats call the temperature attribute differently
            if hasattr(thermostat, "temp"):
                start_temperature = thermostat.temp
            elif hasattr(thermostat, "temperature"):
                start_temperature = thermostat.temperature
            self.start_temperature = start_temperature / units.kB

        ramp = step / total_steps * (self.end_temperature - self.start_temperature)
        oscillation = self.temperature_amplitude * np.sin(
            2 * np.pi * step / total_steps * self.num_oscillations
        )
        new_temperature = self.start_temperature + ramp + oscillation

        new_temperature = max(0, new_temperature)  # prevent negative temperature

        if step % self.interval == 0:
            thermostat.set_temperature(temperature_K=new_temperature)


@dataclasses.dataclass
class PressureRampModifier:
    """Ramp the temperature from start_temperature to temperature.
    Works only for the NPT thermostat (not NPTBerendsen).

    Attributes
    ----------
    start_pressure_au: float, optional
        pressure to start from, if None, the pressure of the thermostat is used.
        Uses ASE units.
    end_pressure_au: float
        pressure to ramp to. Uses ASE units.
    interval: int, default 1
        interval in which the pressure is changed.
    """

    end_pressure_au: float
    start_pressure_au: float | None = None
    interval: int = 1

    def modify(self, thermostat, step, total_steps):
        if self.start_pressure_au is None:
            self.start_pressure_au = thermostat.externalstress

        frac = step / total_steps
        new_pressure = (-self.start_pressure_au[0]) ** (1 - frac)
        new_pressure *= self.end_pressure_au ** (frac)

        if step % self.interval == 0:
            thermostat.set_stress(new_pressure)
