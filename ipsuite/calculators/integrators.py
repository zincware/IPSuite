import math

import numpy as np
from ase import units
from ase.md.md import MolecularDynamics
from ase.parallel import world



class StochasticCellRescalingCSVR(MolecularDynamics):
    """Bussi stochastic velocity rescaling (NVT) molecular dynamics.
    Based on the paper from Bussi et al. (https://arxiv.org/abs/0803.4060)
    Thermostat is based on the ASE  implementation of SVR.

    Parameters
    ----------
    atoms : Atoms
        The atoms object.
    timestep : float
        The time step in ASE time units.
    temperature_K : float
        The desired temperature, in Kelvin.
    taut : float
        Time constant for Bussi temperature coupling in ASE time units.
    rng : numpy.random, optional
        Random number generator.
    **md_kwargs : dict, optional
        Additional arguments passed to :class:~ase.md.md.MolecularDynamics
        base class.
    """

    def __init__(
        self,
        atoms,
        timestep,
        temperature_K,
        betaT = 4.57e-5 / units.bar,
        pressure_au= 1.01325 * units.bar,
        taut = 100 * units.fs,
        taup = 1000 * units.fs,
        rng=np.random,
        **md_kwargs,
    ):
        super().__init__(
            atoms,
            timestep,
            **md_kwargs,
        )

        self.temp = temperature_K * units.kB
        self.taut = taut
        self.taup = taup
        self.communicator = world
        self.rng = rng
        self.betaT = betaT
        self.pressure = pressure_au

        self.ndof = len(self.atoms) * 3

        self.target_kinetic_energy = 0.5 * self.temp * self.ndof

        if np.isclose(
            self.atoms.get_kinetic_energy(), 0.0, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "Initial kinetic energy is zero. "
                "Please set the initial velocities before running Bussi NVT."
            )

        self._exp_term = math.exp(-self.dt / self.taut)
        self._masses = self.atoms.get_masses()[:, np.newaxis]

        self.transferred_energy = 0.0

    def scale_velocities(self):
        """Do the NVT Bussi stochastic velocity scaling."""
        kinetic_energy = self.atoms.get_kinetic_energy()
        alpha = self.calculate_alpha(kinetic_energy)

        momenta = self.atoms.get_momenta()
        self.atoms.set_momenta(alpha * momenta)

        self.transferred_energy += (alpha**2 - 1.0) * kinetic_energy

    def calculate_alpha(self, kinetic_energy):
        """Calculate the scaling factor alpha using equation (A7)
        from the Bussi paper."""

        energy_scaling_term = (
            (1 - self._exp_term)
            * self.target_kinetic_energy
            / kinetic_energy
            / self.ndof
        )

        # R1 in Eq. (A7)
        normal_noise = self.rng.standard_normal()
        # \sum_{i=2}^{Nf} R_i^2 in Eq. (A7)
        # 2 * standard_gamma(n / 2) is equal to chisquare(n)
        sum_of_noises = 2.0 * self.rng.standard_gamma(0.5 * (self.ndof - 1))

        return math.sqrt(
            self._exp_term
            + energy_scaling_term * (sum_of_noises + normal_noise**2)
            + 2
            * normal_noise
            * math.sqrt(self._exp_term * energy_scaling_term)
        )

    def scale_positions_and_cell(self):
        """ Do the Berendsen pressure coupling,
        scale the atom position and the simulation cell."""

        stress = self.atoms.get_stress(voigt=False, include_ideal_gas=True)

        volume= self.atoms.cell.volume
        pint= -stress.trace() / 3
        pint += 2.0/3.0 * self.atoms.get_kinetic_energy() / volume

        dW = self.rng.standard_normal(size=1)
        deterministic = -self.betaT/self.taup*(self.pressure-pint)*self.dt
        stochastic = np.sqrt(2*self.temp*self.betaT/volume/self.taup*self.dt)*dW
        depsilon= deterministic + stochastic

        scaling=np.exp(depsilon/3.0)

        cell = self.atoms.get_cell()
        cell = scaling * cell
        self.atoms.set_cell(cell, scale_atoms=True)
    
        velocities = self.atoms.get_velocities()
        velocities = velocities / scaling
        self.atoms.set_velocities(velocities)


    def step(self, forces=None):
        """Move one timestep forward using Bussi NVT molecular dynamics."""
        if forces is None:
            forces = self.atoms.get_forces(md=True)

        self.scale_positions_and_cell()

        self.scale_velocities()

        self.atoms.set_momenta(
            self.atoms.get_momenta() + 0.5 * self.dt * forces
        )
        momenta = self.atoms.get_momenta()

        self.atoms.set_positions(
            self.atoms.positions + self.dt * momenta / self._masses
        )

        forces = self.atoms.get_forces(md=True)

        self.atoms.set_momenta(momenta + 0.5 * self.dt * forces)

        return forces
