from typing import IO, Optional, Union

import numpy as np

from ase import Atoms, units
from ase.parallel import DummyMPI, world
from ase.md.langevin import Langevin

class LangevinHalfSteps(Langevin):
    """New class to implement half steps functions"""

    def __init__(
            self,
            atoms: Atoms,
            timestep: float,
            temperature: Optional[float] = None,
            friction: Optional[float] = None,
            fixcm: bool = True,
            *,
            temperature_K: Optional[float] = None,
            trajectory: Optional[str] = None,
            logfile: Optional[Union[IO, str]] = None,
            loginterval: int = 1,
            communicator=world,
            rng=None,
            append_trajectory: bool = False,
    ):
        """
                Parameters:

                atoms: Atoms object
                    The list of atoms.

                timestep: float
                    The time step in ASE time units.

                temperature: float (deprecated)
                    The desired temperature, in electron volt.

                temperature_K: float
                    The desired temperature, in Kelvin.

                friction: float
                    A friction coefficient in inverse ASE time units.
                    For example, set ``0.01 / ase.units.fs`` to provide
                    0.01 fs\\ :sup:`−1` (10 ps\\ :sup:`−1`).

                fixcm: bool (optional)
                    If True, the position and momentum of the center of mass is
                    kept unperturbed.  Default: True.

                rng: RNG object (optional)
                    Random number generator, by default numpy.random.  Must have a
                    standard_normal method matching the signature of
                    numpy.random.standard_normal.

                logfile: file object or str (optional)
                    If *logfile* is a string, a file with that name will be opened.
                    Use '-' for stdout.

                trajectory: Trajectory object or str (optional)
                    Attach trajectory object.  If *trajectory* is a string a
                    Trajectory will be constructed.  Use *None* (the default) for no
                    trajectory.

                communicator: MPI communicator (optional)
                    Communicator used to distribute random numbers to all tasks.
                    Default: ase.parallel.world. Set to None to disable communication.

                append_trajectory: bool (optional)
                    Defaults to False, which causes the trajectory file to be
                    overwritten each time the dynamics is restarted from scratch.
                    If True, the new structures are appended to the trajectory
                    file instead.

                The temperature and friction are normally scalars, but in principle one
                quantity per atom could be specified by giving an array.

                RATTLE constraints can be used with these propagators, see:
                E. V.-Eijnden, and G. Ciccotti, Chem. Phys. Lett. 429, 310 (2006)

                The propagator is Equation 23 (Eq. 39 if RATTLE constraints are used)
                of the above reference.  That reference also contains another
                propagator in Eq. 21/34; but that propagator is not quasi-symplectic
                and gives a systematic offset in the temperature at large time steps.
                """
        super().__init__(atoms,
                         timestep,
                         temperature,
                         friction,
                         fixcm,
                         temperature_K=temperature_K,
                         trajectory=trajectory,
                         logfile=logfile,
                         loginterval=loginterval,
                         communicator=communicator,
                         rng=rng,
                         append_trajectory=append_trajectory)

    def _1st_half_step(self, forces):
        atoms = self.atoms
        natoms = len(self.atoms)
        self.v = self.atoms.get_velocities()

        xi = self.rng.standard_normal(size=(natoms, 3))
        eta = self.rng.standard_normal(size=(natoms, 3))

        # When holonomic constraints for rigid linear triatomic molecules are
        # present, ask the constraints to redistribute xi and eta within each
        # triple defined in the constraints. This is needed to achieve the
        # correct target temperature.
        for constraint in self.atoms.constraints:
            if hasattr(constraint, 'redistribute_forces_md'):
                constraint.redistribute_forces_md(atoms, xi, rand=True)
                constraint.redistribute_forces_md(atoms, eta, rand=True)

        self.communicator.broadcast(xi, 0)
        self.communicator.broadcast(eta, 0)

        # To keep the center of mass stationary, we have to calculate
        # the random perturbations to the positions and the momenta,
        # and make sure that they sum to zero.
        self.rnd_pos = self.c5 * eta
        self.rnd_vel = self.c3 * xi - self.c4 * eta
        if self.fix_com:
            self.rnd_pos -= self.rnd_pos.sum(axis=0) / natoms
            self.rnd_vel -= (self.rnd_vel * self.masses).sum(axis=0) / (self.masses * natoms)

        # First halfstep in the velocity.
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v + self.rnd_vel)

        # Full step in positions
        x = atoms.get_positions()

        # Step: x^n -> x^(n+1) - this applies constraints if any.
        atoms.set_positions(x + self.dt * self.v + self.rnd_pos)
        atoms.calc.results['forces'] = np.zeros_like(forces)
        # recalc velocities after RATTLE constraints are applied
        atoms.set_momenta((self.atoms.get_positions() - x - self.rnd_pos) / (self.dt * self.masses))

    def _2nd_half_step(self, forces):
        self.v = self.atoms.get_velocities()
        self.v += (self.c1 * forces / self.masses - self.c2 * self.v + self.rnd_vel)
        self.atoms.set_momenta(self.v * self.masses)
        self.call_observers()