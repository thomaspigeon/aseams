"""Langevin dynamics class."""

import warnings
from typing import Optional

import numpy as np

from ase import Atoms, units
from ase.md.md import MolecularDynamics
from ase.constraints import FixCom

class LangevinOBABO(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 5

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature: Optional[float] = None,
        friction: Optional[float] = None,
        fixcm: bool = False,
        *,
        temperature_K: Optional[float] = None,
        rng=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
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

        **kwargs : dict, optional
            Additional arguments passed to :class:~ase.md.md.MolecularDynamics
            base class.

        The temperature and friction are normally scalars, but in principle one
        quantity per atom could be specified by giving an array.

        RATTLE constraints can be used with these propagators, see:
        E. V.-Eijnden, and G. Ciccotti, Chem. Phys. Lett. 429, 310 (2006)

        The propagator is Equation 23 (Eq. 39 if RATTLE constraints are used)
        of the above reference.  That reference also contains another
        propagator in Eq. 21/34; but that propagator is not quasi-symplectic
        and gives a systematic offset in the temperature at large time steps.
        """
        if 'communicator' in kwargs:
            msg = (
                '`communicator` has been deprecated since ASE 3.25.0 '
                'and will be removed in ASE 3.26.0. Use `comm` instead.'
            )
            warnings.warn(msg, FutureWarning)
            kwargs['comm'] = kwargs.pop('communicator')

        if friction is None:
            raise TypeError("Missing 'friction' argument.")
        self.fr = friction
        self.temp = units.kB * self._process_temperature(
            temperature, temperature_K, 'eV'
        )

        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
        MolecularDynamics.__init__(self, atoms, timestep, **kwargs)
        # clear all constrains
        self._constraints = []
        self.fix_com = fixcm
        if fixcm:
            self.atoms.set_constraint(FixCom())
        self.updatevars()

    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update(
            {
                'temperature_K': self.temp / units.kB,
                'friction': self.fr,
            }
        )
        return d

    def set_temperature(self, temperature=None, temperature_K=None):
        self.temp = units.kB * self._process_temperature(
            temperature, temperature_K, 'eV'
        )
        self.updatevars()

    def set_friction(self, friction):
        self.fr = friction
        self.updatevars()

    def set_timestep(self, timestep):
        self.dt = timestep
        self.updatevars()

    def updatevars(self):
        dt = self.dt
        T = self.temp
        beta = 1 / T
        fr = self.fr
        masses = self.masses

        self.c1 = (1 + (fr * dt) / 4)
        self.c2 = (1 - (fr * dt) / 4)
        self.c3 = np.sqrt((masses * fr * dt) / beta)

    def step(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces(md=True)

            # --- Étape O (Demi-pas de friction/bruit) ---
            xi = self.rng.standard_normal(size=(len(self.atoms), 3))
            self.comm.broadcast(xi, 0)
            p = self.atoms.get_momenta()
            p = (1 / self.c1) * (self.c2 * p + self.c3 * xi)

            # --- Étape B (Demi-kick de force) ---
            p = p + (self.dt / 2) * forces

            # --- Étape A (Drift : Mise à jour des positions) ---
            # Utilisation de p/m et non des forces !
            q = self.atoms.get_positions()
            q = q + self.dt * p / self.masses

            if self.fix_com:
                self.atoms.set_positions(q, apply_constraint=True)
            else:
                self.atoms.set_positions(q, apply_constraint=False)

            # Mise à jour des forces pour la suite
            forces = self.atoms.get_forces()

            # --- Étape B (Demi-kick de force) ---
            p = p + (self.dt / 2) * forces

            # --- Étape O (Demi-pas de friction/bruit) ---
            eta = self.rng.standard_normal(size=(len(self.atoms), 3))
            self.comm.broadcast(eta, 0)
            p = (1 / self.c1) * (self.c2 * p + self.c3 * eta)

            self.atoms.set_momenta(p, apply_constraint=False)
            return forces

    def _1st_half_step(self, forces):
        # --- Étape O (Demi-pas de friction/bruit) ---
        xi = self.rng.standard_normal(size=(len(self.atoms), 3))
        self.comm.broadcast(xi, 0)
        p = self.atoms.get_momenta()
        p = (1 / self.c1) * (self.c2 * p + self.c3 * xi)

        # --- Étape B (Demi-kick de force) ---
        p = p + (self.dt / 2) * forces

        # --- Étape A (Drift : Mise à jour des positions) ---
        # Utilisation de p/m et non des forces !
        q = self.atoms.get_positions()
        q = q + self.dt * p / self.masses

        if self.fix_com:
            self.atoms.set_positions(q, apply_constraint=True)
        else:
            self.atoms.set_positions(q, apply_constraint=False)
        self.atoms.calc.results['forces'] = np.zeros_like(forces)
        self.atoms.set_momenta(p, apply_constraint=False)

    def _2nd_half_step(self, forces):
        p = self.atoms.get_momenta()
        # --- Étape B (Demi-kick de force) ---
        p = p + (self.dt / 2) * forces

        # --- Étape O (Demi-pas de friction/bruit) ---
        eta = self.rng.standard_normal(size=(len(self.atoms), 3))
        self.comm.broadcast(eta, 0)
        p = (1 / self.c1) * (self.c2 * p + self.c3 * eta)
        self.atoms.set_momenta(p, apply_constraint=False)
        self.call_observers()
