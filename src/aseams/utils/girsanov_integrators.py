import warnings
from typing import Optional

import numpy as np

from ase import Atoms, units
from ase.md.md import MolecularDynamics
from ase.constraints import FixCom

class LangevinOBABO(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics.
    Computes the girsanov score (a) and Fischer matrix (A) if compute_girsanov is True
    """

    _lgv_version = 5

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature: Optional[float] = None,
        friction: Optional[float] = None,
        fixcm: bool = False,
        compute_girsanov: bool = False,
        *,
        temperature_K: Optional[float] = None,
        rng=None,
        **kwargs,
    ):
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
        self._constraints = []
        self.fix_com = fixcm
        self.compute_girsanov = compute_girsanov
        
        if fixcm:
            self.atoms.set_constraint(FixCom())
        self.updatevars()

        if self.compute_girsanov:
            if 'girsanov_a' not in self.atoms.info:
                self.atoms.info['girsanov_a'] = 0.0
            if 'girsanov_A' not in self.atoms.info:
                self.atoms.info['girsanov_A'] = 0.0

    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update(
            {
                'temperature_K': self.temp / units.kB,
                'friction': self.fr,
                'compute_girsanov': self.compute_girsanov,
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

    def _get_jacobian_3N(self):
        grad = self.atoms.calc.results['grad_descriptors']
        natoms = len(self.atoms)
        
        if grad.ndim == 3 and grad.shape[1] == natoms and grad.shape[2] == 3:
            return grad.transpose(1, 2, 0).reshape(natoms * 3, -1)
            
        if grad.shape[0] == natoms and grad.ndim == 3 and grad.shape[2] == 3:
            grad = grad.transpose(0, 2, 1)
            return grad.reshape(natoms * 3, -1)
            
        elif grad.shape[-1] == 3 * natoms:
            if grad.ndim == 3 and grad.shape[0] == 1:
                grad = grad.squeeze(0)
            
            if grad.shape[-1] == 3 * natoms:
                 return grad.T

        return grad.reshape(-1, grad.shape[-2]).T if grad.shape[-1] == 3*natoms else grad.reshape(natoms * 3, -1)

    def _update_girsanov(self, J_k, noise, step_type):
        natoms = len(self.atoms)
        noise_flat = noise.flatten()

        def to_3N(val, n_atoms):
            if np.isscalar(val):
                return np.full(3 * n_atoms, val)
            else:
                return np.repeat(np.ravel(val), 3)

        c3_3N = to_3N(self.c3, natoms)

        if step_type == 1:
            coeff = (self.c1 * self.dt) / (2.0 * c3_3N)
        elif step_type == 2:
            coeff = (self.c2 * self.dt) / (2.0 * c3_3N)
        else:
            raise ValueError()

        coeff_col = coeff[:, None]
        Y_noise = coeff_col * J_k

        delta_a = - (noise_flat @ Y_noise)
        delta_A = - (Y_noise.T @ Y_noise)
        if 'girsanov_a' not in self.atoms.info:
            self.atoms.info['girsanov_a'] = 0.0
        if 'girsanov_A' not in self.atoms.info:
            self.atoms.info['girsanov_A'] = 0.0
        self.atoms.info['girsanov_a'] += delta_a
        self.atoms.info['girsanov_A'] += delta_A.flatten()

    def step(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces(md=True)

        if self.compute_girsanov:
            J_k_1 = self._get_jacobian_3N()

        xi = self.rng.standard_normal(size=(len(self.atoms), 3))
        self.comm.broadcast(xi, 0)
        p = self.atoms.get_momenta()
        p = (1 / self.c1) * (self.c2 * p + self.c3 * xi)

        p = p + (self.dt / 2) * forces

        q = self.atoms.get_positions()
        q = q + self.dt * p / self.masses

        if self.fix_com:
            self.atoms.set_positions(q, apply_constraint=True)
        else:
            self.atoms.set_positions(q, apply_constraint=False)

        if self.compute_girsanov:
            self._update_girsanov(J_k_1, xi, step_type=1)

        forces = self.atoms.get_forces()
        
        if self.compute_girsanov:
            J_k_2 = self._get_jacobian_3N()

        p = p + (self.dt / 2) * forces

        eta = self.rng.standard_normal(size=(len(self.atoms), 3))
        self.comm.broadcast(eta, 0)
        p = (1 / self.c1) * (self.c2 * p + self.c3 * eta)

        self.atoms.set_momenta(p, apply_constraint=False)
        
        if self.compute_girsanov:
            self._update_girsanov(J_k_2, eta, step_type=2)

        return forces

    def _1st_half_step(self, forces):
        if self.compute_girsanov:
            J_k_1 = self._get_jacobian_3N()

        xi = self.rng.standard_normal(size=(len(self.atoms), 3))
        self.comm.broadcast(xi, 0)
        p = self.atoms.get_momenta()
        p = (1 / self.c1) * (self.c2 * p + self.c3 * xi)

        p = p + (self.dt / 2) * forces

        q = self.atoms.get_positions()
        q = q + self.dt * p / self.masses

        if self.fix_com:
            self.atoms.set_positions(q, apply_constraint=True)
        else:
            self.atoms.set_positions(q, apply_constraint=False)
            
        self.atoms.calc.results['forces'] = np.zeros_like(forces)
        self.atoms.set_momenta(p, apply_constraint=False)
        
        if self.compute_girsanov:
            self._update_girsanov(J_k_1, xi, step_type=1)

    def _2nd_half_step(self, forces):
        p = self.atoms.get_momenta()
        if self.compute_girsanov:
            J_k_2 = self._get_jacobian_3N()
        
        p = p + (self.dt / 2) * forces

        eta = self.rng.standard_normal(size=(len(self.atoms), 3))
        self.comm.broadcast(eta, 0)
        p = (1 / self.c1) * (self.c2 * p + self.c3 * eta)
        self.atoms.set_momenta(p, apply_constraint=False)
        
        if self.compute_girsanov:
            self._update_girsanov(J_k_2, eta, step_type=2)
        
        self.call_observers()



class OverdampedLangevin(MolecularDynamics):
    """Overdamped Langevin molecular dynamics.
    Computes the girsanov score (a) and Fischer matrix (A) if compute_girsanov is True
    """
    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature: Optional[float] = None,
        friction: Optional[float] = None,
        fixcm: bool = True,
        *,
        temperature_K: Optional[float] = None,
        rng=None,
        compute_girsanov: bool = False,
        **kwargs,
    ):
        if 'communicator' in kwargs:
            warnings.warn('', FutureWarning)
            kwargs['comm'] = kwargs.pop('communicator')

        if friction is None:
            raise TypeError()
        
        self.fr = friction
        self.temp = units.kB * self._process_temperature(
            temperature, temperature_K, 'eV'
        )
        self.fix_com = fixcm

        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
            
        MolecularDynamics.__init__(self, atoms, timestep, **kwargs)
        self.updatevars()
        self.compute_girsanov = compute_girsanov
        if self.compute_girsanov :
            if 'girsanov_a' not in self.atoms.info:
                self.atoms.info['girsanov_a'] = 0.0
            if 'girsanov_A' not in self.atoms.info:
                self.atoms.info['girsanov_A'] = 0.0

    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update(
            {
                'temperature_K': self.temp / units.kB,
                'friction': self.fr,
                'fixcm': self.fix_com,
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
        fr = self.fr
        masses = self.masses

        self.c1 = dt / (masses * fr)
        self.c2 = np.sqrt(2 * T * dt / (masses * fr))
        self.c3 = np.sqrt(dt / (2 * T * masses * fr))

    def _get_jacobian_3N(self):
        grad = self.atoms.calc.results['grad_descriptors']
        natoms = len(self.atoms)
        
        if grad.ndim == 3 and grad.shape[1] == natoms and grad.shape[2] == 3:
            return grad.transpose(1, 2, 0).reshape(natoms * 3, -1)
            
        if grad.shape[0] == natoms and grad.ndim == 3 and grad.shape[2] == 3:
            grad = grad.transpose(0, 2, 1)
            return grad.reshape(natoms * 3, -1)
            
        elif grad.shape[-1] == 3 * natoms:
            if grad.ndim == 3 and grad.shape[0] == 1:
                grad = grad.squeeze(0)
            
            if grad.shape[-1] == 3 * natoms:
                 return grad.T

        return grad.reshape(-1, grad.shape[-2]).T if grad.shape[-1] == 3*natoms else grad.reshape(natoms * 3, -1)
    def _update_girsanov(self, J_k, xi):
        natoms = len(self.atoms)
        xi_flat = xi.flatten()

        def to_3N(val, n_atoms):
            if np.isscalar(val):
                return np.full(3 * n_atoms, val)
            else:
                return np.repeat(val, 3)

        c3_3N = to_3N(self.c3, natoms)
        c3_col = c3_3N[:, None]

        Y_xi = c3_col * J_k

        delta_a = - (xi_flat @ Y_xi)
        delta_A = - (Y_xi.T @ Y_xi)

        if 'girsanov_a' not in self.atoms.info:
            self.atoms.info['girsanov_a'] = 0.0
        if 'girsanov_A' not in self.atoms.info:
            self.atoms.info['girsanov_A'] = 0.0

        self.atoms.info['girsanov_a'] += delta_a
        self.atoms.info['girsanov_A'] += delta_A.flatten()

    def step(self, forces=None):
        atoms = self.atoms
        natoms = len(atoms)

        if forces is None:
            forces = atoms.get_forces(md=True)

        J_k = self._get_jacobian_3N()

        xi = self.rng.standard_normal(size=(natoms, 3))

        for constraint in self.atoms.constraints:
            if hasattr(constraint, 'redistribute_forces_md'):
                constraint.redistribute_forces_md(atoms, xi, rand=True)

        self.comm.broadcast(xi, 0)

        rnd_pos = self.c2 * xi

        

        x = atoms.get_positions()
        
        dr = self.c1 * forces
        
        atoms.set_positions(x + dr + rnd_pos)

        self.v = (dr + rnd_pos) / self.dt
        atoms.set_momenta(self.v * self.masses)
        if self.compute_girsanov:
            self._update_girsanov(J_k, xi)

        return forces