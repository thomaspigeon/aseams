"""
Initial Condition Sampler Architecture
----------------------------------------------
Provides:
    - BaseInitialConditionSampler
    - MDDynamicSampler
    - SingleWalkerSampler
    - MultiWalkerSampler
    - FileBasedSampler
"""

import os, json, time, shutil
import numpy as np
import ase.units as units
from abc import ABC, abstractmethod
from ase.io import Trajectory, read, write
from ase.parallel import world, paropen, barrier
from ase.constraints import FixCom

from src.aseams.ams import NumpyEncoder


# =====================================================================
#  Base abstract class
# =====================================================================

class BaseInitialConditionSampler(ABC):
    """Abstract base class for all initial condition samplers."""

    def __init__(self, xi, cv_interval=1):
        if type(xi).__name__ != "CollectiveVariables":
            raise ValueError("xi must be a CollectiveVariables object")
        if not isinstance(cv_interval, int) or cv_interval < 0:
            raise ValueError("cv_interval must be an int >= 0")

        self.xi = xi
        self.cv_interval = cv_interval
        self.ini_cond_dir = None

        # shared times
        self.t_r_sigma = None
        self.t_sigma_r = None
        self.t_r_sigma_out = None
        self.t_sigma_out = None

    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds", clean=False):
        """Prepare directory for initial conditions."""
        self.ini_cond_dir = ini_cond_dir
        if world.rank == 0:
            if clean and os.path.exists(ini_cond_dir):
                shutil.rmtree(ini_cond_dir)
            os.makedirs(ini_cond_dir, exist_ok=True)

    def apply_rayleigh_bias(self, atoms, temp, bias_temp, resample_ortho=True, rng=None):
        """
        Apply Rayleigh velocity bias to the system's initial conditions
        using the reaction coordinate gradient implemented in the
        CollectiveVariables class (self.xi).

        Parameters
        ----------
        atoms : ase.Atoms
            The system whose velocities will be biased.
        temp : float
            Temperature in Kelvin (reference thermal temperature).
        bias_temp : float
            Temperature parameter for the Rayleigh bias distribution.
        resample_ortho : bool, optional
            Whether to resample orthogonal components from Maxwell–Boltzmann.
        rng : np.random.Generator, optional
            Random number generator (np.random.default_rng() by default).

        Returns
        -------
        biased_atoms : ase.Atoms
            Copy of `atoms` with biased velocities and stored reweighting factor.
        """
        if rng is None:
            rng = np.random.default_rng()

        # --- Check CollectiveVariables object ---
        if not hasattr(self.xi, "rc_grad") or self.xi.rc_grad is None:
            raise AttributeError(
                "CollectiveVariables object (self.xi) must have a callable 'rc_grad(atoms)' "
                "returning the reaction coordinate gradient (n_atoms, 3)."
            )

        # --- Compute and normalize the reaction coordinate gradient ---
        normal = np.array(self.xi.rc_grad(atoms), dtype=float)
        if normal.ndim != 2 or normal.shape[1] != 3:
            raise ValueError(f"Invalid rc_grad shape {normal.shape}; expected (n_atoms, 3).")

        masses = atoms.get_masses()[:, np.newaxis]  # shape (n_atoms, 1)
        normal_flat = normal.ravel()

        # Proper mass-weighted normalization (∑_i n_i·n_i/m_i = 1)
        norm = np.sqrt(np.dot(normal_flat, (normal / masses).ravel()))
        normal /= norm

        # --- Generate Rayleigh-distributed normal velocity magnitude ---
        normal_component = rng.rayleigh(scale=1.0)
        weight = (bias_temp / temp) * np.exp(-0.5 * (normal_component ** 2) * (bias_temp / temp - 1.0))

        # Convert to physical momentum magnitude
        normal_component *= np.sqrt(units.kB * bias_temp)

        # --- Initialize new momenta array ---
        new_momenta = np.zeros_like(atoms.get_momenta())

        # Normal contribution (mass-weighted)
        new_momenta += normal_component * normal

        # --- Orthogonal component ---
        if resample_ortho:
            # Draw random Gaussian orthogonal components
            ortho_momenta = rng.normal(size=(len(masses), 3)) * np.sqrt(masses * units.kB * temp)
        else:
            ortho_momenta = atoms.get_momenta().copy()

        # Remove projection along the normal direction
        proj = (normal * (ortho_momenta / masses)).sum()
        ortho_momenta -= normal * proj

        # Add orthogonal part
        new_momenta += ortho_momenta

        # --- Create output object ---
        biased_atoms = atoms.copy()
        biased_atoms.calc = atoms.calc
        biased_atoms.set_momenta(new_momenta)
        biased_atoms.info["weight"] = weight

        return biased_atoms

    def bias_initial_conditions(
        self,
        input_dir,
        output_dir,
        temp,
        bias_temp,
        resample_ortho=True,
        rng=None,
        overwrite=False,
    ):
        """
        Apply Rayleigh velocity bias to all initial conditions in a directory.

        Parameters
        ----------
        input_dir : str
            Path to the directory containing the original initial conditions (.extxyz files).
        output_dir : str
            Path where the biased initial conditions will be written.
        temp : float
            Reference temperature (in Kelvin).
        bias_temp : float
            Rayleigh biasing temperature (in Kelvin).
        resample_ortho : bool, optional
            Whether to resample orthogonal velocity components (default: True).
        rng : np.random.Generator, optional
            Random number generator (default: np.random.default_rng()).
        overwrite : bool, optional
            Whether to overwrite existing files in output_dir (default: False).

        Returns
        -------
        summary : dict
            Summary dictionary with keys:
                - "n_processed": number of files processed,
                - "weights": list of bias weights,
                - "output_files": list of generated file paths.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Check directories
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # Collect all .extxyz files
        input_files = sorted(
            [f for f in os.listdir(input_dir) if f.endswith(".extxyz")]
        )
        if not input_files:
            raise FileNotFoundError(f"No .extxyz files found in {input_dir}")

        weights = []
        output_files = []

        for fname in input_files:
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)

            if os.path.exists(out_path) and not overwrite:
                print(f"Skipping existing file: {out_path}")
                continue

            # Read Atoms object
            atoms = read(in_path)
            # Apply Rayleigh bias
            biased_atoms = self.apply_rayleigh_bias(
                atoms, temp=temp, bias_temp=bias_temp,
                resample_ortho=resample_ortho, rng=rng
            )

            # Save to output directory
            write(out_path, biased_atoms, format="extxyz")

            weights.append(biased_atoms.info.get("weight", np.nan))
            output_files.append(out_path)

        summary = {
            "n_processed": len(output_files),
            "weights": weights,
            "output_files": output_files,
        }

        return summary

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


# =====================================================================
#  Shared MD logic
# =====================================================================

class MDDynamicSampler(BaseInitialConditionSampler):
    """Base class for samplers that use ASE dynamics."""

    def __init__(self, dyn, xi, cv_interval=1, fixcm=True, rng=None):
        super().__init__(xi, cv_interval)
        self.dyn = dyn
        self.calc = dyn.atoms.calc
        self.fixcm = fixcm
        self.rng = np.random if rng is None else rng

        self.run_dir = None
        self.trajfile = None

        # tracking
        self.n_ini_conds_already = None
        self.first_in_r = False
        self.going_to_sigma = True
        self.going_back_to_r = False
        self.last_r_visited = None

    # --------------------------------------------------------------
    def _set_initialcond_dyn(self, atoms):
        """Reset MD atoms to given state."""
        self.dyn.atoms.set_scaled_positions(atoms.get_scaled_positions())
        self.dyn.atoms.set_momenta(atoms.get_momenta())
        self.dyn.atoms.calc.results['forces'] = atoms.get_forces(apply_constraint=False)
        self.dyn.atoms.calc.results['stress'] = atoms.get_stress(apply_constraint=False)
        self.dyn.atoms.calc.results['energy'] = atoms.get_potential_energy()

    def _write_checkpoint(self, filename):
        data = {
            "run_dir": self.run_dir,
            "ini_cond_dir": self.ini_cond_dir,
            "nsteps": self.dyn.nsteps,
            "cv_interval": self.cv_interval,
            "first_in_r": self.first_in_r,
            "last_r_visited": self.last_r_visited,
            "going_to_sigma": self.going_to_sigma,
            "going_back_to_r": self.going_back_to_r,
            "t_sigma_r": self.t_sigma_r,
            "t_r_sigma": self.t_r_sigma,
            "t_sigma_out": self.t_sigma_out,
            "t_r_sigma_out": self.t_r_sigma_out,
            "trajfile": self.trajfile
        }
        with paropen(filename, "w") as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)

    def _read_checkpoint(self, filename):
        with paropen(filename, "r") as f:
            data = json.load(f)
        self.run_dir = data["run_dir"]
        self.ini_cond_dir = data["ini_cond_dir"]
        self.trajfile = data["trajfile"]
        self.dyn.nsteps = data["nsteps"]
        self.cv_interval = data["cv_interval"]
        self.first_in_r = data["first_in_r"]
        self.last_r_visited = data["last_r_visited"]
        self.going_to_sigma = data["going_to_sigma"]
        self.going_back_to_r = data["going_back_to_r"]
        self.t_sigma_r = data["t_sigma_r"]
        self.t_r_sigma = data["t_r_sigma"]
        self.t_sigma_out = data["t_sigma_out"]
        self.t_r_sigma_out = data["t_r_sigma_out"]

    def _write_current_atoms(self, path="current_atoms.xyz"):
        write(path, self.dyn.atoms, format="extxyz")

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


# =====================================================================
#  Single walker sampler
# =====================================================================

class SingleWalkerSampler(MDDynamicSampler):
    """Sampler generating initial conditions with one MD walker."""

    def set_run_dir(self, run_dir="./ini_conds_md_logs", append_traj=False):
        self.run_dir = run_dir
        if not os.path.exists(run_dir) and world.rank == 0:
            os.mkdir(run_dir)
        n_traj = len([f for f in os.listdir(run_dir) if f.endswith(".traj")])
        if not append_traj:
            self.trajfile = f"{run_dir}/md_traj_{n_traj}.traj"
            traj = self.dyn.closelater(Trajectory(self.trajfile, "a", self.dyn.atoms,
                                                  properties=['energy', 'stress', 'forces']))
            self.dyn.attach(traj.write, interval=self.cv_interval)

    def sample(self, n_conditions=100, n_steps=None):
        """Replicates original InitialConditionsSampler.sample()."""
        if self.run_dir is None or self.ini_cond_dir is None:
            raise ValueError("Run dir or ini_cond_dir not set.")

        if n_steps is None:
            n_steps = -1
            if not isinstance(n_conditions, int) or n_conditions <= 0:
                raise ValueError("""n_conditions should be an int > 0 if n_steps is left to None""")
        elif n_conditions is None:
            n_conditions = -1
            if not isinstance(n_steps, int) or n_steps <= 0:
                raise ValueError("""n_steps should be an int > 0 if n_conditions is set to None""")

        n_cdt, n_stp = 0, 0
        if self.fixcm:
            self.dyn.fix_com = True
            self.dyn.atoms.set_constraint(FixCom())

        self.n_ini_conds_already = len([f for f in os.listdir(self.ini_cond_dir) if f.endswith(".extxyz")])
        if isinstance(self.xi.cv_r, list):
            ncv = len(self.xi.cv_r)
            self.t_r_sigma = [[] for _ in range(ncv)]
            self.t_sigma_r = [[] for _ in range(ncv)]
            self.t_r_sigma_out = [[] for _ in range(ncv)]
            self.t_sigma_out = [[] for _ in range(ncv)]
        else:
            self.t_r_sigma = [[]]
            self.t_sigma_r = [[]]
            self.t_r_sigma_out = [[]]
            self.t_sigma_out = [[]]

        self.first_in_r = False
        # Ensure we start in R
        while not self.first_in_r:
            self.dyn.run(self.cv_interval)
            n_stp += self.cv_interval
            if self.xi.is_out_of_r_zone(self.dyn.atoms):
                list_atoms = read(self.trajfile, index=":")
                at = list_atoms[self.rng.choice(len(list_atoms))]
                self._set_initialcond_dyn(at)
                self.dyn.call_observers()
            if self.xi.in_r(self.dyn.atoms):
                self.first_in_r = True

        # Main loop
        while (n_cdt < n_conditions) or (n_stp < n_steps):
            self.last_r_visited = np.where(self.xi.in_which_r(self.dyn.atoms)
                                           == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
            self.t_r_sigma[self.last_r_visited].append(0)
            self.t_sigma_r[self.last_r_visited].append(0)
            self.going_back_to_r, self.going_to_sigma = False, True

            while not self.xi.above_sigma(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                self.t_r_sigma[self.last_r_visited][-1] += self.cv_interval

            fname = f"{self.ini_cond_dir}/{self.n_ini_conds_already + n_cdt + 1}.extxyz"
            write(fname, self.dyn.atoms, format='extxyz')
            self.going_back_to_r, self.going_to_sigma = True, False
            self._write_checkpoint(f"{self.run_dir}/ini_checkpoint.txt")

            while not self.xi.in_r(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                if self.first_in_r:
                    self.t_sigma_r[self.last_r_visited][-1] += self.cv_interval
                if self.xi.is_out_of_r_zone(self.dyn.atoms):
                    self.first_in_r = False
                    list_atoms = read(self.trajfile, index=":")
                    at = list_atoms[self.rng.choice(range(len(list_atoms)))]
                    self._set_initialcond_dyn(at)
                    self.dyn.call_observers()
                    t_sigma_out = self.t_sigma_r[self.last_r_visited].pop(-1)
                    t_r_sigma_out = self.t_r_sigma[self.last_r_visited].pop(-1)
                    self.t_sigma_out[self.last_r_visited].append(t_sigma_out)
                    self.t_r_sigma_out[self.last_r_visited].append(t_r_sigma_out)
                if self.xi.in_r(self.dyn.atoms):
                    self.first_in_r = True

            n_cdt += 1
            self._write_checkpoint(f"{self.run_dir}/ini_checkpoint.txt")

    # --------------------------------------------------------------
    def sample_step_by_step(self, forces, energy, stress):
        """Incremental single-step sampling."""
        checkpoint = f"{self.run_dir}/ini_checkpoint.txt"
        if os.path.exists(checkpoint):
            self._read_checkpoint(checkpoint)
        else:
            self.dyn.nsteps = 0
            n_traj = len([f for f in os.listdir(self.run_dir) if f.endswith(".traj")])
            self.trajfile = f"{self.run_dir}/md_traj_{n_traj}.traj"

        traj = self.dyn.closelater(Trajectory(self.trajfile, "a", self.dyn.atoms,
                                              properties=['energy', 'stress', 'forces']))
        self.dyn.attach(traj.write, interval=self.cv_interval)
        barrier()

        if self.fixcm:
            self.dyn.fix_com = True
            self.dyn.atoms.set_constraint(FixCom())

        self.n_ini_conds_already = len(
            [ini for ini in os.listdir(self.ini_cond_dir) if ini.endswith("z")])
        if self.dyn.nsteps > 0:
            self.dyn.atoms.calc.results['forces'] = forces
            self.dyn.atoms.calc.results['stress'] = stress
            self.dyn.atoms.calc.results['energy'] = energy
            self.dyn._2nd_half_step(forces)
        else:
            self.dyn.call_observers()
            if isinstance(self.xi.cv_r, list):
                self.t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
                self.t_r_sigma_out = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_out = [[] for i in range(len(self.xi.cv_r))]
            else:
                self.t_r_sigma = [[]]
                self.t_sigma_r = [[]]
                self.t_r_sigma_out = [[]]
                self.t_sigma_out = [[]]
        in_r = self.xi.in_r(self.dyn.atoms)
        if in_r:
            self.last_r_visited = \
            np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
        out_of_r_zone = self.xi.is_out_of_r_zone(self.dyn.atoms)
        above_sigma = self.xi.above_sigma(self.dyn.atoms)
        if self.first_in_r:
            if self.going_to_sigma:
                self.t_r_sigma[self.last_r_visited][-1] += 1
            if self.going_back_to_r:
                self.t_sigma_r[self.last_r_visited][-1] += 1
            if above_sigma and self.going_to_sigma:
                self.going_to_sigma = False
                fname = self.ini_cond_dir + str(self.n_ini_conds_already + 1) + ".extxyz"
                write(fname, self.dyn.atoms, format='extxyz')
                self.going_back_to_r = True
            if in_r and self.going_back_to_r:
                self.going_back_to_r = False
                self.t_r_sigma[self.last_r_visited].append(0)
                self.t_sigma_r[self.last_r_visited].append(0)
                self.going_to_sigma = True
            if out_of_r_zone:
                list_atoms = read(self.trajfile, index=":")
                at = list_atoms[self.rng.choice(len(list_atoms))]
                self._set_initialcond_dyn(at)
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self.first_in_r = False
                self.going_to_sigma = True
                self.going_back_to_r = False
                self.dyn.nsteps = 0
                t_sigma_out = self.t_sigma_r[self.last_r_visited].pop(-1)
                t_r_sigma_out = self.t_r_sigma[self.last_r_visited].pop(-1)
                self.t_sigma_out[self.last_r_visited].append(t_sigma_out)
                self.t_r_sigma_out[self.last_r_visited].append(t_r_sigma_out)
                self._write_checkpoint(checkpoint)
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
            else:
                self.dyn._1st_half_step(forces)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self._write_checkpoint(checkpoint)
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
        else:
            if in_r:
                self.first_in_r = True
                self.t_r_sigma[self.last_r_visited].append(0)
                self.t_sigma_r[self.last_r_visited].append(0)
            if out_of_r_zone:
                list_atoms = read(self.trajfile, index=":")
                at = list_atoms[self.rng.choice(len(list_atoms))]
                self._set_initialcond_dyn(at)
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self.first_in_r = False
                self.going_to_sigma = True
                self.going_back_to_r = False
                self.dyn.nsteps = 0
                self._write_checkpoint(checkpoint)
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
            else:
                self.dyn._1st_half_step(forces)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self._write_checkpoint(checkpoint)
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False


# =====================================================================
#  Multi-walker (Fleming–Viot) sampler
# =====================================================================

class MultiWalkerSampler(MDDynamicSampler):
    """Fleming–Viot multi-replica initial condition sampler."""

    def __init__(self, dyn, xi, n_walkers=4, walker_index=0, **kwargs):
        super().__init__(dyn, xi, **kwargs)
        self.n_walkers = n_walkers
        self.w_i = walker_index

    def set_run_dir(self, run_dir="./ini_conds_walker_", append_traj=True):
        """Where the md logs will be written, if the directory does not exist, it will create it."""
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir + str(self.w_i)) and world.rank == 0:
            os.mkdir(self.run_dir + str(self.w_i))
        n_traj_already = len([fi for fi in os.listdir(self.run_dir + str(self.w_i)) if fi.endswith(".traj")])
        if not append_traj:
            self.trajfile = self.run_dir + str(self.w_i) + "/md_traj_{}.traj".format(n_traj_already)
            traj = self.dyn.closelater(Trajectory(filename=self.trajfile,
                                                  mode="a",
                                                  atoms=self.dyn.atoms,
                                                  properties=['energy', 'stress', 'forces']))
            self.dyn.attach(traj.write, interval=self.cv_interval)

    # --------------------------------------------------------------
    def _write_checkpoint(self, filename=None):
        """Override for multi-walker checkpoints."""
        data = {
            "run_dir": self.run_dir,
            "ini_cond_dir": self.ini_cond_dir,
            "nsteps": self.dyn.nsteps,
            "n_walkers": self.n_walkers,
            "w_i": self.w_i,
            "trajfile": self.trajfile,
            "cv_interval": self.cv_interval,
            "first_in_r": self.first_in_r,
            "last_r_visited": self.last_r_visited,
            "going_to_sigma": self.going_to_sigma,
            "going_back_to_r": self.going_back_to_r,
            "t_sigma_r": self.t_sigma_r,
            "t_r_sigma": self.t_r_sigma,
            "t_sigma_out": self.t_sigma_out,
            "t_r_sigma_out": self.t_r_sigma_out
        }
        fname = f"{self.run_dir}{self.w_i}/ini_fv_{self.w_i}_checkpoint.txt"
        with paropen(fname, "w") as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)

    def _read_checkpoint(self):
        fname = f"{self.run_dir}{self.w_i}/ini_fv_{self.w_i}_checkpoint.txt"
        with paropen(fname, "r") as f:
            data = json.load(f)
        self.run_dir = data["run_dir"]
        self.ini_cond_dir = data["ini_cond_dir"]
        self.dyn.nsteps = data["nsteps"]
        self.n_walkers = data["n_walkers"]
        self.w_i = data["w_i"]
        self.trajfile = data["trajfile"]
        self.cv_interval = data["cv_interval"]
        self.first_in_r = data["first_in_r"]
        self.going_to_sigma = data["going_to_sigma"]
        self.going_back_to_r = data["going_back_to_r"]
        self.last_r_visited = data["last_r_visited"]
        self.t_sigma_r = data["t_sigma_r"]
        self.t_r_sigma = data["t_r_sigma"]
        self.t_sigma_out = data["t_sigma_out"]
        self.t_r_sigma_out = data["t_r_sigma_out"]

    # --------------------------------------------------------------
    def _branch_fv_particle(self):
        traj_idx = None
        current_time = 0
        n_traj_already = len([fi for fi in os.listdir(self.run_dir + str(self.w_i)) if fi.endswith(".traj")])
        for i in range(n_traj_already):
            trajfile = self.run_dir + str(self.w_i) + "/md_traj_{}.traj".format(i)
            list_atoms = read(trajfile, index=":")
            current_time += len(list_atoms)
            del list_atoms
        branch_rep_number = self.rng.choice(np.setdiff1d(np.arange(self.n_walkers), self.w_i))
        while traj_idx is None:
            n_traj_already = len(
                [fi for fi in os.listdir(self.run_dir + str(branch_rep_number)) if fi.endswith(".traj")])
            t = []
            for i in range(n_traj_already):
                trajfile = self.run_dir + str(branch_rep_number) + "/md_traj_{}.traj".format(i)
                list_atoms = read(trajfile, index=":")
                if i == 0:
                    t.append(len(list_atoms))
                    if current_time < t[-1]:
                        traj_idx = i
                else:
                    t.append(len(list_atoms) + t[-1])
                    if t[-2] <= current_time < t[-1]:
                        traj_idx = i
            f = paropen(self.run_dir + str(self.w_i) + "/FV_history.txt", "a")
            f.write("Attempt branching replica " + str(self.w_i) + " from replica index: " + str(
                branch_rep_number) + " at time " + str(current_time) + " \n")
            if traj_idx is not None:
                f.write("Succesful branching\n \n")
                f.close()
            else:
                f.write("Cannot branch as replica " + str(branch_rep_number) + "  did not reach time " + str(
                    current_time) + "\n")
                f.write("Going to sleep for 1 minute\n")
                f.close()
                time.sleep(60)
        trajfile = self.run_dir + str(branch_rep_number) + "/md_traj_{}.traj".format(traj_idx)
        list_atoms = read(trajfile, index=":", format='traj')
        if traj_idx == 0:
            atoms = list_atoms[current_time]
        else:
            atoms = list_atoms[current_time - t[traj_idx - 1]]
        self._set_initialcond_dyn(atoms)
        return atoms
    # --------------------------------------------------------------

    def sample(self, n_conditions=100, n_steps=None):
        """Sampling initial conditions

        Parameters:

        n_conditions: int
            Number of initial conditions to sample, default is 100. If this argument is given, n_steps should be kept to
            None
        n_steps: int
            Number of integration time steps that should be performed. If this argument is given, n_conditions should be
            set to None
        """
        if self.run_dir is None:
            raise ValueError("""The running directory is not set ! Call initialconditionssampler.set_run_dir""")
        if self.ini_cond_dir is None:
            raise ValueError(
                """The directory to store initial conditions is not defined ! 
                Call initialconditionssampler.set_ini_cond_dir""")
        while not os.path.exists(self.ini_cond_dir):
            time.sleep(1)
        if n_steps is None:
            n_steps = -1
            if not isinstance(n_conditions, int) or n_conditions <= 0:
                raise ValueError("""n_conditions should be an int > 0 if n_steps is left to None""")
        elif n_conditions is None:
            n_conditions = -1
            if not isinstance(n_steps, int) or n_steps <= 0:
                raise ValueError("""n_steps should be an int > 0 if n_conditions is set to None""")
        else:
            raise ValueError(
                """n_conditions should be an int > 0 if n_steps is left to None or n_steps should be an
                                int > 0 if n_conditions is set to None"""
            )
        n_cdt, n_stp = 0, 0
        if self.fixcm:
            self.dyn.atoms.set_constraint(FixCom())
            self.dyn.fix_com = True
        n_ini_conds_already = len(
            [fi for fi in os.listdir(self.ini_cond_dir) if fi.startswith("walker_" + str(self.w_i) + "_")])
        if isinstance(self.xi.cv_r, list):
            if self.t_r_sigma is None:
                self.t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
                self.t_r_sigma_out = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_out = [[] for i in range(len(self.xi.cv_r))]
        else:
            if self.t_r_sigma is None:
                self.t_r_sigma = [[]]
                self.t_sigma_r = [[]]
                self.t_r_sigma_out = [[]]
                self.t_sigma_out = [[]]
        while not self.first_in_r:
            self.dyn.run(self.cv_interval)
            n_stp += self.cv_interval
            if self.xi.is_out_of_r_zone(self.dyn.atoms):
                _ = self._branch_fv_particle()
                self.first_in_r = False
            if self.xi.in_r(self.dyn.atoms):
                self.first_in_r = True
        while n_cdt < n_conditions or n_stp < n_steps:
            self.last_r_visited = \
                np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
            self.going_back_to_r = False
            self.going_to_sigma = True
            self.t_r_sigma[self.last_r_visited].append(0)
            self.t_sigma_r[self.last_r_visited].append(0)
            while not self.xi.above_sigma(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                self.t_r_sigma[self.last_r_visited][-1] += self.cv_interval
            fname = self.ini_cond_dir + "/walker_" + str(self.w_i) + '_ini_cond_' + str(
                n_ini_conds_already + n_cdt + 1) + ".extxyz"
            write(fname, self.dyn.atoms, format='extxyz')
            self.going_back_to_r = True
            self.going_to_sigma = False
            self._write_checkpoint()
            while not self.xi.in_r(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                if self.first_in_r:
                    self.t_sigma_r[self.last_r_visited][-1] += self.cv_interval
                if self.xi.is_out_of_r_zone(self.dyn.atoms):
                    _ = self._branch_fv_particle()
                    if self.first_in_r:
                        t_sigma_out = self.t_sigma_r[self.last_r_visited].pop(-1)
                        t_r_sigma_out = self.t_r_sigma[self.last_r_visited].pop(-1)
                        self.t_sigma_out[self.last_r_visited].append(t_sigma_out)
                        self.t_r_sigma_out[self.last_r_visited].append(t_r_sigma_out)
                    self.first_in_r = False
                if self.xi.in_r(self.dyn.atoms):
                    self.first_in_r = True
            n_cdt += 1
            self._write_checkpoint()

    def sample_step_by_step(self, forces, energy, stress):
        """Run sampling of ini-conds calling this function steps by steps"""
        if os.path.exists(self.run_dir + str(self.w_i) + "/ini_fv_" + str(self.w_i) + "_checkpoint.txt"):
            self._read_checkpoint()
        else:
            self.dyn.nsteps = 0
            n_traj_already = len([fi for fi in os.listdir(self.run_dir + str(self.w_i)) if fi.endswith(".traj")])
            self.trajfile = self.run_dir + str(self.w_i) + "/md_traj_{}.traj".format(n_traj_already)
        traj = self.dyn.closelater(Trajectory(filename=self.trajfile,
                                              mode="a",
                                              atoms=self.dyn.atoms,
                                              properties=['energy', 'stress', 'forces']))
        self.dyn.attach(traj.write, interval=self.cv_interval)
        barrier()
        if self.fixcm:
            self.dyn.fix_com = True
            self.dyn.atoms.set_constraint(FixCom())
        self.n_ini_conds_already = len(
            [fi for fi in os.listdir(self.ini_cond_dir) if fi.startswith("walker_" + str(self.w_i) + "_")])
        if self.dyn.nsteps > 0:
            self.dyn.atoms.calc.results['forces'] = forces
            self.dyn.atoms.calc.results['stress'] = stress
            self.dyn.atoms.calc.results['energy'] = energy
            self.dyn._2nd_half_step(forces)
        else:
            self.dyn.call_observers()
            if isinstance(self.xi.cv_r, list):
                self.t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
                self.t_r_sigma_out = [[] for i in range(len(self.xi.cv_r))]
                self.t_sigma_out = [[] for i in range(len(self.xi.cv_r))]
            else:
                self.t_r_sigma = [[]]
                self.t_sigma_r = [[]]
                self.t_r_sigma_out = [[]]
                self.t_sigma_out = [[]]
        in_r = self.xi.in_r(self.dyn.atoms)
        if in_r:
            self.last_r_visited = \
                np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
        out_of_r_zone = self.xi.is_out_of_r_zone(self.dyn.atoms)
        above_sigma = self.xi.above_sigma(self.dyn.atoms)
        if self.first_in_r:
            if self.going_to_sigma:
                self.t_r_sigma[self.last_r_visited][-1] += 1
            if self.going_back_to_r:
                self.t_sigma_r[self.last_r_visited][-1] += 1
            if above_sigma and self.going_to_sigma:
                self.going_to_sigma = False
                fname = self.ini_cond_dir + "/walker_" + str(self.w_i) + '_ini_cond_' + str(
                    self.n_ini_conds_already + 1) + ".extxyz"
                write(fname, self.dyn.atoms, format='extxyz')
                self.going_back_to_r = True
            if in_r and self.going_back_to_r:
                self.going_back_to_r = False
                self.t_r_sigma[self.last_r_visited].append(0)
                self.t_sigma_r[self.last_r_visited].append(0)
                self.going_to_sigma = True
            if out_of_r_zone:
                at = self._branch_fv_particle()
                self._set_initialcond_dyn(at)
                self.dyn.nsteps = 0
                self.first_in_r = False
                self.going_to_sigma = True
                self.going_back_to_r = False
                t_sigma_out = self.t_sigma_r[self.last_r_visited].pop(-1)
                t_r_sigma_out = self.t_r_sigma[self.last_r_visited].pop(-1)
                self.t_sigma_out[self.last_r_visited].append(t_sigma_out)
                self.t_r_sigma_out[self.last_r_visited].append(t_r_sigma_out)
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
            else:
                self.dyn._1st_half_step(forces)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
        else:
            if in_r:
                self.first_in_r = True
                self.t_r_sigma[self.last_r_visited].append(0)
                self.t_sigma_r[self.last_r_visited].append(0)
            if out_of_r_zone:
                at = self._branch_fv_particle()
                self._set_initialcond_dyn(at)
                self.dyn.nsteps = 0
                self.first_in_r = False
                self.going_to_sigma = True
                self.going_back_to_r = False
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False
            else:
                self.dyn._1st_half_step(forces)
                self.dyn.nsteps += 1
                self.dyn.atoms.calc.results['forces'] = np.zeros_like(forces)
                self.dyn.atoms.calc.results['stress'] = np.zeros_like(stress)
                self.dyn.atoms.calc.results['energy'] = np.zeros_like(energy)
                self._write_checkpoint()
                self._write_current_atoms()
                self.dyn.observers.pop(-1)
                return False


# =====================================================================
#  File-based sampler
# =====================================================================

class FileBasedSampler(BaseInitialConditionSampler):
    """Sampler extracting initial conditions from existing trajectory files."""

    def sample(self, file):
        """Equivalent of InitialConditionsSamplerFromFile.sample()."""
        if self.ini_cond_dir is None:
            raise ValueError("ini_cond_dir not set.")

        if isinstance(self.xi.cv_r, list):
            ncv = len(self.xi.cv_r)
            t_r_sigma = [[] for _ in range(ncv)]
            t_sigma_r = [[] for _ in range(ncv)]
        else:
            t_r_sigma = [[]]
            t_sigma_r = [[]]

        n_cdt, n_stp = 0, 0
        n_ini = len([f for f in os.listdir(self.ini_cond_dir) if f.endswith(".extxyz")])
        traj = read(file, index=":")
        n_steps = len(traj)

        while not self.xi.in_r(traj[n_stp]) and n_stp < n_steps:
            n_stp += self.cv_interval

        while n_stp < n_steps:
            which_r = np.where(self.xi.in_which_r(traj[n_stp])
                               == np.max(self.xi.in_which_r(traj[n_stp])))[0][0]
            t_r_sigma[which_r].append(0)
            while not self.xi.above_sigma(traj[n_stp]) and n_stp < n_steps:
                n_stp += self.cv_interval
                if n_stp >= n_steps:
                    t_r_sigma[which_r].pop()
                    break
                t_r_sigma[which_r][-1] += self.cv_interval

            if n_stp >= n_steps: break
            fname = f"{self.ini_cond_dir}/{n_ini + n_cdt + 1}.extxyz"
            write(fname, traj[n_stp], format='extxyz')
            t_sigma_r[which_r].append(0)

            while not self.xi.in_r(traj[n_stp]) and n_stp < n_steps:
                n_stp += self.cv_interval
                if n_stp >= n_steps:
                    t_sigma_r[which_r].pop()
                    break
                t_sigma_r[which_r][-1] += self.cv_interval
            n_cdt += 1
        return t_r_sigma, t_sigma_r
