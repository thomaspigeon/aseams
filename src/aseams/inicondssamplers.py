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

import inspect
import os, json, time, shutil
import numpy as np
import math
import ase.units as units
from scipy.optimize import newton
from scipy.stats import norm
from abc import ABC, abstractmethod
from ase.io import Trajectory, read, write
from ase.parallel import world, paropen, barrier
from ase.constraints import FixCom

from .ams import NumpyEncoder


def sample_rayleigh(sigma, rng=None):
    """
    Sample from a Rayleigh distribution: p(v) = (v/sigma^2) * exp(-v^2 / (2*sigma^2))

    Parameters:
    -----------
    sigma : float
        The scale parameter of the Rayleigh distribution (sqrt(kT/m)).
    rng : numpy.random.Generator, optional
        Random number generator.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Inverse CDF method for Rayleigh: v = sigma * sqrt(-2 * ln(1 - U))
    u = rng.uniform(1e-10, 1.0 - 1e-10)
    return sigma * math.sqrt(-2.0 * math.log(1.0 - u))


def _get_v_r_constants(u_R, sigma_R):
    """
    Calculate the P(0) and Z_star constants required for the CDF inversion.

    Parameters:
    -----------
    u_R : float
        The projection of the bias velocity shift (u_alpha) onto the normal vector e_R.
    sigma_R : float
        The standard deviation of the thermal velocity component along the normal e_R.

    Returns:
    --------
    p_0 : float
        The value of the primitive function P(t) at t=0.
    z_star : float
        The normalization constant (partition function) for the biased flux distribution.
    """
    # Using scipy.stats.norm.cdf for the standard normal cumulative distribution function (Φ)
    prefix = math.sqrt(2 * math.pi) * sigma_R * u_R

    # P(0) calculation
    term1_0 = -(sigma_R**2) * math.exp(-(u_R**2) / (2 * sigma_R**2))
    term2_0 = prefix * norm.cdf(-u_R / sigma_R)
    p_0 = term1_0 + term2_0

    # Z_star = P(inf) - P(0) where P(inf) = prefix
    z_star = prefix - p_0
    return p_0, z_star


def sample_biased_v_r(u_R, sigma_R, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # --- SÉCURITÉ 3 : Fallback (inchangé) ---
    if u_R / sigma_R > 3.0:
        return max(1e-12, rng.normal(u_R, sigma_R))

    U = rng.uniform(1e-10, 1.0 - 1e-10)
    p_0, z_star = _get_v_r_constants(u_R, sigma_R)
    target = p_0 + U * z_star
    prefix = math.sqrt(2 * math.pi) * sigma_R * u_R

    def G(r):
        diff = (r - u_R) / sigma_R
        # Protection contre l'overflow de diff**2
        if diff < -30: return -target
        if diff > 30:  return prefix - target

        p_r = -(sigma_R ** 2) * math.exp(-0.5 * diff ** 2) + prefix * norm.cdf(diff)

        return p_r - target

    def G_prime(r):
        diff = (r - u_R) / sigma_R
        # Si on est trop loin, la dérivée est nulle
        if abs(diff) > 30: return 0.0
        return r * math.exp(-0.5 * diff ** 2)

    x0 = max(1e-3, u_R + sigma_R)

    try:
        # On ignore localement les warnings numpy pour le solver
        with np.errstate(all='ignore'):
            r_root = newton(func=G, x0=x0, fprime=G_prime, tol=1.48e-08, maxiter=50)
    except (RuntimeError, ZeroDivisionError):
        # Si Newton échoue (ex: dérivée nulle), brentq est infaillible pour les fonctions monotones
        from scipy.optimize import brentq
        r_root = brentq(G, 0, max(10.0, u_R + 10 * sigma_R))

    return max(1e-12, r_root)


# =====================================================================
#  Base abstract class
# =====================================================================


class BaseInitialConditionSampler(ABC):
    """Abstract base class for all initial condition samplers."""

    def __init__(self, xi, cv_interval=1):
        """
        Initialize the sampler with collective variables and recording interval.

        Parameters:
        -----------
        xi : CollectiveVariables
            Object containing the reaction coordinate and boundary definitions.
        cv_interval : int
            Interval (in steps) at which collective variables are evaluated.
        """
        if type(xi).__name__ != "CollectiveVariables":
            raise ValueError("xi must be a CollectiveVariables object")
        if not isinstance(cv_interval, int) or cv_interval < 0:
            raise ValueError("cv_interval must be an int >= 0")

        self.xi = xi
        self.cv_interval = cv_interval
        self.ini_cond_dir = None

        # Shared residence times tracking
        self.t_r_sigma = None
        self.t_sigma_r = None
        self.t_r_sigma_out = None
        self.t_sigma_out = None

    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds", clean=False):
        """
        Prepare the directory where generated initial conditions will be stored.

        Parameters:
        -----------
        ini_cond_dir : str
            Path to the directory.
        clean : bool
            If True, removes the directory if it already exists.
        """
        self.ini_cond_dir = ini_cond_dir
        if world.rank == 0:
            if clean and os.path.exists(ini_cond_dir):
                shutil.rmtree(ini_cond_dir)
            os.makedirs(ini_cond_dir, exist_ok=True)

    def bias_one_initial_condition_rayleigh(self, atoms, temp_phys, temp_bias, rng=None):
        """
        Generate a biased initial condition by sampling the normal velocity component
        from a Rayleigh distribution at a higher temperature (temp_bias).

        Parameters:
        -----------
        atoms : ase.Atoms
            The configuration to bias.
        temp_phys : float
            The physical temperature of the system (Kelvin).
        temp_bias : float
            The increased temperature for the escape velocity (Kelvin).
        rng : np.random.Generator, optional
            Random number generator.
        """
        if rng is None:
            rng = np.random.default_rng()

        # --- 1. Identify exit normal e_R ---
        if "from_which_r" in atoms.info:
            if inspect.isfunction(self.xi.cv_r):
                in_r_mask = [True]
            else:
                in_r_mask = [False for _ in self.xi.cv_r]
                in_r_mask[atoms.info["from_which_r"]] = True
        else:
            raise ValueError(
                """No "from_which_r" metadata in initial condition file, problem in sampler or cvs. 
            There must be a problem in you sampler of definition of collective variables"""
            )
        if not any(in_r_mask):
            raise ValueError(
                """System is not in any defined state R.
            There must be a problem in you sampler of definition of collective variables"""
            )

        idx = np.where(in_r_mask)[0][0]
        grad_r_func = self.xi.cv_r_grad[idx]
        raw_grad_r = grad_r_func(atoms).flatten()

        if inspect.isfunction(self.xi.cv_r):
            condition = self.xi.r_crit
            val_threshold = self.xi.in_r_boundary
            current_val = self.xi.evaluate_cv_r(atoms)
        elif isinstance(self.xi.cv_r, list):
            condition = self.xi.r_crit[idx]
            val_threshold = self.xi.in_r_boundary[idx]
            current_val = self.xi.evaluate_cv_r(atoms)[idx]
        else:
            raise ValueError("""Problem of CollectiveVariables definition""")

        # Select the outward normal direction

        if condition == "below":
            e_R = raw_grad_r
        elif condition == "above":
            e_R = -raw_grad_r
        elif condition == "between":
            v_min, v_max = val_threshold
            g_R = raw_grad_r if abs(current_val - v_max) < abs(current_val - v_min) else -raw_grad_r
        else:
            g_R = raw_grad_r

        # --- 2. Physics Parameters (CORRIGÉ) ---
        masses = atoms.get_masses()
        m_3n = np.repeat(masses, 3)

        # Calcul correct de la variance projetée (Moyenne Harmonique)
        inv_m_eff = np.sum((g_R ** 2) / m_3n)

        # Sigmas basés sur la masse effective correcte
        sigma_phys = math.sqrt(units.kB * temp_phys * inv_m_eff)
        sigma_bias = math.sqrt(units.kB * temp_bias * inv_m_eff)

        # --- 3. Sampling ---
        # Normal component: Rayleigh at biased temperature
        v_R_sampled = sample_rayleigh(sigma_bias, rng=rng)
        w_R = (g_R / m_3n) / inv_m_eff

        # Tangential components: standard Maxwell-Boltzmann at physical temperature
        v_thermal_3n = np.sqrt(units.kB * temp_phys / m_3n)
        v_full_MB = rng.normal(0, v_thermal_3n)

        # Remove any normal component from the thermal MB draw and add our biased v_R
        v_perp = v_full_MB - np.dot(v_full_MB, g_R) * w_R
        v_final_flat = v_R_sampled * w_R + v_perp

        m_eff_R = 1.0 / inv_m_eff  # Masse effective réelle
        temp_ratio = temp_bias / temp_phys
        energy_factor = (m_eff_R * v_R_sampled**2) / (2.0 * units.kB)
        diff_beta = (1.0 / temp_phys) - (1.0 / temp_bias)

        weight = temp_ratio * math.exp(-energy_factor * diff_beta)

        # --- 5. Finalize Atoms object ---
        atoms.set_velocities(v_final_flat.reshape((-1, 3)), apply_constraint=True)
        if not hasattr(atoms, 'info'):
            atoms.info = {}
        atoms.info["weight"] = weight

        return atoms

    def bias_one_initial_condition_flux(self, atoms, alpha, temp, rng=None):
        """
        Generate a biased velocity vector for a single configuration and compute the
        associated Importance Sampling (IS) weight.

        Parameters:
        -----------
        atoms : ase.Atoms
            The atomic configuration (positions) to be biased.
        alpha : float
            Dimensionless bias parameter (alpha=0 corresponds to unbiased flux).
        temp : float
            Simulation temperature in Kelvin.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns:
        --------
        atoms : ase.Atoms
            The configuration with updated biased velocities and the IS weight in atoms.info.
        """
        if rng is None:
            rng = np.random.default_rng()

        # --- Check CollectiveVariables object ---
        if not hasattr(self.xi, "rc_grad") or self.xi.rc_grad is None:
            raise AttributeError("CollectiveVariables object (self.xi) must have a callable 'rc_grad(atoms)' " "returning the reaction coordinate gradient (n_atoms, 3).")
        if not hasattr(self.xi, "cv_r_grad") or self.xi.cv_r_grad is None:
            raise AttributeError("CollectiveVariables object (self.xi) must have a callable 'cv_r_grad(atoms)' " "returning the gradient of the cv_r function(s) (n_atoms, 3).")

        beta = 1 / (units.kB * temp)
        masses = atoms.get_masses()
        m_3n = np.repeat(masses, 3)  # Mass vector flattened to 3N

        # 1. Geometrical Directions Calculation
        # Identify the index of the current state R
        if "from_which_r" in atoms.info:
            if inspect.isfunction(self.xi.cv_r):
                in_r_mask = [True]
            else:
                in_r_mask = [False for _ in self.xi.cv_r]
                in_r_mask[atoms.info["from_which_r"]] = True
        else:
            raise ValueError(
                """No "from_which_r" metadata in initial condition file, problem in sampler or cvs. 
            There must be a problem in you sampler of definition of collective variables"""
            )
        if not any(in_r_mask):
            raise ValueError(
                """System is not in any defined state R.
            There must be a problem in you sampler of definition of collective variables"""
            )

        idx = np.where(in_r_mask)[0][0]

        # Retrieve the gradient for the corresponding cv_r function
        grad_r_func = self.xi.cv_r_grad[idx]
        raw_grad_r = grad_r_func(atoms).flatten()

        # Determine the direction of the normal e_R (must be outward from R)
        if inspect.isfunction(self.xi.cv_r):
            condition = self.xi.r_crit
            val_threshold = self.xi.in_r_boundary
            current_val = self.xi.evaluate_cv_r(atoms)
        elif isinstance(self.xi.cv_r, list):
            condition = self.xi.r_crit[idx]
            val_threshold = self.xi.in_r_boundary[idx]
            current_val = self.xi.evaluate_cv_r(atoms)[idx]
        else:
            raise ValueError("""Problem of CollectiveVariables definition""")

        if condition == "below":
            # R = {q | zeta <= val}. Exiting means increasing zeta. e_R = +grad(zeta)
            g_R = raw_grad_r
        elif condition == 'above':
            # R = {q | zeta >= val}. Exiting means decreasing zeta. e_R = -grad(zeta)
            g_R = -raw_grad_r
        elif condition == 'between':
            # R = {q | v_min <= zeta <= v_max}.
            # We check which boundary is closer to define the exit face
            v_min, v_max = val_threshold
            if abs(current_val - v_max) < abs(current_val - v_min):
                g_R = raw_grad_r  # Exiting through the upper bound
            else:
                g_R = -raw_grad_r  # Exiting through the lower bound
        else:
            g_R = raw_grad_r
        # Bias Direction (Reaction Coordinate)
        g_xi = self.xi.rc_grad(atoms).flatten()

        # 2. Mass and Variance Parameters (CORRIGÉ)
        # Calcul correct des masses effectives inverses
        inv_m_eff_xi = max(np.sum((g_xi ** 2) / m_3n), 1e-14)
        inv_m_eff_R = max(np.sum((g_R ** 2) / m_3n), 1e-14)

        # Sigmas corrects
        sigma_R = math.sqrt(units.kB * temp * inv_m_eff_R)
        # Paramètres pour la direction de biais xi
        # v_thermal_xi est l'écart type de la vitesse thermique selon xi
        v_thermal_xi = math.sqrt(units.kB * temp * inv_m_eff_xi)

        # 3. Define the Bias Shift Vector u_alpha
        u_direction = (g_xi / m_3n) / inv_m_eff_xi
        u_alpha = (alpha * v_thermal_xi) * u_direction
        u_R_raw = np.dot(u_alpha, g_R)
        # --- SÉCURITÉ 2 : Clamping ---
        # Théoriquement, u_R / sigma_R = alpha * cos(theta).
        # On force ce ratio à rester dans les bornes [-alpha, alpha]
        u_R = np.clip(u_R_raw, -alpha * sigma_R, alpha * sigma_R)

        # 4. Normal Velocity Sampling (v_R)
        v_R_sampled = sample_biased_v_r(u_R, sigma_R, rng=rng)

        # 5. Tangential Components Sampling (Shifted Gaussian)
        # Generate thermal noise shifted by u_alpha
        w_R = (g_R / m_3n) / inv_m_eff_R
        v_thermal_3n = 1.0 / np.sqrt(beta * m_3n)
        v_full = u_alpha + rng.normal(0, v_thermal_3n)

        # Projection to ensure v_R_sampled is preserved on the normal axis
        v_perp = v_full - np.dot(v_full, g_R) * w_R
        v_final_flat = v_R_sampled * w_R + v_perp

        # 6. Weight Calculation W(v)
        p_0, z_star = _get_v_r_constants(u_R, sigma_R)
        R_Z = z_star / (sigma_R ** 2)
        v_dot_xi = np.dot(v_final_flat, g_xi)
        arg_exp = -alpha * (v_dot_xi / v_thermal_xi) + (alpha ** 2 / 2.0)
        log_weight = math.log(R_Z) + arg_exp
        # On limite le poids pour éviter l'OverflowError final
        if log_weight > 700:
            weight = 1e308  # Valeur maximale représentable
            print(f"Warning: Weight capped for alpha={alpha} (log_w={log_weight:.2f})")
        else:
            weight = math.exp(log_weight)

        # Update the Atoms object
        atoms.set_velocities(v_final_flat.reshape((-1, 3)), apply_constraint=True)
        if not hasattr(atoms, "info"):
            atoms.info = {}
        atoms.info["weight"] = weight

        return atoms

    def bias_initial_conditions(
        self,
        input_dir,
        output_dir,
        temp,
        alpha=0.0,
        temp_bias=None,
        method="flux",
        rng=None,
        overwrite=False,
    ):
        """
        Apply velocity biasing to all initial conditions in a directory using either
        Flux Biasing or Rayleigh Biasing.

        Parameters:
        -----------
        input_dir : str
            Path to the directory containing the original .extxyz configurations.
        output_dir : str
            Path where the biased configurations will be written.
        temp : float
            The physical reference temperature (Kelvin).
        alpha : float, optional
            Dimensionless bias parameter for 'flux' method. Defaults to 0.0.
        temp_bias : float, optional
            The increased temperature (Kelvin) for the 'rayleigh' method.
            Required if method='rayleigh'.
        method : str
            The biasing scheme to use: 'flux' or 'rayleigh'. Defaults to 'flux'.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        overwrite : bool, optional
            If True, existing files in output_dir will be overwritten.

        Returns:
        --------
        summary : dict
            A dictionary containing processing statistics, weights, and file paths.
        """
        # 1. Validation and Directory Setup
        if method not in ["flux", "rayleigh"]:
            raise ValueError("method must be either 'flux' or 'rayleigh'")

        if method == "rayleigh" and temp_bias is None:
            raise ValueError("temp_bias must be provided when using method='rayleigh'")

        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        if world.rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        barrier()

        # 2. File Collection
        input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".extxyz")])
        if not input_files:
            raise FileNotFoundError(f"No .extxyz files found in {input_dir}")

        weights = []
        output_files = []

        # 3. Processing Loop
        for fname in input_files:
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)

            if os.path.exists(out_path) and not overwrite:
                if world.rank == 0:
                    print(f"Skipping existing file: {out_path}")
                continue

            atoms = read(in_path)

            # Select the biasing method
            if method == "flux":
                # Uses the Newton-Raphson scheme with shift alpha
                biased_atoms = self.bias_one_initial_condition_flux(atoms, alpha=alpha, temp=temp, rng=rng)
            else:
                # Uses the Rayleigh distribution at temp_bias
                biased_atoms = self.bias_one_initial_condition_rayleigh(atoms, temp_phys=temp, temp_bias=temp_bias, rng=rng)

            # Save to output directory
            write(out_path, biased_atoms, format="extxyz")

            weights.append(biased_atoms.info.get("weight", np.nan))
            output_files.append(out_path)

        summary = {
            "n_processed": len(output_files),
            "method_used": method,
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
        if self.fixcm:
            self.dyn.atoms.set_scaled_positions(atoms.get_scaled_positions(apply_constraint=True))
        else:
            self.dyn.atoms.set_scaled_positions(atoms.get_scaled_positions(apply_constraint=False))
        self.dyn.atoms.set_momenta(atoms.get_momenta(apply_constraint=False))
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
            "trajfile": self.trajfile,
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
            traj = self.dyn.closelater(Trajectory(self.trajfile, "a", self.dyn.atoms, properties=["energy", "stress", "forces"]))
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
            self.dyn.atoms._constraints = []
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
            self.last_r_visited = np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
            self.t_r_sigma[self.last_r_visited].append(0)
            self.t_sigma_r[self.last_r_visited].append(0)
            self.going_back_to_r, self.going_to_sigma = False, True
            valid_exit = False
            while not (self.xi.above_sigma(self.dyn.atoms) and valid_exit):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                self.t_r_sigma[self.last_r_visited][-1] += self.cv_interval
                ## Check whether the velocity is pointing out
                if self.xi.cv_r_grad is not None:
                    if self.xi.above_sigma(self.dyn.atoms):
                        grad_r_func = self.xi.cv_r_grad[self.last_r_visited]
                        g_R_raw = grad_r_func(self.dyn.atoms).flatten()
                        if inspect.isfunction(self.xi.cv_r):
                            condition = self.xi.r_crit
                            val_threshold = self.xi.in_r_boundary
                            current_val = self.xi.evaluate_cv_r(self.dyn.atoms)
                        elif isinstance(self.xi.cv_r, list):
                            condition = self.xi.r_crit[self.last_r_visited]
                            val_threshold = self.xi.in_r_boundary[self.last_r_visited]
                            current_val = self.xi.evaluate_cv_r(self.dyn.atoms)[self.last_r_visited]
                        else:
                            raise ValueError("""Problem of CollectiveVariables definition""")
                        # Select the outward normal direction
                        if condition == 'below':
                            g_R = g_R_raw
                        elif condition == 'above':
                            g_R = -g_R_raw
                        elif condition == 'between':
                            v_min, v_max = val_threshold
                            g_R = g_R_raw if abs(current_val - v_max) < abs(current_val - v_min) else -g_R_raw
                        else:
                            g_R = g_R_raw
                        v_dot_n = np.dot(self.dyn.atoms.get_velocities().flatten(), g_R)
                else:
                    v_dot_n = 1
                valid_exit = v_dot_n > 0
            if not hasattr(self.dyn.atoms, "info"):
                self.dyn.atoms.info = {}
            self.dyn.atoms.info["from_which_r"] = self.last_r_visited
            fname = f"{self.ini_cond_dir}/{self.n_ini_conds_already + n_cdt + 1}.extxyz"
            write(fname, self.dyn.atoms, format="extxyz")
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

        traj = self.dyn.closelater(Trajectory(self.trajfile, "a", self.dyn.atoms, properties=["energy", "stress", "forces"]))
        self.dyn.attach(traj.write, interval=self.cv_interval)
        barrier()

        if self.fixcm:
            self.dyn.fix_com = True
            self.dyn.atoms._constraints = []
            self.dyn.atoms.set_constraint(FixCom())

        self.n_ini_conds_already = len([ini for ini in os.listdir(self.ini_cond_dir) if ini.endswith("z")])
        if self.dyn.nsteps > 0:
            self.dyn.atoms.calc.results["forces"] = forces
            self.dyn.atoms.calc.results["stress"] = stress
            self.dyn.atoms.calc.results["energy"] = energy
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
            self.last_r_visited = np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
        out_of_r_zone = self.xi.is_out_of_r_zone(self.dyn.atoms)
        above_sigma = self.xi.above_sigma(self.dyn.atoms)
        if self.first_in_r:
            if self.going_to_sigma:
                self.t_r_sigma[self.last_r_visited][-1] += 1
            if self.going_back_to_r:
                self.t_sigma_r[self.last_r_visited][-1] += 1
            if above_sigma and self.going_to_sigma:
                if self.xi.cv_r_grad is not None:
                    grad_r_func = self.xi.cv_r_grad[self.last_r_visited]
                    g_R_raw = grad_r_func(self.dyn.atoms).flatten()
                    if inspect.isfunction(self.xi.cv_r):
                        condition = self.xi.r_crit
                        val_threshold = self.xi.in_r_boundary
                        current_val = self.xi.evaluate_cv_r(self.dyn.atoms)
                    elif isinstance(self.xi.cv_r, list):
                        condition = self.xi.r_crit[self.last_r_visited]
                        val_threshold = self.xi.in_r_boundary[self.last_r_visited]
                        current_val = self.xi.evaluate_cv_r(self.dyn.atoms)[self.last_r_visited]
                    else:
                        raise ValueError("""Problem of CollectiveVariables definition""")
                    # Select the outward normal direction
                    if condition == 'below':
                        g_R = g_R_raw
                    elif condition == 'above':
                        g_R = -g_R_raw
                    elif condition == 'between':
                        v_min, v_max = val_threshold
                        g_R = g_R_raw if abs(current_val - v_max) < abs(current_val - v_min) else -g_R_raw
                    else:
                        g_R = g_R_raw
                    v_dot_n = np.dot(self.dyn.atoms.get_velocities().flatten(), g_R)
                else:
                    v_dot_n = 1
                if v_dot_n > 0:
                    self.going_to_sigma = False
                    if not hasattr(self.dyn.atoms, 'info'):
                        self.dyn.atoms.info = {}
                    self.dyn.atoms.info['from_which_r'] = self.last_r_visited
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
                self.dyn.atoms.calc.results["forces"] = np.zeros_like(forces)
                self.dyn.atoms.calc.results["stress"] = np.zeros_like(stress)
                self.dyn.atoms.calc.results["energy"] = np.zeros_like(energy)
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
                self.dyn.atoms.calc.results["forces"] = np.zeros_like(forces)
                self.dyn.atoms.calc.results["stress"] = np.zeros_like(stress)
                self.dyn.atoms.calc.results["energy"] = np.zeros_like(energy)
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
                self.dyn.atoms.calc.results["forces"] = np.zeros_like(forces)
                self.dyn.atoms.calc.results["stress"] = np.zeros_like(stress)
                self.dyn.atoms.calc.results["energy"] = np.zeros_like(energy)
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
                self.dyn.atoms.calc.results["forces"] = np.zeros_like(forces)
                self.dyn.atoms.calc.results["stress"] = np.zeros_like(stress)
                self.dyn.atoms.calc.results["energy"] = np.zeros_like(energy)
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

    def set_run_dir(self, run_dir="./ini_conds_walker_", append_traj=False):
        """Where the md logs will be written, if the directory does not exist, it will create it."""
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir + str(self.w_i)) and world.rank == 0:
            os.mkdir(self.run_dir + str(self.w_i))
        n_traj_already = len([fi for fi in os.listdir(self.run_dir + str(self.w_i)) if fi.endswith(".traj")])
        if not append_traj:
            self.trajfile = self.run_dir + str(self.w_i) + "/md_traj_{}.traj".format(n_traj_already)
            traj = self.dyn.closelater(Trajectory(filename=self.trajfile, mode="a", atoms=self.dyn.atoms, properties=["energy", "stress", "forces"]))
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
            "t_r_sigma_out": self.t_r_sigma_out,
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
            n_traj_already = len([fi for fi in os.listdir(self.run_dir + str(branch_rep_number)) if fi.endswith(".traj")])
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
            f.write("Attempt branching replica " + str(self.w_i) + " from replica index: " + str(branch_rep_number) + " at time " + str(current_time) + " \n")
            if traj_idx is not None:
                f.write("Succesful branching\n \n")
                f.close()
            else:
                f.write("Cannot branch as replica " + str(branch_rep_number) + "  did not reach time " + str(current_time) + "\n")
                f.write("Going to sleep for 1 minute\n")
                f.close()
                time.sleep(60)
        trajfile = self.run_dir + str(branch_rep_number) + "/md_traj_{}.traj".format(traj_idx)
        list_atoms = read(trajfile, index=":", format="traj")
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
                Call initialconditionssampler.set_ini_cond_dir"""
            )
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
            self.dyn.atoms._constraints = []
            self.dyn.atoms.set_constraint(FixCom())
            self.dyn.fix_com = True
        n_ini_conds_already = len([fi for fi in os.listdir(self.ini_cond_dir) if fi.startswith("walker_" + str(self.w_i) + "_")])
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
            self.last_r_visited = np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
            self.going_back_to_r = False
            self.going_to_sigma = True
            self.t_r_sigma[self.last_r_visited].append(0)
            self.t_sigma_r[self.last_r_visited].append(0)
            valid_exit = False
            while not (self.xi.above_sigma(self.dyn.atoms) and valid_exit):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                self.t_r_sigma[self.last_r_visited][-1] += self.cv_interval
                ## Check whether the velocity is pointing out
                if self.xi.cv_r_grad is not None:
                    if self.xi.above_sigma(self.dyn.atoms):
                        grad_r_func = self.xi.cv_r_grad[self.last_r_visited]
                        g_R_raw = grad_r_func(self.dyn.atoms).flatten()
                        if inspect.isfunction(self.xi.cv_r):
                            condition = self.xi.r_crit
                            val_threshold = self.xi.in_r_boundary
                            current_val = self.xi.evaluate_cv_r(self.dyn.atoms)
                        elif isinstance(self.xi.cv_r, list):
                            condition = self.xi.r_crit[self.last_r_visited]
                            val_threshold = self.xi.in_r_boundary[self.last_r_visited]
                            current_val = self.xi.evaluate_cv_r(self.dyn.atoms)[self.last_r_visited]
                        else:
                            raise ValueError("""Problem of CollectiveVariables definition""")
                        # Select the outward normal direction
                        if condition == 'below':
                            g_R = g_R_raw
                        elif condition == 'above':
                            g_R = -g_R_raw
                        elif condition == 'between':
                            v_min, v_max = val_threshold
                            g_R = g_R_raw if abs(current_val - v_max) < abs(current_val - v_min) else -g_R_raw
                        else:
                            g_R = g_R_raw
                        v_dot_n = np.dot(self.dyn.atoms.get_velocities().flatten(), g_R)
                else:
                    v_dot_n = 1
                valid_exit = v_dot_n > 0
            if not hasattr(self.dyn.atoms, "info"):
                self.dyn.atoms.info = {}
            self.dyn.atoms.info["from_which_r"] = self.last_r_visited
            fname = self.ini_cond_dir + "/walker_" + str(self.w_i) + "_ini_cond_" + str(n_ini_conds_already + n_cdt + 1) + ".extxyz"
            write(fname, self.dyn.atoms, format="extxyz")
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
        traj = self.dyn.closelater(Trajectory(filename=self.trajfile, mode="a", atoms=self.dyn.atoms, properties=["energy", "stress", "forces"]))
        self.dyn.attach(traj.write, interval=self.cv_interval)
        barrier()
        if self.fixcm:
            self.dyn.fix_com = True
            self.dyn.atoms._constraints = []
            self.dyn.atoms.set_constraint(FixCom())
        self.n_ini_conds_already = len([fi for fi in os.listdir(self.ini_cond_dir) if fi.startswith("walker_" + str(self.w_i) + "_")])
        if self.dyn.nsteps > 0:
            self.dyn.atoms.calc.results["forces"] = forces
            self.dyn.atoms.calc.results["stress"] = stress
            self.dyn.atoms.calc.results["energy"] = energy
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
            self.last_r_visited = np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
        out_of_r_zone = self.xi.is_out_of_r_zone(self.dyn.atoms)
        above_sigma = self.xi.above_sigma(self.dyn.atoms)
        if self.first_in_r:
            if self.going_to_sigma:
                self.t_r_sigma[self.last_r_visited][-1] += 1
            if self.going_back_to_r:
                self.t_sigma_r[self.last_r_visited][-1] += 1
            if above_sigma and self.going_to_sigma:
                if self.xi.cv_r_grad is not None:
                    grad_r_func = self.xi.cv_r_grad[self.last_r_visited]
                    g_R_raw = grad_r_func(self.dyn.atoms).flatten()
                    if inspect.isfunction(self.xi.cv_r):
                        condition = self.xi.r_crit
                        val_threshold = self.xi.in_r_boundary
                        current_val = self.xi.evaluate_cv_r(self.dyn.atoms)
                    elif isinstance(self.xi.cv_r, list):
                        condition = self.xi.r_crit[self.last_r_visited]
                        val_threshold = self.xi.in_r_boundary[self.last_r_visited]
                        current_val = self.xi.evaluate_cv_r(self.dyn.atoms)[self.last_r_visited]
                    else:
                        raise ValueError("""Problem of CollectiveVariables definition""")
                    # Select the outward normal direction
                    if condition == 'below':
                        g_R = g_R_raw
                    elif condition == 'above':
                        g_R = -g_R_raw
                    elif condition == 'between':
                        v_min, v_max = val_threshold
                        g_R = g_R_raw if abs(current_val - v_max) < abs(current_val - v_min) else -g_R_raw
                    else:
                        g_R = g_R_raw
                    v_dot_n = np.dot(self.dyn.atoms.get_velocities().flatten(), g_R)
                else:
                    v_dot_n = 1
                if v_dot_n > 0:
                    self.going_to_sigma = False
                    if not hasattr(self.dyn.atoms, 'info'):
                        self.dyn.atoms.info = {}
                    self.dyn.atoms.info['from_which_r'] = self.last_r_visited
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
                self.dyn.atoms.calc.results["forces"] = np.zeros_like(forces)
                self.dyn.atoms.calc.results["stress"] = np.zeros_like(stress)
                self.dyn.atoms.calc.results["energy"] = np.zeros_like(energy)
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
                self.dyn.atoms.calc.results["forces"] = np.zeros_like(forces)
                self.dyn.atoms.calc.results["stress"] = np.zeros_like(stress)
                self.dyn.atoms.calc.results["energy"] = np.zeros_like(energy)
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
            self.t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
            self.t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
            self.t_r_sigma_out = [[] for i in range(len(self.xi.cv_r))]
            self.t_sigma_out = [[] for i in range(len(self.xi.cv_r))]
        else:
            self.t_r_sigma = [[]]
            self.t_sigma_r = [[]]
            self.t_r_sigma_out = [[]]
            self.t_sigma_out = [[]]

        n_cdt, n_stp = 0, 0
        n_ini = len([f for f in os.listdir(self.ini_cond_dir) if f.endswith(".extxyz")])
        traj = read(file, index=":")
        n_steps = len(traj)

        while not self.xi.in_r(traj[n_stp]) and n_stp < n_steps:
            n_stp += self.cv_interval

        while n_stp < n_steps:
            which_r = np.where(self.xi.in_which_r(traj[n_stp]) == np.max(self.xi.in_which_r(traj[n_stp])))[0][0]
            self.t_r_sigma[which_r].append(0)
            valid_exit = False
            while not self.xi.above_sigma(traj[n_stp]) and n_stp < n_steps and not valid_exit:
                n_stp += self.cv_interval
                if n_stp >= n_steps:
                    self.t_r_sigma[which_r].pop()
                    break
                self.t_r_sigma[which_r][-1] += self.cv_interval
                ## Check whether the velocity is pointing out
                if self.xi.cv_r_grad is not None:
                    if self.xi.above_sigma(traj[n_stp]):
                        grad_r_func = self.xi.cv_r_grad[which_r]
                        g_R_raw = grad_r_func(traj[n_stp]).flatten()
                        if inspect.isfunction(self.xi.cv_r):
                            condition = self.xi.r_crit
                            val_threshold = self.xi.in_r_boundary
                            current_val = self.xi.evaluate_cv_r(traj[n_stp])
                        elif isinstance(self.xi.cv_r, list):
                            condition = self.xi.r_crit[which_r]
                            val_threshold = self.xi.in_r_boundary[which_r]
                            current_val = self.xi.evaluate_cv_r(traj[n_stp])[which_r]
                        else:
                            raise ValueError("""Problem of CollectiveVariables definition""")
                        # Select the outward normal direction
                        if condition == 'below':
                            g_R = g_R_raw
                        elif condition == 'above':
                            g_R = -g_R_raw
                        elif condition == 'between':
                            v_min, v_max = val_threshold
                            g_R = g_R_raw if abs(current_val - v_max) < abs(current_val - v_min) else -g_R_raw
                        else:
                            g_R = g_R_raw
                        v_dot_n = np.dot(traj[n_stp].get_velocities().flatten(), g_R)
                else:
                    v_dot_n = 1
                valid_exit = v_dot_n > 0
            if n_stp >= n_steps: break
            fname = f"{self.ini_cond_dir}/{n_ini + n_cdt + 1}.extxyz"
            at = traj[n_stp].copy()
            if not hasattr(at, "info"):
                at.info = {}
            at.info["from_which_r"] = which_r
            write(fname, at, format="extxyz")
            self.t_sigma_r[which_r].append(0)

            while not self.xi.in_r(traj[n_stp]) and n_stp < n_steps:
                n_stp += self.cv_interval
                if n_stp >= n_steps:
                    self.t_sigma_r[which_r].pop()
                    break
                if self.xi.is_out_of_r_zone(traj[n_stp]):
                    t_sigma_out = self.t_sigma_r[which_r].pop(-1)
                    t_r_sigma_out = self.t_r_sigma[which_r].pop(-1)
                    self.t_sigma_out[which_r].append(t_sigma_out)
                    self.t_r_sigma_out[which_r].append(t_r_sigma_out)
                self.t_sigma_r[which_r][-1] += self.cv_interval
            n_cdt += 1
