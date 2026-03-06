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
from scipy.special import erfcx, erfc

from src.aseams.ams import NumpyEncoder


def sample_rayleigh(sigma, rng=None):
    """
    Échantillonne une loi de Rayleigh : f(p_n) = (p_n/sigma^2) * exp(-p_n^2 / (2*sigma^2))

    Dans le cadre des moments avec métrique de masse, sigma vaut sqrt(k_B T).
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.uniform(1e-10, 1.0 - 1e-10)
    return sigma * math.sqrt(-2.0 * math.log(1.0 - u))


def _get_p_r_constants(u_R, sigma_R):
    """
    Calcule p_0 et Z_star de manière robuste pour le moment normal décalé.
    u_n : décalage (alpha * sigma_n * rho)
    sigma_n : sqrt(k_B T)
    """
    # Calcul de p_0 (valeur de la densité non normalisée en 0)
    # Nécessaire pour l'algo de Newton
    p_0 = math.exp(-u_R ** 2 / (2 * sigma_R ** 2))

    # Calcul robuste de Z_star
    # On utilise l'identité analytique :
    # Z* = sigma^2 * exp(-u^2/2sigma^2) + u * sigma * sqrt(2pi) * Phi(u/sigma)
    # Mais formulée avec erfcx pour éviter les overflows/underflows :

    x = -u_R / (math.sqrt(2) * sigma_R)
    # erfcx(x) = exp(x^2) * erfc(x). C'est toujours > 0.

    # Le terme intégral exact s'écrit :
    # Z* = sigma^2 * exp(-u^2/2sigma^2) * [ 1 + u/sigma * sqrt(pi/2) * erfcx( -u / (sqrt(2)*sigma) ) ]

    term_erfcx = (u_R / sigma_R) * math.sqrt(math.pi / 2) * erfcx(x)

    # Facteur pré-exponentiel (attention p_0 contient déjà l'exp)
    # Z_star = p_0 * sigma^2 * (1 + term_erfcx_scaled...)
    # Plus simplement, on réutilise p_0 calculé au dessus :

    z_star = (sigma_R ** 2) * p_0 * (1.0 + term_erfcx)

    return p_0, z_star


def sample_biased_p_r(u_R, sigma_R, rng=None):
    """
    Échantillonne le moment normal p_n selon une Rayleigh décalée.
    """
    if rng is None:
        rng = np.random.default_rng()
    U = rng.uniform(1e-10, 1.0 - 1e-10)

    # 1. Calcul des constantes de normalisation
    p_0, z_star = _get_p_r_constants(u_R, sigma_R)

    # 2. Définition de G(v) = CDF(v) - U
    # On utilise l'identité : CDF(v) = 1 - (f(v) / f(0))
    # où f(0) est proportionnel à z_star.
    def G_func(v):
        if v <= 0: return -U

        # y_v = (v - u_R) / (sqrt(2) * sigma_R)
        # y_0 = -u_R / (sqrt(2) * sigma_R)
        y_v = (v - u_R) / (1.4142135623730951 * sigma_R)
        y_0 = -u_R / (1.4142135623730951 * sigma_R)

        # Calcul du ratio f(v)/f(0) de façon stable
        # ratio = exp(y_0^2 - y_v^2) * [ (sigma + u*sqrt(pi/2)*erfcx(y_v)) / (sigma + u*sqrt(pi/2)*erfcx(y_0)) ]
        arg_exp = (v * (2 * u_R - v)) / (2 * sigma_R ** 2)
        num = sigma_R + u_R * 1.2533141373155001 * erfcx(y_v)
        den = sigma_R + u_R * 1.2533141373155001 * erfcx(y_0)

        ratio = math.exp(arg_exp) * (num / den)
        return (1.0 - ratio) - U

    def G_prime(v):
        # La dérivée est simplement la PDF : p(v) = (v/z_star) * exp(-(v-u_R)^2 / 2sigma^2)
        arg_exp = -(v - u_R) ** 2 / (2 * sigma_R ** 2)
        return (v / z_star) * math.exp(arg_exp)

    # 3. Résolution robuste
    # Estimation du mode de la distribution pour un bon x0
    # Le mode de v*exp(-(v-u)^2/2s^2) est (u + sqrt(u^2 + 4s^2))/2
    x0 = (u_R + math.sqrt(u_R ** 2 + 4 * sigma_R ** 2)) / 2.0

    try:
        return newton(func=G_func, x0=x0, fprime=G_prime, tol=1e-8, maxiter=50)
    except RuntimeError:
        # Fallback sur Brentq si Newton échoue (plus lent mais garanti si les signes diffèrent)
        # On définit une borne supérieure sécurisée (10 sigmas après le centre)
        upper_bound = max(10.0 * sigma_R, u_R + 10.0 * sigma_R)
        from scipy.optimize import brentq
        return brentq(G_func, 0, upper_bound)


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
            g_R = raw_grad_r
        elif condition == "above":
            g_R = -raw_grad_r
        elif condition == "between":
            v_min, v_max = val_threshold
            g_R = raw_grad_r if abs(current_val - v_max) < abs(current_val - v_min) else -raw_grad_r
        else:
            g_R = raw_grad_r

        # --- 2. Paramètres Géométriques et Physiques ---
        masses = atoms.get_masses()
        m_3n = np.repeat(masses, 3)

        # Calcul de la norme du gradient dans la métrique de masse inverse
        # norm_g_R = sqrt( grad^T M^-1 grad )
        inv_m_eff = np.sum((g_R ** 2) / m_3n)
        norm_g_R = math.sqrt(inv_m_eff)

        # Vecteurs unitaires (e_R_config = e_R, M_e_R = M * e_R)
        e_R_config = (g_R / m_3n) / norm_g_R  # Dimension [1/sqrt(masse)]
        eta_R = g_R / norm_g_R  # Dimension [sqrt(masse)]

        # Sigma thermique pour le moment : sigma_p = sqrt(k_B * T)
        # Note : p_n^2 / (2 * sigma_p^2) est adimensionnel
        sigma_p_phys = math.sqrt(units.kB * temp_phys)
        sigma_p_bias = math.sqrt(units.kB * temp_bias)
        # --- 3. Sampling du Moment p ---
        # Moment normal p_n (Rayleigh)
        p_n_sampled = sample_rayleigh(sigma_p_bias, rng=rng)

        # Composantes tangentielles (Maxwell-Boltzmann p ~ N(0, M/beta))
        p_th = rng.normal(0, sigma_p_phys * np.sqrt(m_3n))

        # Projection pour obtenir p_perp (orthogonal à e_R au sens de M^-1)
        # p_perp = p_th - (p_th^T e_R) * M_e_R
        p_perp = p_th - np.dot(p_th, e_R_config) * eta_R

        # Synthèse du moment total : p = p_n * M_e_R + p_perp
        p_final = p_n_sampled * eta_R + p_perp

        # --- 4. Calcul du poids (Section 3.2 LaTeX) ---
        temp_ratio = temp_bias / temp_phys
        # p_n_sampled est déjà un moment, l'énergie est p_n^2 / 2
        energy_factor = (p_n_sampled ** 2) / (2.0 * units.kB)
        diff_beta = (1.0 / temp_phys) - (1.0 / temp_bias)

        weight = temp_ratio * math.exp(-energy_factor * diff_beta)

        # --- 5. Finalize Atoms object ---
        atoms.set_momenta(p_final.reshape((-1, 3)), apply_constraint=False)
        if not hasattr(atoms, 'info'):
            atoms.info = {}
        atoms.info["weight_ini_cond"] = weight

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
        sigma_p_R = math.sqrt(units.kB * temp)

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

        # Normalisation métrique (Mass-weighted)
        norm_g_R = math.sqrt(np.sum((g_R ** 2) / m_3n))
        norm_g_xi = math.sqrt(np.sum((g_xi ** 2) / m_3n))

        # Vecteurs unitaires et duals
        e_R_config = (g_R / m_3n) / norm_g_R
        e_xi_config = (g_xi / m_3n) / norm_g_xi
        eta_R = g_R / norm_g_R
        eta_xi = g_xi / norm_g_xi

        # Corrélation géométrique (rho)
        rho_R_xi = np.dot(e_xi_config, eta_R)

        # --- 3. SÉCURITÉ 1 : Clipping du Décalage ---
        # On calcule le décalage théorique u_p_R
        u_p_R_raw = alpha * sigma_p_R * rho_R_xi
        # Protection : rho ne doit jamais dépasser 1 en valeur absolue.
        # Des erreurs numériques sur des systèmes complexes peuvent l'induire.
        u_p_R = np.clip(u_p_R_raw, -alpha * sigma_p_R, alpha * sigma_p_R)

        # --- 4. Sampling ---
        p_n_sampled = sample_biased_p_r(u_p_R, sigma_p_R, rng=rng)

        # Bruit thermique translaté par le boost alpha selon la coordonnée de réaction
        delta_p = (alpha * sigma_p_R) * eta_xi
        p_th = rng.normal(0, sigma_p_R * np.sqrt(m_3n))
        p_full = delta_p + p_th

        # Retrait de la composante normale pour injecter p_n_sampled
        p_perp = p_full - np.dot(p_full, e_R_config) * eta_R
        p_final = p_n_sampled * eta_R + p_perp

        # --- 5. SÉCURITÉ 2 : Calcul du Poids via Logarithme ---
        p_0, z_star = _get_p_r_constants(u_p_R, sigma_p_R)
        R_Z = z_star / (sigma_p_R ** 2)

        p_xi = np.dot(p_final, e_xi_config)
        arg_exp = -alpha * (p_xi / sigma_p_R) + (alpha ** 2 / 2.0)

        # Calcul sécurisé : log(W) = log(R_Z) + arg_exp
        log_weight = math.log(R_Z) + arg_exp

        # Seuil d'overflow pour float64 (~exp(709))
        if log_weight > 700:
            weight = 1e308
            print(f"Warning: Poids plafonné (log_w={log_weight:.2f})")
        else:
            weight = math.exp(log_weight)

        # --- 6. Mise à jour de l'objet Atoms ---
        atoms.set_momenta(p_final.reshape((-1, 3)), apply_constraint=False)
        if not hasattr(atoms, "info"):
            atoms.info = {}
        atoms.info["weight_ini_cond"] = weight

        return atoms

    def bias_initial_conditions(
            self,
            input_dir,
            output_dir,
            temp,
            alpha=0.0,
            temp_bias=None,
            method="flux",
            n_draws=1,  # Nouveau paramètre
            rng=None,
            overwrite=False,
    ):
        """
        Applique le biaisage de vitesse à toutes les conditions initiales d'un répertoire,
        en effectuant éventuellement plusieurs tirages par configuration.

        Parameters:
        -----------
        ... (identique) ...
        n_draws : int
            Nombre de vecteurs de vitesse à générer pour chaque configuration de position.
        """
        if method not in ["flux", "rayleigh"]:
            raise ValueError("method must be either 'flux' or 'rayleigh'")

        if method == "rayleigh" and temp_bias is None:
            raise ValueError("temp_bias must be provided when using method='rayleigh'")

        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        if world.rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        barrier()

        input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".extxyz")])
        if not input_files:
            raise FileNotFoundError(f"No .extxyz files found in {input_dir}")

        all_weights = []
        all_output_paths = []

        # Utilisation d'un RNG par défaut si non fourni
        if rng is None:
            rng = np.random.default_rng()

        for fname in input_files:
            in_path = os.path.join(input_dir, fname)
            atoms_orig = read(in_path)
            base_name = os.path.splitext(fname)[0]

            # --- Boucle de tirages multiples ---
            for i in range(n_draws):
                # On crée un suffixe pour différencier les tirages
                draw_suffix = f"_draw_{i + 1}" if n_draws > 1 else ""
                out_fname = f"{base_name}{draw_suffix}.extxyz"
                out_path = os.path.join(output_dir, out_fname)

                if os.path.exists(out_path) and not overwrite:
                    continue

                # Copie profonde pour éviter de polluer l'objet original
                atoms = atoms_orig.copy()

                if method == "flux":
                    # Utilise la fonction avec les sécurités (clipping, log_weight)
                    biased_atoms = self.bias_one_initial_condition_flux(
                        atoms, alpha=alpha, temp=temp, rng=rng
                    )
                else:
                    biased_atoms = self.bias_one_initial_condition_rayleigh(
                        atoms, temp_phys=temp, temp_bias=temp_bias, rng=rng
                    )

                # Sauvegarde
                write(out_path, biased_atoms, format="extxyz")

                # Stockage des métadonnées
                weight = biased_atoms.info.get("weight_ini_cond", 1.0)
                all_weights.append(weight)
                all_output_paths.append(out_path)

        summary = {
            "n_input_configs": len(input_files),
            "n_draws_per_config": n_draws,
            "total_generated": len(all_output_paths),
            "method_used": method,
            "weights": all_weights,
            "output_files": all_output_paths,
        }

        if world.rank == 0:
            print(f"Done. Generated {len(all_output_paths)} biased configurations.")

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
            self.dyn.atoms.set_positions(atoms.get_positions(apply_constraint=True))
        else:
            self.dyn.atoms.set_positions(atoms.get_positions(apply_constraint=False))
        self.dyn.atoms.set_momenta(atoms.get_momenta(), apply_constraint=False)
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
        self._write_checkpoint(f"{self.run_dir}/ini_checkpoint.txt")
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
                        v_dot_n = -1
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
        self._write_checkpoint()
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
                        v_dot_n = -1
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
