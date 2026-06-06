import warnings, os
from typing import Optional
from ase import Atoms, units
from ase.units import kB
import numpy as np
from ase.md.md import MolecularDynamics
from ase.constraints import FixCom
from pathlib import Path
from ase.io import read


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
        self.atoms.constraints = []
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


class BlueMoonOBABOWithLambdas(MolecularDynamics):
    """
    Intégrateur Langevin OBABO contraint (schéma 3.160, Lelièvre et al.).

    Éq. Langevin (friction scalaire tilde{gamma} = xi, en s^-1) :
        dq = M^{-1} p dt
        dp = -∇V dt - xi p dt + sqrt(2 xi M k_B T) dW + ∇xi dΛ

    Splitting OBABO :
        O(dt/2) - B(dt/2) - A(dt) - B(dt/2) - O(dt/2)

    Contrainte géométrique scalaire : xi(q) = const.

    Stockage à chaque pas dans atoms.info :
        * 'lambda_half'      : λ^{n+1/2}   (step B en q^n)
        * 'lambda_3quarter'  : λ^{n+3/4}   (step B en q^{n+1})
        * 'G_M'              : G_M(q^{n+1}) = ∇xi(q^{n+1})^T M^{-1} ∇xi(q^{n+1})
        * 'G'                : G(q^{n+1}) = ∇xi(q^{n+1})^T ∇xi(q^{n+1})

    Les λ stockés ici sont exactement ceux de l’équation (3.160), c.-à-d. les
    incréments du processus de multiplicateur sur chaque demi-pas B
        p_out = p_star + ∇xi λ^B
    sans facteur dt/2.
    """

    def __init__(self, atoms, timestep, friction, temperature_K,
                 constraint_func, rng=None, iter_max=100, tol=1e-8,
                 logfile=None, trajectory=None, fixcm=False):

        super().__init__(atoms, timestep, logfile, trajectory)

        # tilde{gamma} dans le texte (notée ici xi), en s^-1
        self.xi = friction
        self.temp = temperature_K * kB  # k_B T en unités ASE
        self.dt = timestep
        self.constraint_func = constraint_func

        self.rng = np.random.default_rng() if rng is None else rng

        self.iter_max = iter_max
        self.tol = tol

        # clear all constrains
        self.atoms.constraints = []
        self.fix_com = fixcm
        if fixcm:
            self.atoms.set_constraint(FixCom())

        # Valeur cible de la contrainte (scalaire)
        val, _ = self.constraint_func(self.atoms)
        self.target_val = val

        # Champs info
        self.atoms.info['lambda_half'] = 0.0
        self.atoms.info['lambda_3quarter'] = 0.0
        self.atoms.info['G_M'] = 0.0
        self.atoms.info['G'] = 0.0

    # ---------- G_M(q) = ∇xi^T M^{-1} ∇xi ----------

    def _gram_scalar_M(self, grad):
        """
        G_M(q) = ∇xi(q)^T M^{-1} ∇xi(q) pour une coordonnée de réaction scalaire.
        G(q) = ∇xi(q)^T ∇xi(q) pour une coordonnée de réaction scalaire
        grad : array (N, 3), gradient de xi(q) w.r.t. positions atomiques.
        """
        inv_masses = 1.0 / self.atoms.get_masses()
        return np.sum((grad ** 2) * inv_masses[:, np.newaxis]), np.sum((grad ** 2))

    # ---------- Projections sur l'espace tangent ----------

    def _project_momenta_tangent(self, p, grad, G_M):
        """
        Projection de p sur le sous-espace tangent (∇xi^T M^{-1} p_tan = 0) :

            p_tan = p - alpha * grad,
            alpha = (∇xi^T M^{-1} p) / G_M(q).

        (G_M = ∇xi^T M^{-1} ∇xi, pas de racine carrée.)
        """
        inv_masses = 1.0 / self.atoms.get_masses()
        p_dot_grad_M = np.sum(p * inv_masses[:, np.newaxis] * grad)
        alpha = p_dot_grad_M / G_M
        return p - alpha * grad

    def _project_momenta_and_lambda_B(self, p, grad, G_M):
        """
        Projection pour une étape B et récupération de λ^B cohérent avec (3.160).

        Étape B du schéma (3.160), coordonnée scalaire :

            p_star = p^{n+1/4} + (dt/2) F(q^n),    avec F = -∇V
            p^{n+1/2} = p_star + ∇xi(q^n) λ^{n+1/2}

        et la contrainte :
            ∇xi(q^n)^T M^{-1} p^{n+1/2} = 0.

        Pour une projection :
            p_proj = p_star - alpha * grad
                   = p_star + grad * λ^B

        on obtient :
            alpha = (∇xi^T M^{-1} p_star) / G_M(q^n)
            λ^B   = -alpha

        On renvoie donc p_proj et λ^B = λ^{n+1/2} (ou λ^{n+3/4}), tels que
            p_proj = p_star + ∇xi λ^B
        exactement comme dans l’équation (3.160) (sans facteur dt/2).
        """
        inv_masses = 1.0 / self.atoms.get_masses()
        p_dot_grad_M = np.sum(p * inv_masses[:, np.newaxis] * grad)
        alpha = p_dot_grad_M / G_M

        p_proj = p - alpha * grad

        # *** CORRECTION CLEF ***
        # λ^B doit satisfaire p_proj = p_star + ∇xi λ^B, donc λ^B = -alpha
        lambda_B = -alpha

        return p_proj, lambda_B

    # ---------- Étape O : Langevin (OU) + projection tangentielle ----------

    def _step_O(self, p, grad, G_M):
        """
        Étape O de durée dt/2 pour :
            dp = -xi p dt + sqrt(2 xi M kT) dW.

        Discrétisation OU (point-milieu) sur h = dt/2 :
            p_new = inv_coeff * (drift_coeff * p + bruit)

        avec :
            drift_coeff = 1 - xi * dt / 4
            inv_coeff   = 1 / (1 + xi * dt / 4)
            bruit ~ N(0, 2 xi M kT h)

        Puis projection tangentielle pour imposer :
            ∇xi^T M^{-1} p_new = 0

        Cela est exactement équivalent à l’équation de type
            p^{n+1/4} = p^n - (dt/4) xi (p^n + p^{n+1/4})
                        + sqrt(dt/2) σ G_n + ∇xi λ^{n+1/4}
        du schéma (3.160), avec λ^{n+1/4} absorbé dans la projection.
        """
        masses = self.atoms.get_masses()[:, np.newaxis]
        h = self.dt / 2.0

        drift_coeff = 1.0 - self.xi * self.dt / 4.0
        inv_coeff = 1.0 / (1.0 + self.xi * self.dt / 4.0)

        noise_std = np.sqrt(2.0 * self.xi * masses * self.temp * h)
        noise = self.rng.standard_normal(p.shape) * noise_std

        p_ou = inv_coeff * (p * drift_coeff + noise)

        # Projection tangentielle => impose la contrainte de momenta
        return self._project_momenta_tangent(p_ou, grad, G_M)

    # ---------- Pas de temps complet ----------

    def step(self):
        """
        Pas de temps complet OBABO contraint :

            (q^n,   p^n)
         -> O(dt/2)             : (q^n,   p^{n+1/4})
         -> B(dt/2) + λ^{n+1/2} : (q^n,   p^{n+1/2})
         -> A(dt) + SHAKE       : (q^{n+1}, p^{n+1/2})
         -> B(dt/2) + λ^{n+3/4} : (q^{n+1}, p^{n+3/4})
         -> O(dt/2)             : (q^{n+1}, p^{n+1})
        """
        masses = self.atoms.get_masses()[:, np.newaxis]
        inv_masses = 1.0 / masses

        # p^n
        p = self.atoms.get_momenta()

        # ---- 1. O(dt/2) en (q^n) : p^n -> p^{n+1/4} ----
        _, grad_n = self.constraint_func(self.atoms)
        G_M_n, G_n = self._gram_scalar_M(grad_n)

        p = self._step_O(p, grad_n, G_M_n)

        # ---- 2. B(dt/2) en (q^n) : p^{n+1/4} -> p^{n+1/2}, λ^{n+1/2} ----
        forces_n = self.atoms.get_forces()  # ASE: forces = -∇V
        p += 0.5 * self.dt * forces_n       # p_star = p^{n+1/4} + (dt/2) F(q^n)

        p, lambda_half = self._project_momenta_and_lambda_B(p, grad_n, G_M_n)

        # ---- 3. A(dt) : mise à jour de q + SHAKE ----
        pos_old = self.atoms.get_positions()

        # Proposition non contrainte (sans SHAKE)
        pos_current = pos_old + self.dt * (p * inv_masses)

        # Newton pour imposer xi(q^{n+1}) = target_val (direction ~ M^{-1} grad_n)
        for _ in range(self.iter_max):
            self.atoms.set_positions(pos_current)
            val, grad_current = self.constraint_func(self.atoms)

            diff = val - self.target_val
            if abs(diff) < self.tol:
                break

            # d/dλ xi(q + λ M^{-1} grad_n) ≈ grad_current^T M^{-1} grad_n
            derivative = np.sum(grad_current * inv_masses * grad_n)
            d_lambda = -diff / derivative
            pos_current += d_lambda * (inv_masses * grad_n)
        else:
            raise RuntimeError(
                f"Newton n'a pas convergé après {self.iter_max} itérations "
                f"(diff={diff:.2e})"
            )

        # q^{n+1}
        if self.fix_com:
            self.atoms.set_positions(pos_current, apply_constraint=True)
        else:
            self.atoms.set_positions(pos_current, apply_constraint=False)

        # ---- 4. B(dt/2) en (q^{n+1}) : p^{n+1/2} -> p^{n+3/4}, λ^{n+3/4} ----
        # RATTLE : reconstitue p^{n+1/2} cohérent avec q^{n+1}
        #p = (self.atoms.get_positions() - pos_old) * masses / self.dt

        forces_next = self.atoms.get_forces()  # forces en q^{n+1}
        p += 0.5 * self.dt * forces_next       # p_star en q^{n+1}

        _, grad_next = self.constraint_func(self.atoms)
        G_M_next, G_next = self._gram_scalar_M(grad_next)

        p, lambda_3quarter = self._project_momenta_and_lambda_B(
            p, grad_next, G_M_next
        )

        # ---- 5. O(dt/2) en (q^{n+1}) : p^{n+3/4} -> p^{n+1} ----
        p = self._step_O(p, grad_next, G_M_next)

        # p^{n+1}
        self.atoms.set_momenta(p)

        # ---- Stockage (fin de pas) ----
        # G_M(q^{n+1}) = ∇xi(q^{n+1})^T M^{-1} ∇xi(q^{n+1})
        self.atoms.info['G_M'] = float(G_M_next)
        self.atoms.info['G'] = float(G_next)

        # Multiplicateurs de Lagrange des deux étapes B (notations du livre)
        self.atoms.info['lambda_half'] = float(lambda_half)         # λ^{n+1/2}
        self.atoms.info['lambda_3quarter'] = float(lambda_3quarter) # λ^{n+3/4}


class ReplicaWithMC(BlueMoonOBABOWithLambdas):
    """
    Réplique avec moves Monte Carlo basés sur un réservoir de trajectoires.

    Paramètres
    ----------
    atoms : ase.Atoms
        Système associé à cette réplique.
    replica_index : int
        Indice de cette réplique (0, 1, ..., n_replicas-1 par exemple).
    replicas_root : str ou Path
        Chemin vers le répertoire qui contient tous les dossiers de répliques.
        Le chemin du dossier de la réplique i est pris comme:
            replicas_root / str(i)
        (adapter si ta convention est différente).
    n_replicas : int
        Nombre total de répliques.
    rng : np.random.Generator, optionnel
        Générateur aléatoire (par défaut: np.random.default_rng()).
    **kwargs :
        Arguments passés au constructeur de BlueMoonOBABOWithLambdas.
    """

    def __init__(self, atoms, replica_index, replicas_root, n_replicas,
                 rng=None, **kwargs):
        super().__init__(atoms, **kwargs)

        self.replica_index = int(replica_index)
        self.replicas_root = Path(replicas_root)
        self.n_replicas = int(n_replicas)
        self.rng = rng or np.random.default_rng()

        # Dossier de cette réplique : root / "<indice>"
        # Adapter si tu utilises "rep_0", "rep_1", etc.
        self.replica_dir = self.replicas_root / str(self.replica_index)

    # ------------------------------------------------------------------
    # Méthodes utilitaires
    # ------------------------------------------------------------------
    def _hamiltonian(self, atoms):
        """Retourne H = K + U pour un objet Atoms donné."""
        # ASE : unités cohérentes (eV) si le calculator utilise eV.
        e_pot = atoms.get_potential_energy()
        e_kin = atoms.get_kinetic_energy()
        return e_pot + e_kin

    def _draw_other_replica_index(self):
        """Tire un indice de réplique différent de self.replica_index."""
        all_indices = list(range(self.n_replicas))
        all_indices.remove(self.replica_index)
        return int(self.rng.choice(all_indices))

    # ------------------------------------------------------------------
    # Move Monte Carlo
    # ------------------------------------------------------------------
    def move_monte_carlo(self, n_hist):
        """
        Propose un move Monte Carlo à partir du réservoir de trajectoires.

        Étapes :
        1. Tire au hasard une autre réplique i_r.
        2. Tire au hasard un temps i_t dans [0, n_hist-1] (index Python).
        3. Lit constrained_md.traj de la réplique i_r, à l'image i_t.
        4. Calcule ΔH = H_new - H_old et teste Metropolis:
               p_acc = min(1, exp(-beta * ΔH)).
        5. Si accepté, remplace l'état courant (positions, vitesses, lambda).

        Paramètres
        ----------
        n_hist : int
            Nombre de frames disponibles dans constrained_md.traj
            (on tire un index i_t dans [0, n_hist-1]).
            Si dans ton fichier tu veux considérer les pas de 1 à n_hist,
            alors on prend i_t_python = i_t_physique - 1.

        Retourne
        -------
        accepted : bool
            True si le move est accepté, False sinon.
        info : dict
            Quelques infos sur le move (i_r, i_t, ΔH, p_acc, etc.).
        """
        # 1. Hamiltonien initial
        H_old = self._hamiltonian(self.atoms)

        # 2. Tirage de la réplique source
        i_r = self._draw_other_replica_index()

        # 3. Tirage du temps i_t
        #    Ici index Python: 0, 1, ..., n_hist-1
        i_t = int(self.rng.integers(0, n_hist))

        # Chemin vers le fichier de trajectoire de la réplique i_r
        # Adapter le nom si nécessaire (contrained_md.traj / constrained_md.traj)
        traj_path = self.replicas_root + "/" + str(i_r) +"/" + "constrained_md.traj"

        # 4. Lecture de la configuration proposée
        atoms_proposed = read(traj_path, index=-i_t+1)


        # 5. Hamiltonien de la configuration proposée
        H_new = self._hamiltonian(atoms_proposed)
        dH = H_new - H_old

        # 7. Probabilité d'acceptation Metropolis
        #    p_acc = min(1, exp(-beta * dH))
        log_p_acc = float(-dH / self.temp)

        # 8. Tirage uniform pour accepter / refuser
        u = float(self.rng.random())
        accepted = (np.log(u) <= log_p_acc)

        # 9. Si accepté, on met à jour self.atoms à partir de atoms_proposed
        if accepted:
            self.atoms = atoms_proposed.copy()
        info = {
            "accepted": accepted,
            "i_r": i_r,
            "i_t": i_t,
            "H_old": H_old,
            "H_new": H_new,
            "dH": dH,
            "log(p_acc)": log_p_acc,
            "log(u)": np.log(u),
        }
        return accepted, info


class EABFLangevinOBABOMultiDim(MolecularDynamics):
    """
    Intégrateur eABF multidimensionnel avec support de redémarrage (checkpoint)
    et stockage des CVs dans atoms.info.
    """

    def __init__(
        self,
        atoms,
        timestep,
        temperature_K,
        friction_at,
        friction_lambda,
        mass_lambda,
        k_spring,
        cv_func,
        cv_bounds,
        n_bins,
        n_full=1000,
        save_freq=1000,
        save_file="eabf_data.npz",
        rng=None,
        fixcm=False,
        k_wall=50.0,
        max_force=None,
        max_force_lambda=None,
        **kwargs,
    ):
        super().__init__(atoms, timestep, **kwargs)

        self.temp = temperature_K * kB
        self.fr_at = friction_at
        self.fr_lam = friction_lambda
        self.mass_lam = mass_lambda
        self.k_spring = k_spring
        self.k_wall = k_wall
        self.max_force = max_force
        self.max_force_lambda = max_force_lambda
        self.cv_func = cv_func
        self.rng = np.random.default_rng() if rng is None else rng

        # Configuration de la grille
        self.cv_bounds = cv_bounds
        self.n_bins = n_bins
        self.d = len(cv_bounds)
        self.bin_edges = [np.linspace(b[0], b[1], n + 1) for b, n in zip(cv_bounds, n_bins)]
        self.n_full = n_full
        self.save_freq = save_freq
        self.save_file = save_file
        self.bin_edges_z = self.bin_edges.copy()  # Grille pour la CV physique z
        self.z_centers = [(edges[:-1] + edges[1:]) / 2.0 for edges in self.bin_edges_z]
        self.z_counts = np.zeros(n_bins, dtype=np.float64)
        self.sum_lambda_z = np.zeros(list(n_bins) + [self.d], dtype=np.float64)

        # --- TENTATIVE DE CHARGEMENT (REDÉMARRAGE) ---
        if os.path.exists(self.save_file):
            print(f"[eABF] Fichier de sauvegarde détecté : {self.save_file}. Chargement des données...")
            self._load_tensors()
        else:
            print("[eABF] Aucun fichier de sauvegarde trouvé. Initialisation à zéro.")
            self.N_visits = np.zeros(n_bins, dtype=np.int64)
            self.F_bar = np.zeros(list(n_bins) + [self.d], dtype=np.float64)
            self.step_counter = 0

        # Variables auxiliaires
        xi_init, _ = self.cv_func(self.atoms)
        self.lambda_val = np.array(xi_init, dtype=np.float64)
        self.pi_val = self.rng.standard_normal(self.d) * np.sqrt(self.mass_lam * self.temp)

        self._update_thermostat_coeffs()

	    # clear all constrains
        self.atoms.constraints = []
        self.fix_com = fixcm
        if fixcm:
            self.atoms.set_constraint(FixCom())
	    # Do not write the first configuration
        self.nsteps = 1

    def _get_wall_force(self, lambda_val):
        """Calcule la force de rappel du mur harmonique multidimensionnel sur lambda."""
        f_wall = np.zeros(self.d)
        for i in range(self.d):
            b_min, b_max = self.cv_bounds[i]
            if lambda_val[i] < b_min:
                # Force positive pour ramener lambda vers l'intérieur
                f_wall[i] = -self.k_wall * (lambda_val[i] - b_min)
            elif lambda_val[i] > b_max:
                # Force négative pour ramener lambda vers l'intérieur
                f_wall[i] = -self.k_wall * (lambda_val[i] - b_max)
        return f_wall

    def _clip_forces(self, forces):
        """Limite la norme du vecteur force de chaque atome à max_force en préservant sa direction."""
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        # On évite la division par zéro avec np.maximum
        rescale_factor = self.max_force / np.maximum(norms, 1e-12)
        # On applique le facteur multiplicatif uniquement là où la norme dépasse le seuil
        return np.where(norms > self.max_force, forces * rescale_factor, forces)

    def _load_tensors(self):
        """Charge l'état d'apprentissage depuis le fichier binaire."""
        data = np.load(self.save_file)
        self.N_visits = data['N_visits']
        self.F_bar = data['F_bar']
        self.step_counter = int(data.get('step', 0))
        if 'z_counts' in data:
            self.z_counts = data['z_counts'].copy()
        else:
            print("[eABF][Warning] 'z_counts' introuvable dans le restart. Initialisé à zéro.")

        if 'sum_lambda_z' in data:
            self.sum_lambda_z = data['sum_lambda_z'].copy()
        else:
            print("[eABF][Warning] 'sum_lambda_z' introuvable dans le restart. Initialisé à zéro.")
        if self.N_visits.shape != tuple(self.n_bins):
            raise ValueError("La grille dans le fichier de sauvegarde ne correspond pas à la configuration actuelle.")

    def _update_thermostat_coeffs(self):
        dt = self.dt
        beta = 1.0 / self.temp
        masses = self.atoms.get_masses()[:, np.newaxis]
        self.c1_at = 1.0 + (self.fr_at * dt) / 4.0
        self.c2_at = 1.0 - (self.fr_at * dt) / 4.0
        self.c3_at = np.sqrt((masses * self.fr_at * dt) / beta)
        self.c1_lam = 1.0 + (self.fr_lam * dt) / 4.0
        self.c2_lam = 1.0 - (self.fr_lam * dt) / 4.0
        self.c3_lam = np.sqrt((self.mass_lam * self.fr_lam * dt) / beta)

    def _get_bin_indices(self, vals):
        indices = []
        for val, edges in zip(vals, self.bin_edges):
            if val < edges[0] or val >= edges[-1]:
                return None
            indices.append(np.digitize(val, edges) - 1)
        return tuple(indices)

    def _get_f_bias(self, bin_idx):
        # Si lambda est hors bornes, bin_idx est None
        if bin_idx is None:
            # On projette proprement lambda sur le bin valide le plus proche
            clamped_indices = []
            for i in range(self.d):
                val = self.lambda_val[i]
                edges = self.bin_edges[i]
                # Trouver l'index le plus proche (0 ou n_bins-1)
                idx = np.digitize(val, edges) - 1
                idx = max(0, min(idx, self.n_bins[i] - 1))
                clamped_indices.append(idx)
            bin_idx = tuple(clamped_indices)

        n_visits = self.N_visits[bin_idx]
        alpha = min(1.0, n_visits / self.n_full) if n_visits > 0 else 0.0
        return -alpha * self.F_bar[bin_idx]

    def step(self, forces=None):
        p = self.atoms.get_momenta()
        masses = self.atoms.get_masses()[:, np.newaxis]

        if forces is None:
            forces = self.atoms.get_forces(md=True)

        # 1. O (Thermostat)
        p, self.pi_val = self._step_O(p, self.pi_val)

        # 2. B (Kick)
        xi_val, grad_xi = self.cv_func(self.atoms)
        bin_idx = self._get_bin_indices(self.lambda_val)

        fc = self.k_spring * (xi_val - self.lambda_val)
        f_bias = self._get_f_bias(bin_idx)
        f_wall_lam = self._get_wall_force(self.lambda_val)

        # Force TOTALE subie par les atomes (Physique + Rappel Harmonique)
        f_at_total = forces - np.tensordot(fc, grad_xi, axes=1)

        # --- AJOUT : Clipping de la force totale sur les atomes ---
        if self.max_force is not None:
            f_at_total = self._clip_forces(f_at_total)

        p += (self.dt / 2.0) * f_at_total

        # Force TOTALE subie par la variable étendue lambda
        f_lam_total = fc + f_bias + f_wall_lam

        # --- AJOUT : Clipping de la force totale sur lambda ---
        max_f_lam = getattr(self, 'max_force_lambda', None)
        if max_f_lam is not None:
            norm_lam = np.linalg.norm(f_lam_total)
            if norm_lam > max_f_lam:
                f_lam_total = f_lam_total * (max_f_lam / max(norm_lam, 1e-12))

        self.pi_val += (self.dt / 2.0) * f_lam_total

        # 3. A (Drift)
        q = self.atoms.get_positions()
        q += self.dt * (p / masses)
        self.lambda_val += self.dt * (self.pi_val / self.mass_lam)
        if self.fix_com:
            self.atoms.set_positions(q, apply_constraint=True)
        else:
            self.atoms.set_positions(q, apply_constraint=False)

        # 4. Apprentissage
        xi_new, grad_xi_new = self.cv_func(self.atoms)
        fc_new = self.k_spring * (xi_new - self.lambda_val)
        new_bin_idx = self._get_bin_indices(self.lambda_val)
        idx_z = self._get_bin_indices(xi_new)
        if idx_z is not None:
            self.z_counts[idx_z] += 1.0
            self.sum_lambda_z[idx_z] += self.lambda_val

        if new_bin_idx is not None:
            self.N_visits[new_bin_idx] += 1
            n = self.N_visits[new_bin_idx]
            self.F_bar[new_bin_idx] += (fc_new - self.F_bar[new_bin_idx]) / n

        # --- STOCKAGE DANS ATOMS.INFO ---
        self.atoms.info['cv_val'] = xi_new
        self.atoms.info['lambda_val'] = self.lambda_val.copy()
        self.atoms.info['bin_idx'] = new_bin_idx

        # 5. B (Kick 2)
        forces_new = self.atoms.get_forces(md=True)
        f_bias_new = self._get_f_bias(new_bin_idx)
        f_wall_lam_new = self._get_wall_force(self.lambda_val)

        # Force TOTALE à la fin du pas pour les atomes
        f_at_total_new = forces_new - np.tensordot(fc_new, grad_xi_new, axes=1)

        # --- AJOUT : Clipping de la force totale finale sur les atomes ---
        if self.max_force is not None:
            f_at_total_new = self._clip_forces(f_at_total_new)

        p += (self.dt / 2.0) * f_at_total_new

        # Force TOTALE à la fin du pas pour lambda
        f_lam_total_new = fc_new + f_bias_new + f_wall_lam_new

        # --- AJOUT : Clipping de la force totale finale sur lambda ---
        if max_f_lam is not None:
            norm_lam_new = np.linalg.norm(f_lam_total_new)
            if norm_lam_new > max_f_lam:
                f_lam_total_new = f_lam_total_new * (max_f_lam / max(norm_lam_new, 1e-12))

        self.pi_val += (self.dt / 2.0) * f_lam_total_new

        # 6. O (Thermostat 2)
        p, self.pi_val = self._step_O(p, self.pi_val)
        self.atoms.set_momenta(p)

        self.step_counter += 1
        if self.step_counter % self.save_freq == 0:
            self.save_tensors()

        return forces_new

    def save_tensors(self):
        np.savez(
            self.save_file,
            bin_edges=self.bin_edges,
            N_visits=self.N_visits,
            F_bar=self.F_bar,
            step=self.step_counter,
            z_counts=self.z_counts,
            sum_lambda_z=self.sum_lambda_z,
            bin_edges_z=self.bin_edges_z
        )

    def _step_O(self, p, pi):
        noise_at = self.rng.standard_normal(p.shape)
        p_new = (1.0 / self.c1_at) * (self.c2_at * p + self.c3_at * noise_at)
        noise_lam = self.rng.standard_normal(self.d)
        pi_new = (1.0 / self.c1_lam) * (self.c2_lam * pi + self.c3_lam * noise_lam)
        return p_new, pi_new


class EABFProductionSegments(EABFLangevinOBABOMultiDim):
    """
    Subclass pour exécuter une simulation de production eABF à force de biais gelée.
    Permet de découper la trajectoire en segments chronologiques indépendants
    pour estimer proprement les incertitudes statistiques.
    """
    def __init__(
        self,
        atoms,
        timestep,
        temperature_K,
        friction_at,
        friction_lambda,
        mass_lambda,
        k_spring,
        cv_func,
        cv_bounds,
        n_bins,
        converged_file="eabf_data.npz",
        block_steps=100000,
        output_prefix="eabf_prod",
        fixcm=False,
        k_wall=50.0,
        max_force=None,
        max_force_lambda=None,
        **kwargs
    ):
        # On passe un nom temporaire pour éviter d'écraser le fichier de run original
        kwargs['save_file'] = "dummy_prod.npz"

        super().__init__(
            atoms=atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            friction_at=friction_at,
            friction_lambda=friction_lambda,
            mass_lambda=mass_lambda,
            k_spring=k_spring,
            cv_func=cv_func,
            cv_bounds=cv_bounds,
            n_bins=n_bins,
            fixcm=fixcm,
            k_wall=k_wall,
            max_force=max_force,
            max_force_lambda=max_force_lambda,
            **kwargs
        )

        # Chargement obligatoire du checkpoint eABF convergé
        if not os.path.exists(converged_file):
            raise FileNotFoundError(f"Fichier eABF convergé introuvable : {converged_file}")

        print(f"[Production] Chargement des forces de biais convergées depuis {converged_file}")
        data_conv = np.load(converged_file)
        self.F_bar_bias = data_conv['F_bar'].copy()

        # Configuration de la gestion des segments chronologiques
        self.block_steps = block_steps
        self.output_prefix = output_prefix
        self.block_counter = 0

        # Initialisation propre des accumulateurs du premier bloc
        self.reset_block_accumulators()

    def reset_block_accumulators(self):
        """Remet à zéro l'échantillonnage pour le bloc chronologique courant."""
        self.N_visits_block = np.zeros(self.n_bins, dtype=np.int64)
        self.F_bar_block = np.zeros(list(self.n_bins) + [self.d], dtype=np.float64)
        self.current_block_step = 0
        self.z_counts = np.zeros(self.n_bins, dtype=np.float64)
        self.sum_lambda_z = np.zeros(list(self.n_bins) + [self.d], dtype=np.float64)

    def _get_f_bias(self, bin_idx):
        """Surcharge du biais : renvoie la force convergée gelée (constante)."""
        if bin_idx is None:
            return np.zeros(self.d)
        # On utilise le profil gelé de la simulation convergée.
        # alpha est implicitement égal à 1 car l'apprentissage est fini.
        return -self.F_bar_bias[bin_idx]

    def save_tensors(self):
        """Désactive la méthode d'écriture globale automatique de la classe mère."""
        pass

    def step(self, forces=None):
        """Exécute un pas de dynamique en accumulant la force libre dans le bloc courant."""
        p = self.atoms.get_momenta()
        masses = self.atoms.get_masses()[:, np.newaxis]

        if forces is None:
            forces = self.atoms.get_forces(md=True)

        # 1. O (Thermostat)
        p, self.pi_val = self._step_O(p, self.pi_val)

        # 2. B (Kick)
        xi_val, grad_xi = self.cv_func(self.atoms)
        bin_idx = self._get_bin_indices(self.lambda_val)

        fc = self.k_spring * (xi_val - self.lambda_val)
        f_bias = self._get_f_bias(bin_idx)
        f_wall_lam = self._get_wall_force(self.lambda_val)

        f_at_total = forces - np.tensordot(fc, grad_xi, axes=1)
        p += (self.dt / 2.0) * f_at_total
        self.pi_val += (self.dt / 2.0) * (fc + f_bias + f_wall_lam)

        # 3. A (Drift)
        q = self.atoms.get_positions()
        q += self.dt * (p / masses)
        self.lambda_val += self.dt * (self.pi_val / self.mass_lam)
        if self.fix_com:
            self.atoms.set_positions(q, apply_constraint=True)
        else:
            self.atoms.set_positions(q, apply_constraint=False)

        # 4. Apprentissage LOCAL (uniquement pour le bloc en cours)
        xi_new, grad_xi_new = self.cv_func(self.atoms)
        fc_new = self.k_spring * (xi_new - self.lambda_val)
        new_bin_idx = self._get_bin_indices(self.lambda_val)
        idx_z = self._get_bin_indices(xi_new)
        if idx_z is not None:
            self.z_counts[idx_z] += 1.0
            self.sum_lambda_z[idx_z] += self.lambda_val

        if new_bin_idx is not None:
            self.N_visits_block[new_bin_idx] += 1
            n = self.N_visits_block[new_bin_idx]
            self.F_bar_block[new_bin_idx] += (fc_new - self.F_bar_block[new_bin_idx]) / n

        # Mise à jour de atoms.info pour l'écriture de la trajectoire d'ASE
        self.atoms.info['cv_val'] = xi_new
        self.atoms.info['lambda_val'] = self.lambda_val.copy()
        self.atoms.info['bin_idx'] = new_bin_idx

        # 5. B (Kick 2)
        forces_new = self.atoms.get_forces(md=True)
        f_bias_new = self._get_f_bias(new_bin_idx)
        f_wall_lam_new = self._get_wall_force(self.lambda_val)

        f_at_total_new = forces_new - np.tensordot(fc_new, grad_xi_new, axes=1)
        p += (self.dt / 2.0) * f_at_total_new
        self.pi_val += (self.dt / 2.0) * (fc_new + f_bias_new + f_wall_lam_new)

        # 6. O (Thermostat 2)
        p, self.pi_val = self._step_O(p, self.pi_val)
        self.atoms.set_momenta(p)

        self.step_counter += 1
        self.current_block_step += 1

        # Vérification de la fin du segment courant
        if self.current_block_step >= self.block_steps:
            self.save_block_tensors()
            self.reset_block_accumulators()
            self.block_counter += 1

        return forces_new

    def save_block_tensors(self):
        """Écrit les tenseurs du bloc actuel au format compatible posttreatment.py."""
        filename = f"{self.output_prefix}_block_{self.block_counter}.npz"
        print(f"[Production] Fin du segment {self.block_counter}. Sauvegarde dans : {filename}")
        np.savez(
            filename,
            bin_edges=self.bin_edges,
            N_visits=self.N_visits_block,
            F_bar=self.F_bar_block,
            step=self.current_block_step,
            z_counts=self.z_counts,
            sum_lambda_z=self.sum_lambda_z,
            bin_edges_z=self.bin_edges_z
        )

