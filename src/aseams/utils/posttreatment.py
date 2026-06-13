import numpy as np
from scipy.sparse import coo_matrix, linalg
import matplotlib.pyplot as plt
from ase.io import Trajectory


def block_bootstrap_mean(data, n_boot=1000, block_size=1, rng=None):
    """
    Bootstrap par blocs pour estimer la moyenne et son incertitude.

    On découpe la série en blocs contigus de longueur `block_size`,
    puis on rééchantillonne ces blocs avec remise pour construire
    des séries "bootstrapées" de même longueur totale.

    Paramètres
    ----------
    data : np.ndarray, shape (N,)
        Série scalaire (une observable en fonction du temps).
    n_boot : int
        Nombre de rééchantillonnages bootstrap.
    block_size : int
        Taille des blocs (en nombre de pas) pour le bootstrap par blocs.
        block_size = 1 revient à un bootstrap iid (non conseillé si
        corrélations fortes).
    rng : np.random.Generator ou None
        Générateur de nombres aléatoires. Si None, np.random.default_rng().

    Renvoie
    -------
    mean_hat : float
        Moyenne empirique de `data`.
    std_boot : float
        Ecart-type bootstrap de la moyenne (erreur statistique estimée).
    boot_samples : np.ndarray, shape (n_boot,)
        Les moyennes bootstrap (facultatif pour la suite, mais utile si tu
        veux inspecter la distribution).
    """
    data = np.asarray(data, dtype=float)
    N = data.size
    if N == 0:
        raise ValueError("block_bootstrap_mean: data vide.")

    if block_size <= 0:
        raise ValueError("block_size doit être >= 1.")
    if block_size > N:
        raise ValueError("block_size ne peut pas dépasser la taille des données.")

    if rng is None:
        rng = np.random.default_rng()

    # Moyenne empirique de la série originale
    mean_hat = data.mean()

    # Construction des blocs contigus
    n_blocks = N // block_size
    if n_blocks == 0:
        # si block_size > N/2 etc : utiliser un bloc unique
        block_size = N
        n_blocks = 1

    # Tronquer pour tomber juste
    truncated = data[: n_blocks * block_size]
    blocks = truncated.reshape(n_blocks, block_size)

    boot_means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        # Rééchantillonnage avec remise des indices de blocs
        idx = rng.integers(0, n_blocks, size=n_blocks)
        sample = blocks[idx].reshape(-1)
        boot_means[i] = sample.mean()

    std_boot = boot_means.std(ddof=1)

    return mean_hat, std_boot, boot_means


def extract_series_from_traj(traj, discard=0.2, key_G="G_M"):
    """
    Extrait, après thermalisation, les séries lambda_half, lambda_3quarter
    et G_M (ou autre clé) d'une trajectoire ASE.

    Paramètres
    ----------
    traj : ase.io.Trajectory ou iterable d'ase.Atoms
        Trajectoire issue de BlueMoonOBABOWithLambdas à z fixé.
    discard : float
        Fraction initiale des pas à jeter (équilibration).
    key_G : str
        Clé dans atoms.info pour la quantité G_M(q).
        Par défaut "blue_moon_Z" (alias de "G_M" dans l'intégrateur).

    Renvoie
    -------
    lambda_half : np.ndarray, shape (N_k,)
        Série temporelle de λ^{n+1/2} après thermalisation.
    lambda_3q : np.ndarray, shape (N_k,)
        Série temporelle de λ^{n+3/4} après thermalisation.
    G_M : np.ndarray, shape (N_k,)
        Série temporelle de G_M(q) = ∇xi^T M^{-1} ∇xi (coordonnée scalaire).
    """
    # Rendre la trajectoire itérable une seule fois
    frames = list(traj)

    n = len(frames)
    if n == 0:
        raise ValueError("Trajectoire vide.")

    i0 = int(discard * n)
    if i0 >= n:
        raise ValueError(
            f"discard trop grand : discard={discard}, n={n} => i0={i0} >= n."
        )

    lambda_half = np.array([fr.info["lambda_half"] for fr in frames[i0:]], dtype=float)
    lambda_3q   = np.array([fr.info["lambda_3quarter"] for fr in frames[i0:]], dtype=float)

    try:
        G_M = np.array([fr.info[key_G] for fr in frames[i0:]], dtype=float)
    except KeyError:
        raise KeyError(f"Clé '{key_G}' absente de atoms.info dans la trajectoire.")

    return lambda_half, lambda_3q, G_M

def estimate_window_observables(
    traj,
    dt,
    discard=0.2,
    key_G="blue_moon_Z",
    n_boot=0,
    block_size=1,
    rng=None,
):
    """
    Calcule, pour une fenêtre (un z fixé), les observables nécessaires
    + éventuellement leurs erreurs par bootstrap.

    Observables :
        - f_rgd(z) ≈ ∂_z F_rgd^M(z) (force moyenne rigide),
          via (3.181) :
              f_rgd(z) ≈ (1 / (N Δt)) ∑_n [λ^{n+1/2} + λ^{n+3/4}]

        - C_geom(z) ≈ ⟨ G_M(q)^{-1/2} ⟩_{ν^M_{Σ(z)}}

    Si n_boot > 0, on effectue un bootstrap par blocs pour estimer
    les erreurs statistiques sur f_rgd et C_geom.

    Paramètres
    ----------
    traj : Trajectory ou iterable d'Atoms
    dt : float
    discard : float
    key_G : str
    n_boot : int
        Nombre de rééchantillonnages bootstrap (0 => pas de bootstrap).
    block_size : int
        Taille des blocs pour le bootstrap par blocs.
    rng : np.random.Generator ou None

    Renvoie
    -------
    f_rgd : float
    C_geom : float
    n_samples : int
    f_rgd_err : float
        Erreur (écart-type) bootstrap sur f_rgd (0 si n_boot=0).
    C_geom_err : float
        Erreur (écart-type) bootstrap sur C_geom (0 si n_boot=0).
    """
    lambda_half, lambda_3q, G_M = extract_series_from_traj(
        traj, discard=discard, key_G=key_G
    )

    n_samples = len(lambda_half)
    if n_samples == 0:
        raise ValueError("Aucun échantillon après discard.")

    if np.any(G_M <= 0.0):
        raise ValueError(
            "G_M contient des valeurs non positives, impossible de prendre la racine."
        )

    # --- (3.181) : force moyenne rigide ---
    lambda_sum = lambda_half + lambda_3q  # série du numérateur de (3.181)
    # moyenne "directe"
    f_rgd = lambda_sum.mean() / dt

    # --- Correction géométrique : moyenne de G_M^{-1/2} ---
    G_inv_sqrt = G_M ** (-0.5)
    C_geom = G_inv_sqrt.mean()

    # Initialiser les erreurs à 0 par défaut
    f_rgd_err = 0.0
    C_geom_err = 0.0

    if n_boot > 0:
        if rng is None:
            rng = np.random.default_rng()

        # Bootstrap pour lambda_sum
        mean_lambda, std_lambda, _ = block_bootstrap_mean(
            lambda_sum, n_boot=n_boot, block_size=block_size, rng=rng
        )
        # f_rgd est la moyenne de lambda_sum divisée par dt
        # on suppose dt exact => propager simplement l'erreur :
        f_rgd = mean_lambda / dt
        f_rgd_err = std_lambda / dt

        # Bootstrap pour G_inv_sqrt
        mean_C, std_C, _ = block_bootstrap_mean(
            G_inv_sqrt, n_boot=n_boot, block_size=block_size, rng=rng
        )
        C_geom = mean_C
        C_geom_err = std_C

    return f_rgd, C_geom, n_samples, f_rgd_err, C_geom_err

def integrate_mean_force(z_grid, f_rgd, f_rgd_err=None, z_ref=None):
    """
    Intègre la force moyenne rigide f_rgd(z) pour obtenir un profil
    d'énergie libre rigide F_rgd^M(z) et, optionnellement, son erreur.

    Si f_rgd_err est fourni, on propage l'incertitude par quadrature
    linéaire dans la règle des trapèzes.

    Paramètres
    ----------
    z_grid : array-like, shape (K,)
    f_rgd : array-like, shape (K,)
    f_rgd_err : array-like ou None, shape (K,)
        Erreurs (écarts-types) sur f_rgd(z_k). Si None, pas d'erreur propagée.
    z_ref : float ou None

    Renvoie
    -------
    z_sorted : np.ndarray, shape (K,)
    F_rgd : np.ndarray, shape (K,)
    idx_sort : np.ndarray, shape (K,)
    F_rgd_err : np.ndarray, shape (K,)
        Erreurs sur F_rgd (array de zéros si f_rgd_err is None).
    """
    z_grid = np.asarray(z_grid, dtype=float)
    f_rgd = np.asarray(f_rgd, dtype=float)

    if z_grid.shape != f_rgd.shape:
        raise ValueError("z_grid et f_rgd doivent avoir la même forme.")

    if f_rgd_err is not None:
        f_rgd_err = np.asarray(f_rgd_err, dtype=float)
        if f_rgd_err.shape != f_rgd.shape:
            raise ValueError("f_rgd_err doit avoir la même forme que f_rgd.")

    idx_sort = np.argsort(z_grid)
    z_sorted = z_grid[idx_sort]
    f_sorted = f_rgd[idx_sort]
    f_err_sorted = None
    if f_rgd_err is not None:
        f_err_sorted = f_rgd_err[idx_sort]

    K = len(z_sorted)
    F_rgd = np.zeros(K, dtype=float)
    F_rgd_err = np.zeros(K, dtype=float)

    # Intégration (trapèzes) + propagation d'erreur
    for k in range(1, K):
        dz = z_sorted[k] - z_sorted[k - 1]
        # contribution moyenne à F(k) - F(k-1)
        F_rgd[k] = F_rgd[k - 1] + 0.5 * (f_sorted[k - 1] + f_sorted[k]) * dz

        if f_err_sorted is not None:
            # ΔF = 0.5 dz (f_{k-1} + f_k)
            # variance(ΔF) ≈ (0.5 dz)^2 (σ_{k-1}^2 + σ_k^2), en négligeant corrélations
            coeff = 0.5 * dz
            var_delta = (coeff**2) * (
                f_err_sorted[k - 1] ** 2 + f_err_sorted[k] ** 2
            )
            F_rgd_err[k] = np.sqrt(F_rgd_err[k - 1] ** 2 + var_delta)

    # Fixer la référence F_rgd(z_ref) = 0
    if z_ref is None:
        ref_index = 0
    else:
        ref_index = int(np.argmin(np.abs(z_sorted - z_ref)))

    F_shift = F_rgd[ref_index]
    F_rgd -= F_shift
    # l'incertitude relative ne change pas (on enlève une constante exacte)
    # si tu veux, tu peux aussi poser F_rgd_err[ref_index] = 0 explicitement
    # pour signifier que la référence est fixée.
    F_rgd_err[ref_index] = 0.0

    return z_sorted, F_rgd, idx_sort, F_rgd_err


def build_free_energy_profile(
    traj_list,
    z_grid,
    dt,
    kBT,
    discard=0.2,
    key_G="G_M",
    z_ref=None,
    n_boot=0,
    block_size=1,
    rng=None,
):
    """
    Construit le profil d'énergie libre rigide F_rgd^M(z) et le profil
    d'énergie libre global F(z), avec estimation d'erreurs statistiques
    optionnelle (bootstrap par blocs).

    Paramètres
    ----------
    traj_list : list of Trajectory or list of iterables de Atoms
    z_grid : array-like, shape (K,)
    dt : float
    kBT : float
    discard : float
    key_G : str
    z_ref : float ou None
    n_boot : int
        Nombre de rééchantillonnages bootstrap par fenêtre (0 => pas d'erreur).
    block_size : int
        Taille de blocs pour le bootstrap par blocs dans chaque fenêtre.
    rng : np.random.Generator ou None

    Renvoie
    -------
    z_sorted : np.ndarray, shape (K,)
    mean_force_rgd : np.ndarray, shape (K,)
    F_rgd : np.ndarray, shape (K,)
    C_geom : np.ndarray, shape (K,)
    F_global : np.ndarray, shape (K,)
    mean_force_rgd_err : np.ndarray, shape (K,)
    F_rgd_err : np.ndarray, shape (K,)
    C_geom_err : np.ndarray, shape (K,)
    F_global_err : np.ndarray, shape (K,)
    """
    z_grid = np.asarray(z_grid, dtype=float)
    K = len(z_grid)

    if len(traj_list) != K:
        raise ValueError("traj_list et z_grid doivent avoir la même longueur.")

    beta = 1.0 / kBT

    mean_force_rgd = np.zeros(K, dtype=float)
    C_geom = np.zeros(K, dtype=float)
    n_samples = np.zeros(K, dtype=int)

    mean_force_rgd_err = np.zeros(K, dtype=float)
    C_geom_err = np.zeros(K, dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    # --- 1. Boucle sur les fenêtres ---
    for i, traj in enumerate(traj_list):
        f_rgd_i, C_i, n_i, f_err_i, C_err_i = estimate_window_observables(
            traj,
            dt=dt,
            discard=discard,
            key_G=key_G,
            n_boot=n_boot,
            block_size=block_size,
            rng=rng,
        )
        mean_force_rgd[i] = f_rgd_i
        C_geom[i] = C_i
        n_samples[i] = n_i
        mean_force_rgd_err[i] = f_err_i
        C_geom_err[i] = C_err_i

    # --- 2. Intégration thermodynamique de la force moyenne rigide ---
    z_sorted, F_rgd, idx_sort, F_rgd_err = integrate_mean_force(
        z_grid,
        mean_force_rgd,
        f_rgd_err=mean_force_rgd_err if n_boot > 0 else None,
        z_ref=z_ref,
    )

    # Réordonner C_geom et C_geom_err comme z_sorted
    C_sorted = C_geom[idx_sort]
    C_err_sorted = C_geom_err[idx_sort]
    mf_rgd_sorted = mean_force_rgd[idx_sort]
    mf_rgd_err_sorted = mean_force_rgd_err[idx_sort]

    # --- 3. Correction géométrique (3.168) + erreurs ---
    if z_ref is None:
        ref_index = 0
    else:
        ref_index = int(np.argmin(np.abs(z_sorted - z_ref)))

    logC = np.log(C_sorted)
    F_global = F_rgd - (1.0 / beta) * (logC - logC[ref_index])

    # Erreurs sur F_global
    F_global_err = np.zeros_like(F_global)

    if n_boot > 0:
        # erreur sur logC : σ_logC ≈ σ_C / C
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_logC = np.where(
                C_sorted > 0.0, C_err_sorted / C_sorted, 0.0
            )

        # variance de (logC_k - logC_ref) :
        # supposée = σ_logC_k^2 + σ_logC_ref^2 (indépendance)
        var_log_ratio = (
            sigma_logC**2 + sigma_logC[ref_index] ** 2
        )  # shape (K,)

        # variance sur F_global :
        # Var(F_global_k) = Var(F_rgd_k) + (1/β^2) Var(logC_k - logC_ref)
        F_global_err = np.sqrt(
            F_rgd_err**2 + (1.0 / beta**2) * var_log_ratio
        )

        # par construction, F_global_err[ref_index] est non nul à cause de logC_ref,
        # mais comme on impose F(z_ref)=0, tu peux si tu veux le fixer à 0 :
        # F_global_err[ref_index] = 0.0

    return (
        z_sorted,
        mf_rgd_sorted,
        F_rgd,
        C_sorted,
        F_global,
        mf_rgd_err_sorted,
        F_rgd_err,
        C_err_sorted,
        F_global_err,
    )

def build_free_energy_profile_from_npz(
    npz_paths,
    z_grid,
    dt,
    kBT,
    z_ref=None,
    n_boot=1000,
    block_size=50,
    rng=None
):
    K = len(z_grid)  # Doit être égal à 40 ici
    beta = 1.0 / kBT
    
    mean_force_rgd = np.zeros(K)
    C_geom = np.zeros(K)
    mean_force_rgd_err = np.zeros(K)
    C_geom_err = np.zeros(K)
    
    if rng is None:
        rng = np.random.default_rng()
        
    # --- 1. Boucle de lecture dans les répertoires i (0 à 39) ---
    for i, npz_path in enumerate(npz_paths):
        
        # Chargement instantané du fichier npz
        data = np.load(npz_path)
        
        # Extraction et reconstruction des observables à partir des clés enregistrées
        lambda_sum = data['lambda_half'] + data['lambda_3quarter']
        G_inv_sqrt = data['G_M'] ** (-0.5)
        
        # Calcul des moyennes directes
        f_rgd = lambda_sum.mean() / dt
        C_i = G_inv_sqrt.mean()
        
        f_err = 0.0
        C_err = 0.0
        
        # Bootstrap par blocs si n_boot > 0
        if n_boot > 0:
            _, std_lambda, _ = block_bootstrap_mean(lambda_sum, n_boot, block_size, rng)
            f_err = std_lambda / dt
            
            _, std_C, _ = block_bootstrap_mean(G_inv_sqrt, n_boot, block_size, rng)
            C_err = std_C
            
        mean_force_rgd[i] = f_rgd
        C_geom[i] = C_i
        mean_force_rgd_err[i] = f_err
        C_geom_err[i] = C_err

    # --- 2. Intégration thermodynamique par la méthode des trapèzes ---
    z_sorted, F_rgd, idx_sort, F_rgd_err = integrate_mean_force(
        z_grid,
        mean_force_rgd,
        f_rgd_err=mean_force_rgd_err if n_boot > 0 else None,
        z_ref=z_ref
    )
    
    # Réordonner les facteurs géométriques selon le tri de la grille z
    C_sorted = C_geom[idx_sort]
    C_err_sorted = C_geom_err[idx_sort]
    
    # Recherche de l'index de référence
    if z_ref is None:
        ref_index = 0
    else:
        ref_index = int(np.argmin(np.abs(z_sorted - z_ref)))
        
    # --- 3. Application de la correction géométrique globale ---
    logC = np.log(C_sorted)
    F_global = F_rgd - (1.0 / beta) * (logC - logC[ref_index])
    
    # Propagation des incertitudes sur le profil global
    F_global_err = np.zeros_like(F_global)
    if n_boot > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_logC = np.where(C_sorted > 0.0, C_err_sorted / C_sorted, 0.0)
        var_log_ratio = sigma_logC**2 + sigma_logC[ref_index] ** 2
        F_global_err = np.sqrt(F_rgd_err**2 + (1.0 / beta**2) * var_log_ratio)
        
    return (
        z_sorted,
        mean_force_rgd[idx_sort],
        F_rgd,
        C_sorted,
        F_global,
        mean_force_rgd_err[idx_sort],
        F_rgd_err,
        C_err_sorted,
        F_global_err
    )


def compute_tst_rate(z_sorted, F_global, C_sorted, kBT, z_ts, z_reactant_range=None):
    """
    Calcule le taux de la théorie de l'état de transition (TST) à partir du profil
    d'énergie libre globale et des facteurs de correction géométrique issus du Blue-Moon sampling.

    Paramètres
    ----------
    z_sorted : np.ndarray
        Grille ordonnée de la coordonnée de réaction (z).
    F_global : np.ndarray
        Profil d'énergie libre globale F(z) (en eV), corrigé géométriquement.
    C_sorted : np.ndarray
        Moyenne de G_M^{-1/2} pour chaque fenêtre (<G_M^{-1/2}>).
    kBT : float
        Énergie thermique k_B * T (dans les mêmes unités que F, typiquement eV).
    z_ts : float
        Valeur de z correspondant à l'état de transition (Transition State, TS).
    z_reactant_range : tuple (float, float) ou None
        Bornes (z_min, z_max) définissant le puits des réactifs pour l'intégration.
        Si None, l'intégration se fait du début de la grille (z_sorted[0]) jusqu'à z_ts.

    Renvoie
    -------
    rate_ase : float
        Taux de réaction TST exprimé dans l'unité de temps interne d'ASE (inverse du temps d'ASE).
    """
    beta = 1.0 / kBT

    # 1. Trouver l'index correspondant à l'état de transition (TS)
    idx_ts = np.argmin(np.abs(z_sorted - z_ts))
    F_ts = F_global[idx_ts]
    C_ts = C_sorted[idx_ts] # C_ts vaut exactement <G_M^{-1/2}> au TS

    # 2. Définir la plage du puits des réactifs pour le calcul de la fonction de partition
    if z_reactant_range is not None:
        z_min, z_max = z_reactant_range
        mask_reactant = (z_sorted >= z_min) & (z_sorted <= z_max)
    else:
        # Par défaut, on intègre de -infini (début de la grille) jusqu'au TS
        mask_reactant = z_sorted <= z_sorted[idx_ts]

    z_part = z_sorted[mask_reactant]
    F_part = F_global[mask_reactant]

    if len(z_part) < 2:
        raise ValueError("La plage des réactifs spécifiée contient moins de 2 points. Impossible d'intégrer.")

    # 3. Calcul du dénominateur : intégration de exp(-\beta * F(z)) sur le puits réactif
    # Utilisation d'une formule des trapèzes explicite pour éviter les warnings de numpy 2.0
    integrand = np.exp(-beta * F_part)
    dz = np.diff(z_part)
    denom_integral = 0.5 * np.sum((integrand[:-1] + integrand[1:]) * dz)

    # 4. Application de la formule TST fournie
    # Prefactor = 1 / sqrt(2 * pi * beta)
    prefactor = 1.0 / np.sqrt(2.0 * np.pi * beta)
    
    # Terme lié à la correction géométrique au TS : ( \int_{\Sigma(z_TS)} G_M^{-1/2} \nu^M )^{-1}
    # Ce qui équivaut exactement à 1 / C_ts
    inv_C_ts = 1.0 / C_ts
    
    # Terme exponentiel du numérateur
    exp_numerator = np.exp(-beta * F_ts)

    # Calcul final du taux en unités de temps internes d'ASE
    rate_ase = prefactor * inv_C_ts * (exp_numerator / denom_integral)

    return rate_ase

class EABFAnalyseurRigoureux:
    def __init__(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.N_visits = data['N_visits']
        self.F_bar = data['F_bar']
        self.bin_edges = data['bin_edges']
        self.d = len(self.bin_edges)
        
        self.bin_centers = [(e[:-1] + e[1:]) / 2.0 for e in self.bin_edges]
        self.shape = self.N_visits.shape
        self.dx = [c[1] - c[0] for c in self.bin_centers]

    def _get_divergence(self, W):
        """Calcul de -div(W * F_bar) pour n'importe quelle dimension."""
        div_WF = np.zeros(self.shape)
        # On itère sur chaque dimension de la CV
        for i in range(self.d):
            WF_i = W * self.F_bar[..., i]
            # Différences finies centrées pour la divergence
            grad_WF_i = np.gradient(WF_i, self.dx[i], axis=i)
            div_WF += grad_WF_i
        return -div_WF.flatten()

    def reconstruct_fes_1d(self, tolerance=1e-8):
        """Reconstruction rigoureuse en 1D avec schéma de divergence compatible."""
        if self.d != 1:
            raise ValueError("Le fichier chargé n'est pas en 1D.")
        
        nx = self.shape[0]
        dx = self.dx[0]
        W = self.N_visits.astype(float)
        mask = W > 0
        W[~mask] = 1e-10
        
        if self.F_bar.ndim == 2:
            F_1d = self.F_bar[:, 0]
        else:
            F_1d = self.F_bar

        # RECONSTRUCTION DU RHS COMPATIBLE AVEC L : on applique ici -div(W * F)
        rhs = np.zeros(nx)
        data, rows, cols = [], [], []
        
        for i in range(nx):
            # Lien Gauche (i-1 <-> i)
            if i > 0:
                w_avg = (W[i] + W[i-1]) / 2.0
                val = w_avg / dx**2
                data.extend([-val, val]); rows.extend([i, i]); cols.extend([i, i-1])
                
                # Contribution à -div(W * F) -> Signe inversé (+)
                F_edge = (F_1d[i] + F_1d[i-1]) / 2.0
                rhs[i] += (w_avg * F_edge) / dx
                
            # Lien Droite (i <-> i+1)
            if i < nx - 1:
                w_avg = (W[i] + W[i+1]) / 2.0
                val = w_avg / dx**2
                data.extend([-val, val]); rows.extend([i, i]); cols.extend([i, i+1])
                
                # Contribution à -div(W * F) -> Signe inversé (-)
                F_edge = (F_1d[i] + F_1d[i+1]) / 2.0
                rhs[i] -= (w_avg * F_edge) / dx
        
        L = coo_matrix((data, (rows, cols)), shape=(nx, nx)).tocsr()
        res, _ = linalg.cg(L, rhs, atol=tolerance)
        
        # Calage du minimum des zones échantillonnées à 0 eV
        fes = res - np.min(res[mask])
        fes[~mask] = np.nan
        return self.bin_centers[0], fes

    def calculer_pmf_czar(npz_path, temperature_K, k_spring):
        """
        Calcule le profil d'énergie libre (PMF) via l'estimateur CZAR.
    
        Paramètres
        ----------
        npz_path : str
            Chemin vers le fichier de données .npz contenant les histogrammes.
        temperature_K : float
            Température de la simulation en Kelvin.
        k_spring : float
            Constante de raideur du ressort de couplage eABF (en eV/Å^2 ou unités équivalentes).
        """
        from ase import units
        import numpy as np
        from scipy.integrate import cumulative_trapezoid
   
        # 1. Chargement des données
        data = np.load(npz_path)
        z_counts = data['z_counts']
        sum_lambda_z = data['sum_lambda_z']
        bin_edges_z = data['bin_edges_z']
    
        beta = 1.0 / (units.kB * temperature_K)
        z_centers = (bin_edges_z[:-1] + bin_edges_z[1:]) / 2.0
        dz = bin_edges_z[1] - bin_edges_z[0]
    
        # Masque pour éviter les divisions par zéro dans les zones non échantillonnées
        sampled_mask = z_counts > 5 # Seuil minimal de statistiques
     
        # 2. Calcul de la moyenne conditionnelle <lambda>_z
        lambda_avg_z = np.zeros_like(z_centers)
        lambda_avg_z[sampled_mask] = sum_lambda_z[sampled_mask] / z_counts[sampled_mask]
      
        # Premier terme du gradient : Force moyenne du ressort de rappel
        force_spring = k_spring * (lambda_avg_z - z_centers)
    
        # 3. Calcul du terme statistique : -1/beta * d(ln rho_tilde)/dz
        ln_rho = np.zeros_like(z_centers)
        # Régularisation locale pour le calcul du log
        ln_rho[sampled_mask] = np.log(z_counts[sampled_mask])
    
        # Évaluation numérique du gradient de ln(rho) par différences centrales
        dln_rho_dz = np.zeros_like(z_centers)
    
        # Points intérieurs
        dln_rho_dz[1:-1] = (ln_rho[2:] - ln_rho[:-2]) / (2.0 * dz)
        # Bords (différences asymétriques)
        if len(z_centers) > 1:
            dln_rho_dz[0] = (ln_rho[1] - ln_rho[0]) / dz
            dln_rho_dz[-1] = (ln_rho[-1] - ln_rho[-2]) / dz
        
        # Terme correctif entropique thermique
        force_entropic = - (1.0 / beta) * dln_rho_dz
    
        # 4. Gradient CZAR total complet : A'(z)
        A_prime = np.zeros_like(z_centers)
        A_prime[sampled_mask] = force_spring[sampled_mask] + force_entropic[sampled_mask]
    
        # 5. Intégration numérique pour obtenir le PMF A(z)
        A = np.zeros_like(z_centers)
        # On intègre le gradient le long de la grille
        A[1:] = cumulative_trapezoid(A_prime, z_centers, initial=0)
    
        # Ajustement cosmétique : fixe le minimum du profil à 0 eV
        if np.any(sampled_mask):
            A[sampled_mask] -= np.min(A[sampled_mask])
            A[~sampled_mask] = np.nan # Assigne NaN aux zones non explorées
        
        return z_centers, A, A_prime

    def reconstruct_czar_fes_1d(self, filename, temperature_K, k_spring, tolerance=1e-8):
        """
        Reconstruction rigoureuse du PMF CZAR en 1D via l'équation de Poisson
        pondérée par l'histogramme de la variable physique z.
        """
        from ase import units
        import numpy as np
        from scipy.sparse import coo_matrix, linalg

        # 1. Chargement des données spécifiques CZAR depuis le fichier .npz
        data = np.load(self.filename)
        if 'z_counts' not in data or 'sum_lambda_z' not in data:
            raise KeyError("Le fichier ne contient pas les clés CZAR ('z_counts', 'sum_lambda_z').")

        z_counts = data['z_counts'].astype(float)
        sum_lambda_z = data['sum_lambda_z'].astype(float)
        bin_edges_z = data['bin_edges_z']
        
        nx = len(z_counts)
        z_centers = (bin_edges_z[:-1] + bin_edges_z[1:]) / 2.0
        dx = z_centers[1] - z_centers[0]
        beta = 1.0 / (units.kB * temperature_K)
        
        # Définition du poids statistique W (histogramme de la CV physique z)
        mask = z_counts > 5  # Seuil statistique minimal
        W = z_counts.copy()
        W[~mask] = 1e-10     # Évite les matrices singulières sur les bords non échantillonnés
        
        # 2. Calcul de la force CZAR brute aux centres des bins : F_czar = -A'(z)
        lambda_avg_z = np.zeros_like(z_centers)
        lambda_avg_z[mask] = sum_lambda_z[mask] / z_counts[mask]
        
        # Terme du ressort : k * (z - <lambda>_z)
        f_spring = k_spring * (z_centers - lambda_avg_z)
        
        # Terme entropique dérivé de l'histogramme : (1/beta) * d(ln rho_tilde)/dz
        ln_rho = np.zeros_like(z_centers)
        ln_rho[mask] = np.log(z_counts[mask])
        dln_rho_dz = np.zeros_like(z_centers)
        
        if nx > 1:
            dln_rho_dz[1:-1] = (ln_rho[2:] - ln_rho[:-2]) / (2.0 * dx)
            dln_rho_dz[0] = (ln_rho[1] - ln_rho[0]) / dx
            dln_rho_dz[-1] = (ln_rho[-1] - ln_rho[-2]) / dx
            
        f_entropic = (1.0 / beta) * dln_rho_dz
        
        # Force CZAR totale (f_czar = -dA/dz)
        f_czar = f_spring + f_entropic
        f_czar[~mask] = 0.0  # Annulation de la force hors échantillonnage
        
        # 3. Assemblage du problème de Poisson : -div(W * grad A) = -div(W * F_czar)
        rhs = np.zeros(nx)
        data_matrix, rows, cols = [], [], []
        
        for i in range(nx):
            # Lien Gauche (i-1 <-> i)
            if i > 0:
                w_avg = (W[i] + W[i-1]) / 2.0
                val = w_avg / dx**2
                data_matrix.extend([-val, val]); rows.extend([i, i]); cols.extend([i, i-1])
                
                f_edge = (f_czar[i] + f_czar[i-1]) / 2.0
                rhs[i] += (w_avg * f_edge) / dx
                
            # Lien Droite (i <-> i+1)
            if i < nx - 1:
                w_avg = (W[i] + W[i+1]) / 2.0
                val = w_avg / dx**2
                data_matrix.extend([-val, val]); rows.extend([i, i]); cols.extend([i, i+1])
                
                f_edge = (f_czar[i] + f_czar[i+1]) / 2.0
                rhs[i] -= (w_avg * f_edge) / dx
        
        # Résolution du système linéaire creux par gradient conjugué
        L = coo_matrix((data_matrix, (rows, cols)), shape=(nx, nx)).tocsr()
        res, _ = linalg.cg(L, rhs, atol=tolerance)
        
        # Calage du minimum global de la FES à 0 eV
        fes = res - np.min(res[mask])
        fes[~mask] = np.nan
        
        return z_centers, fes

    def reconstruct_fes_2d(self, tolerance=1e-6):
        """Reconstruction rigoureuse en 2D."""
        if self.d != 2:
            raise ValueError("Le fichier chargé n'est pas en 2D.")

        nx, ny = self.shape
        dx, dy = self.dx
        W = self.N_visits.astype(float)
        mask = W > 0
        W[~mask] = 1e-10
        rhs = self._get_divergence(W)

        data, rows, cols = [], [], []
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                # X-links
                if i > 0:
                    w = (W[i,j] + W[i-1,j]) / 2.0
                    val = w / dx**2
                    data.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx-ny])
                if i < nx - 1:
                    w = (W[i,j] + W[i+1,j]) / 2.0
                    val = w / dx**2
                    data.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx+ny])
                # Y-links
                if j > 0:
                    w = (W[i,j] + W[i,j-1]) / 2.0
                    val = w / dy**2
                    data.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx-1])
                if j < ny - 1:
                    w = (W[i,j] + W[i,j+1]) / 2.0
                    val = w / dy**2
                    data.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx+1])

        L = coo_matrix((data, (rows, cols)), shape=(nx*ny, nx*ny)).tocsr()
        res, _ = linalg.cg(L, rhs, tol=tolerance)
        fes = res.reshape(self.shape)
        fes -= np.min(fes[mask])
        fes[~mask] = np.nan
        return self.bin_centers, fes

    def reconstruct_czar_fes_2d(self, filename, temperature_K, k_springs, tolerance=1e-8):
        """
        Reconstruction du PMF CZAR en 2D via l'équation de Poisson.
        
        k_springs : tuple ou array de taille 2 (k_x, k_y)
        """
        from ase import units
        import numpy as np
        from scipy.sparse import coo_matrix, linalg

        data = np.load(filename)
        z_counts = data['z_counts'].astype(float)      # Shape: (nx, ny)
        sum_lambda_z = data['sum_lambda_z'].astype(float)  # Shape: (2, nx, ny)
        bin_edges_x = data['bin_edges_x']
        bin_edges_y = data['bin_edges_y']

        nx, ny = z_counts.shape
        x_centers = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2.0
        y_centers = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2.0
        dx = x_centers[1] - x_centers[0]
        dy = y_centers[1] - y_centers[0]
        beta = 1.0 / (units.kB * temperature_K)

        mask = z_counts > 5
        W = z_counts.copy()
        W[~mask] = 1e-10

        # Maillage pour les coordonnées physiques
        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

        # Moyennes conditionnelles <lambda_x> et <lambda_y>
        lambda_avg_x = np.zeros_like(z_counts)
        lambda_avg_y = np.zeros_like(z_counts)
        lambda_avg_x[mask] = sum_lambda_z[0][mask] / z_counts[mask]
        lambda_avg_y[mask] = sum_lambda_z[1][mask] / z_counts[mask]

        # Forces des ressorts
        f_spring_x = k_springs[0] * (X - lambda_avg_x)
        f_spring_y = k_springs[1] * (Y - lambda_avg_y)

        # Terme entropique par gradients numériques multidimensionnels
        ln_rho = np.log(W)
        dln_rho_dx, dln_rho_dy = np.gradient(ln_rho, dx, dy, edge_order=1)
        
        f_entropic_x = (1.0 / beta) * dln_rho_dx
        f_entropic_y = (1.0 / beta) * dln_rho_dy

        # Forces CZAR totales
        f_czar_x = f_spring_x + f_entropic_x
        f_czar_y = f_spring_y + f_entropic_y
        f_czar_x[~mask] = 0.0
        f_czar_y[~mask] = 0.0

        # Assemblage du système de Poisson
        n_cells = nx * ny
        rhs = np.zeros(n_cells)
        data_m, rows, cols = [], [], []

        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                
                # Liens X
                if i > 0:
                    w = (W[i, j] + W[i-1, j]) / 2.0
                    val = w / dx**2
                    data_m.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx-ny])
                    rhs[idx] += (w * (f_czar_x[i, j] + f_czar_x[i-1, j]) / 2.0) / dx
                if i < nx - 1:
                    w = (W[i, j] + W[i+1, j]) / 2.0
                    val = w / dx**2
                    data_m.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx+ny])
                    rhs[idx] -= (w * (f_czar_x[i, j] + f_czar_x[i+1, j]) / 2.0) / dx

                # Liens Y
                if j > 0:
                    w = (W[i, j] + W[i, j-1]) / 2.0
                    val = w / dy**2
                    data_m.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx-1])
                    rhs[idx] += (w * (f_czar_y[i, j] + f_czar_y[i, j-1]) / 2.0) / dy
                if j < ny - 1:
                    w = (W[i, j] + W[i, j+1]) / 2.0
                    val = w / dy**2
                    data_m.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx+1])
                    rhs[idx] -= (w * (f_czar_y[i, j] + f_czar_y[i, j+1]) / 2.0) / dy

        L = coo_matrix((data_m, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
        res, _ = linalg.cg(L, rhs, atol=tolerance)
        
        fes = res.reshape((nx, ny))
        fes = fes - np.min(fes[mask])
        fes[~mask] = np.nan
        return x_centers, y_centers, fes

    def reconstruct_fes_3d(self, tolerance=1e-6):
        """Reconstruction rigoureuse en 3D."""
        if self.d != 3:
            raise ValueError("Le fichier chargé n'est pas en 3D.")
        
        nx, ny, nz = self.shape
        dx, dy, dz = self.dx
        W = self.N_visits.astype(float)
        mask = W > 0
        W[~mask] = 1e-10
        rhs = self._get_divergence(W)

        data, rows, cols = [], [], []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = i * (ny * nz) + j * nz + k
                    # X-links
                    if i > 0:
                        w = (W[i,j,k] + W[i-1,j,k]) / 2.0
                        val = w / dx**2
                        data.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx-(ny*nz)])
                    if i < nx - 1:
                        w = (W[i,j,k] + W[i+1,j,k]) / 2.0
                        val = w / dx**2
                        data.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx+(ny*nz)])
                    # Y-links
                    if j > 0:
                        w = (W[i,j,k] + W[i,j-1,k]) / 2.0
                        val = w / dy**2
                        data.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx-nz])
                    if j < ny - 1:
                        w = (W[i,j,k] + W[i,j+1,k]) / 2.0
                        val = w / dy**2
                        data.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx+nz])
                    # Z-links
                    if k > 0:
                        w = (W[i,j,k] + W[i,j,k-1]) / 2.0
                        val = w / dz**2
                        data.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx-1])
                    if k < nz - 1:
                        w = (W[i,j,k] + W[i,j,k+1]) / 2.0
                        val = w / dz**2
                        data.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx+1])

        L = coo_matrix((data, (rows, cols)), shape=(nx*ny*nz, nx*ny*nz)).tocsr()
        res, _ = linalg.cg(L, rhs, tol=tolerance)
        fes = res.reshape(self.shape)
        fes -= np.min(fes[mask])
        fes[~mask] = np.nan
        return self.bin_centers, fes

    def reconstruct_czar_fes_3d(self, filename, temperature_K, k_springs, tolerance=1e-8):
        """
        Reconstruction du PMF CZAR en 3D via l'équation de Poisson.
        
        k_springs : tuple ou array de taille 3 (k_x, k_y, k_z)
        """
        from ase import units
        import numpy as np
        from scipy.sparse import coo_matrix, linalg

        data = np.load(filename)
        z_counts = data['z_counts'].astype(float)        # Shape: (nx, ny, nz)
        sum_lambda_z = data['sum_lambda_z'].astype(float)    # Shape: (3, nx, ny, nz)
        bin_edges_x = data['bin_edges_x']
        bin_edges_y = data['bin_edges_y']
        bin_edges_z = data['bin_edges_z']

        nx, ny, nz = z_counts.shape
        x_centers = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2.0
        y_centers = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2.0
        z_centers = (bin_edges_z[:-1] + bin_edges_z[1:]) / 2.0
        dx, dy, dz = x_centers[1]-x_centers[0], y_centers[1]-y_centers[0], z_centers[1]-z_centers[0]
        beta = 1.0 / (units.kB * temperature_K)

        mask = z_counts > 5
        W = z_counts.copy()
        W[~mask] = 1e-10

        X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

        lambda_avg_x = np.zeros_like(z_counts)
        lambda_avg_y = np.zeros_like(z_counts)
        lambda_avg_z = np.zeros_like(z_counts)
        lambda_avg_x[mask] = sum_lambda_z[0][mask] / z_counts[mask]
        lambda_avg_y[mask] = sum_lambda_z[1][mask] / z_counts[mask]
        lambda_avg_z[mask] = sum_lambda_z[2][mask] / z_counts[mask]

        f_spring_x = k_springs[0] * (X - lambda_avg_x)
        f_spring_y = k_springs[1] * (Y - lambda_avg_y)
        f_spring_z = k_springs[2] * (Z - lambda_avg_z)

        ln_rho = np.log(W)
        dln_rho_dx, dln_rho_dy, dln_rho_dz = np.gradient(ln_rho, dx, dy, dz, edge_order=1)
        
        f_entropic_x = (1.0 / beta) * dln_rho_dx
        f_entropic_y = (1.0 / beta) * dln_rho_dy
        f_entropic_z = (1.0 / beta) * dln_rho_dz

        f_czar_x = f_spring_x + f_entropic_x
        f_czar_y = f_spring_y + f_entropic_y
        f_czar_z = f_spring_z + f_entropic_z
        f_czar_x[~mask] = 0.0; f_czar_y[~mask] = 0.0; f_czar_z[~mask] = 0.0

        n_cells = nx * ny * nz
        rhs = np.zeros(n_cells)
        data_m, rows, cols = [], [], []

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = i * (ny * nz) + j * nz + k
                    
                    # Liens X (stride = ny * nz)
                    if i > 0:
                        w = (W[i,j,k] + W[i-1,j,k]) / 2.0
                        val = w / dx**2
                        data_m.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx-(ny*nz)])
                        rhs[idx] += (w * (f_czar_x[i,j,k] + f_czar_x[i-1,j,k]) / 2.0) / dx
                    if i < nx - 1:
                        w = (W[i,j,k] + W[i+1,j,k]) / 2.0
                        val = w / dx**2
                        data_m.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx+(ny*nz)])
                        rhs[idx] -= (w * (f_czar_x[i,j,k] + f_czar_x[i+1,j,k]) / 2.0) / dx

                    # Liens Y (stride = nz)
                    if j > 0:
                        w = (W[i,j,k] + W[i,j-1,k]) / 2.0
                        val = w / dy**2
                        data_m.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx-nz])
                        rhs[idx] += (w * (f_czar_y[i,j,k] + f_czar_y[i,j-1,k]) / 2.0) / dy
                    if j < ny - 1:
                        w = (W[i,j,k] + W[i,j+1,k]) / 2.0
                        val = w / dy**2
                        data_m.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx+nz])
                        rhs[idx] -= (w * (f_czar_y[i,j,k] + f_czar_y[i,j+1,k]) / 2.0) / dy

                    # Liens Z (stride = 1)
                    if k > 0:
                        w = (W[i,j,k] + W[i,j,k-1]) / 2.0
                        val = w / dz**2
                        data_m.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx-1])
                        rhs[idx] += (w * (f_czar_z[i,j,k] + f_czar_z[i,j,k-1]) / 2.0) / dz
                    if k < nz - 1:
                        w = (W[i,j,k] + W[i,j,k+1]) / 2.0
                        val = w / dz**2
                        data_m.extend([-val, val]); rows.extend([idx, idx]); cols.extend([idx, idx+1])
                        rhs[idx] -= (w * (f_czar_z[i,j,k] + f_czar_z[i,j,k+1]) / 2.0) / dz

        L = coo_matrix((data_m, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
        res, _ = linalg.cg(L, rhs, atol=tolerance)
        
        fes = res.reshape((nx, ny, nz))
        fes = fes - np.min(fes[mask])
        fes[~mask] = np.nan
        return x_centers, y_centers, z_centers, fes



