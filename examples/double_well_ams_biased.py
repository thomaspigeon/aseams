import numpy as np
import os, datetime, ase, shutil
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
from ase.io import read, write
import ase.units as units

# Import des classes personnalisées
from src.aseams.ams import AMS
from src.aseams.cvs import CollectiveVariables
from src.aseams.inicondssamplers import SingleWalkerSampler
from ase.parallel import parprint, world, barrier

# =====================================================================
# 1. PARAMÈTRES DE LA SIMULATION
# =====================================================================
# Paramètres AMS
n_ams = 10  # Nombre d'exécutions AMS par point
n_rep = 100  # Nombre de répliques
n_samples = n_ams * n_rep  # Pool de conditions initiales

# Paramètres de la Dynamique
temperature_K = 300.0
timestep = 1.0 * units.fs
friction = 0.01 / units.fs
max_length_iter = 10000

# Paramètres du Potentiel (Double Well)
a_param = 0.1
rc_param = 100.0
d1_param = 1.0
d2_param = 2.0

# Paramètres de Biais
alphas = [0.0, 0.5, 1.0]  # Pour Flux Biasing
temp_biases = [300.0, 350.0, 400.0]  # Pour Rayleigh Biasing (en Kelvin)

# Random generators
rng_ams, rng_ini, rng_dyn_ini, rng_dyn_ams, rng_bias = [np.random.default_rng(s) for s in [0, 0, 0, 0, 0]]

# =====================================================================
# 2. SETUP DU SYSTÈME ET CVs
# =====================================================================
atoms = ase.Atoms("N2", positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
calc = DoubleWell(a=a_param, rc=rc_param, d1=d1_param, d2=d2_param)
atoms.calc = calc
atoms.set_cell((8.0, 8.0, 8.0))
atoms.set_constraint(FixCom())  # Fix the COM


def distance(atoms):
    return atoms.get_distance(0, 1, mic=True)


def dist_grad(atoms):
    """
    Compute gradient of the cv with respect to atomic positions
    """
    r = atoms.get_positions()[[1], :] - atoms.get_positions()[[0], :]
    grad_r = ase.geometry.get_distances_derivatives(r, cell=atoms.cell, pbc=atoms.pbc)
    return grad_r.squeeze()


cv = CollectiveVariables(cv_r=distance,
                         cv_p=distance,
                         reaction_coordinate=distance,
                         rc_grad=dist_grad,
                         cv_r_grad=[dist_grad])
cv.set_r_crit("below")
cv.set_in_r_boundary(1.03)
cv.set_sigma_r_level(1.05)
cv.set_out_of_r_zone(1.5)
cv.set_p_crit("above")
cv.set_in_p_boundary(1.95)


# --- Génération initiale "Raw" ---
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng_dyn_ini)
dyn_ini = Langevin(atoms,
                   fixcm=True,
                   timestep=timestep,
                   temperature_K=temperature_K,
                   friction=friction,
                   rng=rng_dyn_ini)

sampler = SingleWalkerSampler(dyn_ini,
                              cv,
                              cv_interval=1,
                              fixcm=True,
                              rng=rng_ini)
sampler.set_run_dir("./ini_ams_raw", append_traj=False)
sampler.set_ini_cond_dir("./ini_ams_raw", clean=False)
parprint(f"Génération de {n_samples} conditions initiales brutes...")
sampler.sample(n_samples)

# =====================================================================
# 3. INITIALISATION DU FICHIER DE RÉSULTATS
# =====================================================================
filename = "ams_biased_comparison.txt"
if world.rank == 0:
    with open(filename, "w") as f:
        f.write("====================================================\n")
        f.write(f"COMPARAISON AMS (UNBIASED VS BIASED) - {datetime.datetime.now()}\n")
        f.write("====================================================\n\n")
        f.write("--- PARAMÈTRES DE LA DYNAMIQUE ---\n")
        f.write(f"Température physique (K)        : {temperature_K}\n")
        f.write(f"Timestep (fs)                   : {timestep / units.fs}\n")
        f.write(f"Friction (fs^-1)                : {friction * units.fs}\n")
        f.write(f"Potentiel (a, d1, d2)           : {a_param}, {d1_param}, {d2_param}\n")
        f.write(f"AMS (N_rep, K_min, M_real)      : {n_rep}, 1, {n_ams}\n")
        f.write(f"N_samples (Initial)             : {n_ams*n_rep}\n\n")
        f.write(f"{'Method':<12} | {'Param':<10} | {'P_moy':<15} | {'Std. err.':<15} | {'Rel. err.'}\n")
        f.write("-" * 60 + "\n")


# =====================================================================
# 4. BOUCLES DE CALCUL
# =====================================================================

def run_ams_batch(method_name, param_val, input_dir):
    """Lance une série de calculs AMS pour un dossier de conditions donné"""
    parprint(f"\n>>> Mode: {method_name} (Param: {param_val})")
    p_list = []

    for i in range(n_ams):
        dyn_ams = Langevin(atoms,
                           timestep=timestep,
                           temperature_K=temperature_K,
                           friction=friction,
                           logfile=None,
                           rng=rng_dyn_ams)
        ams = AMS(n_rep=n_rep,
                  k_min=1,
                  dyn=dyn_ams,
                  xi=cv,
                  rc_threshold=1e-6,
                  verbose=False,
                  rng=rng_ams)
        ams.set_ini_cond_dir(input_dir)
        ams.set_ams_dir(f"./AMS_{method_name}_{param_val}_run_{i}", clean=True)
        ams.run()
        p_list.append(ams.p_ams())

    mean_p, std_p = np.mean(p_list), np.std(p_list, ddof=1)

    if world.rank == 0:
        with open(filename, "a") as f:
            f.write(
                f"{method_name:<12} | {param_val:<10} | {mean_p:<15.5e} | {std_p / np.sqrt(n_ams):<15.5e} | {(std_p / np.sqrt(n_ams)) / mean_p:<15.5e}\n")

# =====================================================================
# 5. EXÉCUTION DES DIFFÉRENTS CAS
# =====================================================================

# --- CAS 1 : UNBIASED (Référence) ---
# On prépare un dossier où les poids sont forcés à 1.0 (sans modification de vitesses)
unbiased_dir = "./ini_ams_unbiased"
if world.rank == 0:
    if os.path.exists(unbiased_dir): shutil.rmtree(unbiased_dir)
    os.makedirs(unbiased_dir)
    for f in [fname for fname in os.listdir("./ini_ams_raw") if fname.endswith('.extxyz')]:
        at = read(os.path.join("./ini_ams_raw", f))
        at.info['weight'] = 1.0
        write(os.path.join(unbiased_dir, f), at)
barrier() # Attendre que le rang 0 finisse de copier

run_ams_batch("Unbiased_ams", "_", unbiased_dir)

# --- CAS 2 : FLUX BIASING ---
for alpha in alphas:
    out_dir = f"./ini_ams_flux_{alpha}"
    sampler.bias_initial_conditions(input_dir="./ini_ams_raw",
                                    output_dir=out_dir,
                                    method='flux',
                                    temp=temperature_K,
                                    alpha=alpha,
                                    overwrite=True,
                                    rng=rng_bias)
    run_ams_batch("Flux_ams", alpha, out_dir)

# --- CAS 3 : RAYLEIGH BIASING ---
for tb in temp_biases:
    out_dir = f"./ini_ams_rayleigh_{tb}"
    sampler.bias_initial_conditions(input_dir="./ini_ams_raw",
                                    output_dir=out_dir,
                                    method='rayleigh',
                                    temp=temperature_K,
                                    temp_bias=tb,
                                    overwrite=True,
                                    rng=rng_bias)
    run_ams_batch("Rayleigh_ams", tb, out_dir)

parprint(f"\nÉtude terminée. Voir {filename} pour les résultats.")
