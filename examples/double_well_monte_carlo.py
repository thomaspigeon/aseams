import numpy as np
import os, ase, datetime, math, shutil
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.units as units
from ase.io import read, write
from ase.constraints import FixCom
from ase.parallel import parprint, world, barrier


# Import your custom classes
from double_well_calculator import DoubleWell
from src.aseams.cvs import CollectiveVariables
from src.aseams.inicondssamplers import SingleWalkerSampler


def run_direct_mc_batch(input_dir, cv, temp, friction, timestep, max_steps=10000, calc=DoubleWell(a=0.1, rc=100.0)):
    """
    Runs Direct Monte Carlo for all files in a directory to estimate
    the probability of reaching P before R.
    """
    files = [f for f in os.listdir(input_dir) if f.endswith('.extxyz')]
    n_total = len(files)
    successes = []
    weights = []

    for fname in files:
        atoms = read(os.path.join(input_dir, fname))
        atoms.set_constraint(FixCom())  # Fix the COM
        atoms.calc = calc
        weight = atoms.info.get('weight', 1.0)

        # Setup dynamics for this specific sample
        dyn = Langevin(atoms,
                       fixcm=True,
                       timestep=timestep,
                       temperature_K=temp,
                       friction=friction,
                       logfile=None,
                       rng=rng_dyn_mc)
        step = 0
        reached_p = False
        while step < max_steps:
            dyn.run(1)  # Run 1 step (or use cv_interval)

            # Check stopping conditions
            if cv.in_r(atoms):
                reached_p = False
                break
            if cv.in_p(atoms):
                reached_p = True
                break
            step += 1

        successes.append(1.0 if reached_p else 0.0)
        weights.append(weight)

    # Importance Sampling Estimator: P = (1/N) * sum(W_i * I_i)
    successes = np.array(successes)
    weights = np.array(weights)
    weighted_outcomes = weights * successes

    p_est = np.mean(weighted_outcomes)

    # Variance of the estimator: Var(P_est) = Var(W * I) / N
    # Note: For unbiased MC, W=1, so it simplifies to p(1-p)/N
    var_weighted = np.var(weighted_outcomes, ddof=1)
    p_var = var_weighted / n_total

    return p_est, p_var, n_total


# =====================================================================
# Main Simulation Script
# =====================================================================


# =====================================================================
# 1. PARAMÈTRES DE LA SIMULATION
# =====================================================================
# Paramètres Monte Carlo
n_samples = 1000  # Pool de conditions initiales

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
rng_ini, rng_dyn_ini, rng_bias, rng_dyn_mc = [np.random.default_rng(s) for s in [0, 0, 0, 0]]

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

# =====================================================================
# 3. INITIALISATION DU FICHIER DE RÉSULTATS
# =====================================================================
filename = "mc_biased_comparison.txt"
if world.rank == 0:
    with open(filename, "w") as f:
        f.write("====================================================\n")
        f.write(f"COMPARAISON MC (UNBIASED VS BIASED) - {datetime.datetime.now()}\n")
        f.write("====================================================\n\n")
        f.write("--- PARAMÈTRES DE LA DYNAMIQUE ---\n")
        f.write(f"Température physique (K) : {temperature_K}\n")
        f.write(f"Timestep (fs)           : {timestep / units.fs}\n")
        f.write(f"Friction (fs^-1)        : {friction * units.fs}\n")
        f.write(f"Potentiel (a, d1, d2)   : {a_param}, {d1_param}, {d2_param}\n")
        f.write(f"N_samples (Initial)     : {n_samples}\n\n")
        f.write(f"{'Method':<12} | {'Param':<10} | {'P_moy':<15} | {'Std. err.':<15} | {'Rel. err.'}\n")
        f.write("-" * 60 + "\n")


# --- Génération initial "Raw" ---
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

sampler.set_run_dir("./ini_mc_raw")
sampler.set_ini_cond_dir("./ini_mc_raw", clean=False)
# Run a short MD to collect crossings of the boundary
sampler.sample(n_samples)

# =====================================================================
# 4. EXÉCUTION DES DIFFÉRENTS CAS
# =====================================================================

# --- CAS 1 : UNBIASED (Référence) ---
# On prépare un dossier où les poids sont forcés à 1.0 (sans modification de vitesses)
unbiased_dir = "./ini_mc_unbiased"
if world.rank == 0:
    if os.path.exists(unbiased_dir): shutil.rmtree(unbiased_dir)
    os.makedirs(unbiased_dir)
    for f in [fname for fname in os.listdir("./ini_mc_raw") if fname.endswith('.extxyz')]:
        at = read(os.path.join("./ini_mc_raw", f))
        at.info['weight'] = 1.0
        write(os.path.join(unbiased_dir, f), at)
barrier() # Attendre que le rang 0 finisse de copier

p_est, p_var, _ = run_direct_mc_batch(unbiased_dir, cv, temperature_K, friction, timestep,
                                      max_steps=10000, calc=DoubleWell(a=0.1, rc=100.0))
if world.rank == 0:
    with open(filename, "a") as f:
        f.write(f"{'Unbiased':<12} | {'_':<10} | {p_est:<15.5e} | {math.sqrt(p_var):<15.5e} | {math.sqrt(p_var)/p_est:<15.5e}\n")

# --- CAS 2 : FLUX BIASING ---
for alpha in alphas:
    out_dir = f"./ini_mc_flux_{alpha}"
    sampler.bias_initial_conditions(input_dir="./ini_mc_raw",
                                    output_dir=out_dir,
                                    method='flux',
                                    temp=temperature_K,
                                    alpha=alpha,
                                    overwrite=True,
                                    rng=rng_bias)
    p_est, p_var, _ = run_direct_mc_batch(unbiased_dir, cv, temperature_K, friction, timestep,
                                          max_steps=10000, calc=DoubleWell(a=0.1, rc=100.0))
    if world.rank == 0:
        with open(filename, "a") as f:
            f.write(f"{'Flux':<12} | {alpha:<10} | {p_est:<15.5e} | {math.sqrt(p_var):<15.5e} | {math.sqrt(p_var)/p_est:<15.5e}\n")

# --- CAS 3 : RAYLEIGH BIASING ---
for tb in temp_biases:
    out_dir = f"./ini_mc_rayleigh_{tb}"
    sampler.bias_initial_conditions(input_dir="./ini_mc_raw",
                                    output_dir=out_dir,
                                    method='rayleigh',
                                    temp=temperature_K,
                                    temp_bias=tb,
                                    overwrite=True,
                                    rng=rng_bias)
    p_est, p_var, _ = run_direct_mc_batch(unbiased_dir, cv, temperature_K, friction, timestep,
                                          max_steps=10000, calc=DoubleWell(a=0.1, rc=100.0))
    if world.rank == 0:
        with open(filename, "a") as f:
            f.write(f"{'Rayleigh':<12} | {tb:<10} | {p_est:<15.5e} | {math.sqrt(p_var):<15.5e} | {math.sqrt(p_var)/p_est:<15.5e}\n")

parprint(f"\nÉtude terminée. Voir {filename} pour les résultats.")
