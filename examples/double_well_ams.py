import numpy as np
from ase import Atoms
import datetime
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units
from ase.parallel import parprint, world

from src.aseams import AMS
from src.aseams import CollectiveVariables
from src.aseams import SingleWalkerSampler


# =====================================================================
# 1. PARAMÈTRES DE LA SIMULATION
# =====================================================================
# Paramètres AMS
n_ams = 10  # Nombre d'exécutions AMS par point
n_rep = 25  # Nombre de répliques
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


# Random generators
rng_ams, rng_ini, rng_dyn_ini, rng_dyn_ams = [np.random.default_rng(s) for s in [0, 0, 0, 0]]
n_ams = 10
n_rep = 25

# =====================================================================
# 2. SETUP DU SYSTÈME ET CVs
# =====================================================================
# # Initial state.
atoms = Atoms("N2", positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])  # Start from contact pair COM at 0,0,0
calc = DoubleWell(a=a_param, rc=rc_param)
atoms.calc = calc
atoms.set_cell((8.0, 8.0, 8.0))
# Setup dynamics
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng_dyn_ini, force_temp=False)
atoms.set_constraint(FixCom())  # Fix the COM
dyn = Langevin(atoms,
               fixcm=True,
               timestep=1.0 * units.fs,
               temperature_K=temperature_K,
               friction=0.001 / units.fs,
               logfile=None,
               trajectory=None,
               rng=rng_dyn_ini)  # temperature in K

def distance(atoms):
    return atoms.get_distance(0, 1, mic=False)

cv = CollectiveVariables(distance, distance, distance)

cv.set_r_crit("below")
cv.set_in_r_boundary(1.03)
cv.set_sigma_r_level(1.05)
cv.set_out_of_r_zone(1.5)
cv.set_p_crit("above")
cv.set_in_p_boundary(1.95)

inicondsampler = SingleWalkerSampler(dyn, cv, rng=rng_ini, fixcm=True)
inicondsampler.set_run_dir("ini_conds_normal/", append_traj=False)
inicondsampler.set_ini_cond_dir("ini_conds_normal/")
inicondsampler.sample(n_rep * n_ams)
dyn.observers.pop(-1)
"""
list_atoms = read(inicondsampler.run_dir + "/md_traj_0.traj", index=":")
cv_traj = []
for i in range(len(list_atoms)):
    cv_traj.append(distance(list_atoms[i]))

plt.plot(cv_traj)
plt.show()
del(dyn)
"""

# =====================================================================
# 3. INITIALISATION DU FICHIER DE RÉSULTATS
# =====================================================================
filename = "ams_normal.txt"
if world.rank == 0:
    with open(filename, "w") as f:
        f.write("====================================================\n")
        f.write(f"RÉSULTATS AMS - {datetime.datetime.now()}\n")
        f.write("====================================================\n\n")
        f.write("--- PARAMÈTRES DE LA DYNAMIQUE ---\n")
        f.write(f"Température physique (K) : {temperature_K}\n")
        f.write(f"Timestep (fs)           : {timestep / units.fs}\n")
        f.write(f"Friction (fs^-1)        : {friction * units.fs}\n")
        f.write(f"Potentiel (a, d1, d2)   : {a_param}, {d1_param}, {d2_param}\n")
        f.write(f"AMS (Replicas, K_min)   : {n_rep}, 1\n")
        f.write(f"N_samples (Initial)     : {n_ams*n_rep}\n\n")
        f.write(f"{'Méthode':<12} | {'Param':<10} | {'P_moy':<15} | {'Ecart-type':<15}\n")
        f.write("-" * 60 + "\n")


# =====================================================================
# 4. BOUCLES DE CALCUL
# =====================================================================


probas = []
seeds_ams = rng_ams.choice(10**6, size=10)
seeds_dyn = rng_dyn_ams.choice(10**6, size=10)
for k in range(n_ams):
    rng_ams = np.random.default_rng(seeds_ams[k])
    rng_dyn_ams = np.random.default_rng(seeds_dyn[k])
    dyn_ams = Langevin(atoms,
                       fixcm=True,
                       timestep=1.0 * units.fs,
                       temperature_K=temperature_K,
                       friction=0.001 / units.fs,
                       logfile=None,
                       trajectory=None,
                       rng=rng_dyn_ams)
    ams = AMS(n_rep=n_rep,
              k_min=1,
              dyn=dyn_ams,
              xi=cv,
              fixcm=True,
              save_all=True,
              rc_threshold=1e-6,
              verbose=False,
              rng=rng_ams)
    ams.set_ini_cond_dir('ini_conds_normal/')
    ams.set_ams_dir("AMS_normal_" + str(k) + "/", clean=True)
    ams._initialize()
    ams.run()
    probas.append(ams.p_ams())

res = np.array([np.mean(probas), np.std(probas, ddof=1) / np.sqrt(len(probas))])
np.savetxt('results_normal.txt', res)
