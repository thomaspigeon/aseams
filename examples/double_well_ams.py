import numpy as np
import ase, datetime
from ase import Atoms
from ase.io import read, write
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units
from ase.parallel import parprint, world, barrier

from src.aseams import AMS
from src.aseams import CollectiveVariables
from src.aseams import SingleWalkerSampler
from src.aseams.utils import LangevinHalfSteps


# =====================================================================
# 1. PARAMÈTRES DE LA SIMULATION
# =====================================================================
# Paramètres AMS
n_ams = 5  # Nombre d'exécutions AMS par point
n_rep = 25  # Nombre de répliques
n_samples = n_ams * n_rep  # Pool de conditions initiales

# Paramètres de la Dynamique
temperature_K = 300.0
timestep = 1. * units.fs
friction = 0.01 / units.fs
max_length_iter = 10000

# Paramètres du Potentiel (Double Well)
a_param = 0.1
rc_param = 100.0
d1_param = 1.0
d2_param = 2.0

# Random generators
rng_ams, rng_ini, rng_dyn_ini, rng_dyn_ams = [np.random.default_rng(s) for s in [0, 0, 0, 0]]

# =====================================================================
# 2. SETUP DU SYSTÈME ET CVs
# =====================================================================
# # Initial state.
atoms = Atoms("N2", positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])  # Start from contact pair COM at 0,0,0
calc = DoubleWell(a=a_param, rc=rc_param)
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
cv.set_in_p_boundary(1.9)

# =====================================================================
# 3. INITIALISATION DU FICHIER DE RÉSULTATS
# =====================================================================

filename = "ams_normal_vs_steps_by_steps.txt"
if world.rank == 0:
    with open(filename, "w") as f:
        f.write("====================================================\n")
        f.write(f"RÉSULTATS AMS - {datetime.datetime.now()}\n")
        f.write("====================================================\n\n")
        f.write("--- PARAMÈTRES DE LA DYNAMIQUE ---\n")
        f.write(f"Température physique (K)        : {temperature_K}\n")
        f.write(f"Timestep (fs)                   : {timestep / units.fs}\n")
        f.write(f"Friction (fs^-1)                : {friction * units.fs}\n")
        f.write(f"Potentiel (a, d1, d2)           : {a_param}, {d1_param}, {d2_param}\n")
        f.write(f"AMS (N_rep, K_min, M_real)      : {n_rep}, 1, {n_ams}\n")
        f.write(f"N_samples (Initial)             : {n_ams*n_rep}\n\n")
        f.write(f"{'Method':<20} | {'P_moy':<15} | {'Std. err.':<15} | {'Rel. err.'}\n")
        f.write("-" * 60 + "\n")

# =====================================================================
# 4. GENERATION DE CONDITIONS INITIALES "NORMALES"
# =====================================================================
# --- Generate initial conditions "normal" ---
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
sampler.set_run_dir("./ini_conds_normal", append_traj=False)
sampler.set_ini_cond_dir("./ini_conds_normal", clean=False)
parprint(f"Génération de {n_samples} conditions initiales brutes...")
sampler.sample(n_samples)

"""
list_atoms = read(sampler.run_dir + "/md_traj_0.traj", index=":")
cv_traj = []
for i in range(len(list_atoms)):
    cv_traj.append(distance(list_atoms[i]))
plt.plot(cv_traj)
plt.show()
del(dyn)
"""

# =====================================================================
# 5. EVALUATION AMS "NORMAL"
# =====================================================================
p_list = []
seeds_ams = rng_ams.choice(10**6, size=n_ams)
seeds_dyn = rng_dyn_ams.choice(10**6, size=n_ams)
for i in range(n_ams):
    rng_ams = np.random.default_rng(seeds_ams[i])
    rng_dyn_ams = np.random.default_rng(seeds_dyn[i])
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
    ams.set_ini_cond_dir("./ini_conds_normal")
    ams.set_ams_dir(f"./AMS_normal_run_{i}", clean=True)
    ams.run()
    p_list.append(ams.p_ams())

mean_p, std_p = np.mean(p_list), np.std(p_list, ddof=1)
if world.rank == 0:
    with open(filename, "a") as f:
        f.write(f"{'AMS Normal':<20} | {mean_p:<15.5e} | {std_p / np.sqrt(n_ams):<15.5e} | {(std_p / np.sqrt(n_ams)) / mean_p:<15.5e}\n")

# =====================================================================
# 6. GENERATION DE CONDITIONS INITIALES "NORMALES"
# =====================================================================
# Reset Random generators and initial atoms
rng_ams, rng_ini, rng_dyn_ini, rng_dyn_ams = [np.random.default_rng(s) for s in [0, 0, 0, 0]]
atoms = Atoms("N2", positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])  # Start from contact pair COM at 0,0,0
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng_dyn_ini)
calc = DoubleWell(a=a_param, rc=rc_param)
atoms.calc = calc
atoms.set_cell((8.0, 8.0, 8.0))
atoms.set_constraint(FixCom())  # Fix the COM
# --- Generate initial conditions "steps by steps" ---
parprint(f"Génération de {n_samples} conditions initiales brutes...")

stop = False
write('current_atoms.xyz', atoms)
while not stop:
    atoms = read('current_atoms.xyz')
    atoms.set_constraint(FixCom())
    dyn = LangevinHalfSteps(atoms,
                            fixcm=True,
                            timestep=1.0 * units.fs,
                            temperature_K=temperature_K,
                            friction=friction,
                            logfile=None,
                            trajectory=None,
                            rng=rng_dyn_ini)  # temperature in K
    inicondsampler = SingleWalkerSampler(dyn, cv, rng=rng_ini, fixcm=True)
    inicondsampler.set_run_dir("ini_conds_steps_by_steps/", append_traj=True)
    inicondsampler.set_ini_cond_dir("ini_conds_steps_by_steps/")
    inicondsampler.dyn.atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces(md=True)
    stress = atoms.get_stress()
    stop = inicondsampler.sample_step_by_step(forces, energy, stress)
    stop = inicondsampler.n_ini_conds_already >= n_rep * n_ams and inicondsampler.going_to_sigma
    del(inicondsampler)
    del(dyn)
    del(atoms)


# =====================================================================
# 7. EVALUATION AMS "STEPS BY STEPS"
# =====================================================================
p_list = []
seeds_ams = rng_ams.choice(10**6, size=n_ams)
seeds_dyn = rng_dyn_ams.choice(10**6, size=n_ams)
for k in range(n_ams):
    stop = False
    rng_ams = np.random.default_rng(seeds_ams[k])
    rng_dyn_ams = np.random.default_rng(seeds_dyn[k])
    while not stop:
        atoms = read('current_atoms.xyz')
        atoms.set_constraint(FixCom())
        dyn_ams = LangevinHalfSteps(atoms,
                                    fixcm=True,
                                    timestep=1.0 * units.fs,
                                    temperature_K=temperature_K,
                                    friction=friction,
                                    logfile=None,
                                    trajectory=None,
                                    rng=rng_dyn_ams)  # temperature in K
        ams = AMS(n_rep=n_rep,
                  k_min=1,
                  dyn=dyn_ams,
                  xi=cv,
                  fixcm=True,
                  save_all=False,
                  rc_threshold=1e-6,
                  verbose=False,
                  rng=rng_ams)
        ams.set_ini_cond_dir('ini_conds_steps_by_steps')
        ams.set_ams_dir("AMS_step_by_step_" + str(k) + "/", clean=False)
        at = atoms.copy()
        at.calc = calc
        ams.dyn.atoms.calc.ignored_changes = ['positions', 'cell']
        energy = at.get_potential_energy()
        forces = at.get_forces(md=True)
        stress = at.get_stress()
        stop = ams.run_step_by_step(forces, energy, stress)
        if stop:
            p_list.append(ams.p_ams())
        del(ams)
        del(dyn_ams)
        del(atoms)

mean_p, std_p = np.mean(p_list), np.std(p_list, ddof=1)
if world.rank == 0:
    with open(filename, "a") as f:
        f.write(f"{'AMS steps by steps':<20} | {mean_p:<15.5e} | {std_p / np.sqrt(n_ams):<15.5e} | {(std_p / np.sqrt(n_ams)) / mean_p:<15.5e}\n")