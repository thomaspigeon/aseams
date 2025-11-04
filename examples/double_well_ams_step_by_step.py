import numpy as np
from ase import Atoms
from ase.io import read, write
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import os
import ase.units as units

from aseams.utils import LangevinHalfSteps
from aseams import AMS
from aseams import CollectiveVariables
from aseams import SingleWalkerSampler


n_ams = 10
n_rep = 25

# # Initial state.
atoms = Atoms("N2", positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])  # Start from contact pair COM at 0,0,0

calc = DoubleWell(a=0.1, rc=100)  # At 300k, a=0.05 is a nice value to observe transitions
atoms.calc = calc

atoms.set_cell((8.0, 8.0, 8.0))

temperature_K = 300.0

# Set initial seeds
rng_ams, rng_ini, rng_dyn_ini, rng_dyn_ams = [np.random.default_rng(s) for s in [0, 0, 0, 0]]
# Setup dynamics
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng_dyn_ini)
atoms.set_constraint(FixCom())
def distance(atoms):
    return atoms.get_distance(0, 1, mic=True)

cv = CollectiveVariables(distance, distance, distance)

cv.set_r_crit("below")
cv.set_in_r_boundary(1.03)
cv.set_sigma_r_level(1.05)
cv.set_out_of_r_zone(1.5)
cv.set_p_crit("above")
cv.set_in_p_boundary(1.95)

stop = False
write('current_atoms.xyz', atoms)
while not stop:
    atoms = read('current_atoms.xyz')
    atoms.set_constraint(FixCom())
    dyn = LangevinHalfSteps(atoms,
                            fixcm=True,
                            timestep=1.0 * units.fs,
                            temperature_K=temperature_K,
                            friction=0.001 / units.fs,
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
"""
list_atoms = read("ini_conds_steps_by_steps/" + "md_traj_0.traj", index=":")
cv_traj = []
for i in range(len(list_atoms)):
    cv_traj.append(distance(list_atoms[i]))

plt.plot(cv_traj)
plt.show()
"""
probas = []
seeds_ams = rng_ams.choice(10**6, size=10)
seeds_dyn = rng_dyn_ams.choice(10**6, size=10)
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
                                    friction=0.001 / units.fs,
                                    logfile=None,
                                    trajectory=None,
                                    rng=rng_dyn_ams)  # temperature in K
        ams = AMS(n_rep=25,
                  k_min=1,
                  dyn=dyn_ams,
                  xi=cv,
                  fixcm=True,
                  save_all=True,
                  rc_threshold=1e-6,
                  verbose=False,
                  rng=rng_ams)
        ams.set_ini_cond_dir('ini_conds_steps_by_steps')
        ams.set_ams_dir("AMS_step_by_step_" + str(k) +"/", clean=False)
        at = atoms.copy()
        at.calc = calc
        ams.dyn.atoms.calc.ignored_changes = ['positions', 'cell']
        energy = at.get_potential_energy()
        forces = at.get_forces(md=True)
        stress = at.get_stress()
        stop = ams.run_step_by_step(forces, energy, stress)
        if stop:
            probas.append(ams.p_ams())
        del(ams)
        del(dyn_ams)
        del(atoms)

os.system('rm rnd_vel.txt current_atoms.xyz')
res = np.array([np.mean(probas), np.std(probas, ddof=1) / np.sqrt(len(probas))])
np.savetxt('results_steps_by_steps.txt', res)
