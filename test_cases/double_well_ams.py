import numpy as np
from ase import Atoms
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units
import sys

sys.path.insert(0, "../")

from ams import AMS
from cvs import CollectiveVariables
from inicondsamplers import InitialConditionsSampler
from ase.parallel import parprint
from ase.io import read, write
import matplotlib.pyplot as plt

# # Initial state.
atoms = Atoms("N2", positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])  # Start from contact pair COM at 0,0,0
atoms.set_constraint(FixCom())  # Fix the COM

atoms.calc = DoubleWell(a=0.05, rc=4.0)  # At 300k, a=0.05 is a nice value to observe transitions

atoms.set_cell((8.0, 8.0, 8.0))

temperature_K = 300.0

rng_ams, rng_dyn = [np.random.default_rng(s) for s in [0, 0]]
# Setup dynamics
#MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng_dyn)
write('current_atoms.xyz', atoms)
dyn = Langevin(atoms,
               fixcm=True,
               timestep=1.0 * units.fs,
               temperature_K=temperature_K,
               friction=0.1 / units.fs,
               logfile=None,
               trajectory=None,
               rng=rng_dyn)  # temperature in K

def distance(atoms):
    return atoms.get_distance(0, 1, mic=True)


cv = CollectiveVariables(distance, distance, distance)

cv.set_r_crit("below")
cv.set_in_r_boundary(1.03)
cv.set_sigma_r_level(1.05)
cv.set_out_of_r_zone(1.5)
cv.set_p_crit("above")
cv.set_in_p_boundary(1.95)


inicondsampler = InitialConditionsSampler(dyn, cv, rng=rng_ams)

inicondsampler.set_run_dir("ini_conds_normal/")
inicondsampler.set_ini_cond_dir("ini_conds_normal/")
inicondsampler.sample(20)


list_atoms = read(inicondsampler.run_dir + "/md_traj_0.traj", index=":")
cv_traj = []
for i in range(len(list_atoms)):
    cv_traj.append(distance(list_atoms[i]))
print('ini_conds done')
plt.plot(cv_traj)
plt.show()

"""
probas = []
for i in range(1):
    ams = AMS(n_rep=5, k_min=1, dyn=dyn, xi=cv, fixcm=True, save_all=True, rc_threshold=1e-6, verbose=False, rng=rng_ams)
    ams.set_ini_cond_dir('ini_cds')
    ams.set_ams_dir("AMS_" + str(i) + "/", clean=True)
    ams._initialize()
    ams.run(max_iter=1000)
    probas.append(ams.current_p)

np.savetxt('probas_ams.txt', np.array(probas))
"""