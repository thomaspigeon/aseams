import numpy as np
from ase import Atoms
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units


from aseams.ams import AMS
from aseams.cvs import CollectiveVariables
from aseams.inicondsamplers import InitialConditionsSampler
from ase.parallel import parprint
from ase.io import read
import matplotlib.pyplot as plt

# # Initial state.
atoms = Atoms("N2", positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])  # Start from contact pair COM at 0,0,0
atoms.set_constraint(FixCom())  # Fix the COM

atoms.calc = DoubleWell(a=0.1, rc=4.0)  # At 300k, a=0.05 is a nice value to observe transitions

atoms.set_cell((8.0, 8.0, 8.0))

temperature_K = 300.0


# Setup dynamics
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=temperature_K, friction=0.1 / units.fs, logfile=None, trajectory=None)  # temperature in K


def distance(atoms):
    return atoms.get_distance(0, 1, mic=True)


cv = CollectiveVariables(distance, distance, distance)

cv.set_r_crit("below")
cv.set_in_r_boundary(1.03)
cv.set_sigma_r_level(1.05)
cv.set_out_of_r_zone(1.5)
cv.set_p_crit("above")
cv.set_in_p_boundary(1.95)


inicondsampler = InitialConditionsSampler(dyn, cv)
#
inicondsampler.set_run_dir("ini_conds/")
inicondsampler.set_ini_cond_dir("ini_conds/")
inicondsampler.sample(25)


list_atoms = read(inicondsampler.run_dir + "/md_traj_0.traj", index=":")
cv_traj = []
for i in range(len(list_atoms)):
    cv_traj.append(distance(list_atoms[i]))

plt.plot(cv_traj)
plt.show()


parprint("AMS")
ams = AMS(n_rep=25, k_min=1, dyn=dyn, xi=cv, save_all="both", rc_threshold=1e-6, verbose=True)
ams.set_ini_cond_dir("ini_conds/")
ams.set_ams_dir("AMS/", clean=True)

ams.run(max_iter=1000)


print(ams.current_p)
