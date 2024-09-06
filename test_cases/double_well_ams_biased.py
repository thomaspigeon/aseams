import numpy as np
from ase import Atoms
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units
import sys
import ase.geometry

sys.path.insert(0, "../")

from ams import AMS
from cvs import CollectiveVariables
from inicondsamplers import InitialConditionsSampler

from importance_sampling import rayleigh_bias_init_cond_velocity
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
dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=temperature_K, friction=0.05 / units.fs, logfile=None, trajectory=None)  # temperature in K


def distance(atoms):
    return atoms.get_distance(0, 1, mic=True)


def grad_distance(atoms):
    """
    Compute gradient of the cv with respect to atomic positions
    """
    r = atoms.get_positions()[[1], :] - atoms.get_positions()[[0], :]
    grad_r = ase.geometry.get_distances_derivatives(r, cell=atoms.cell, pbc=atoms.pbc)
    indices = [0,1]
    return grad_r.squeeze(), (np.repeat(indices, 3), np.tile([0, 1, 2], len(indices)))


cv = CollectiveVariables(distance, distance, distance)

cv.set_r_crit("below")
cv.set_in_r_boundary(1.03)
cv.set_sigma_r_level(1.05)
cv.set_out_of_r_zone(1.5)
cv.set_p_crit("above")
cv.set_in_p_boundary(1.95)


# inicondsampler = InitialConditionsSampler(dyn, cv)
# #
# inicondsampler.set_run_dir("ini_conds/")
# inicondsampler.set_ini_cond_dir("ini_conds/")
# inicondsampler.sample(25)


# Bias initial conditions
bias_temps = np.linspace(300, 500, 10)
mean_var_biased = np.zeros((len(bias_temps), 3))

for m, temp_bias in enumerate(bias_temps):
    res_ams = np.zeros(25)
    for n in range(len(res_ams)):
        parprint(f"AMS {n}")
        rayleigh_bias_init_cond_velocity(grad_distance, "ini_conds_for_bias/", "ini_conds_biased/", temperature_K, temp_bias)
        ams = AMS(n_rep=25, k_min=1, dyn=dyn, xi=cv, save_all=True, rc_threshold=1e-6, verbose=False)
        ams.set_ini_cond_dir("ini_conds_biased/")
        ams.set_ams_dir(f"AMS_bias_{n}/", clean=True)

        ams.run(max_iter=1000)
        res_ams[n] = ams.p_ams()
    mean_var_biased[m, 0] = temp_bias
    mean_var_biased[m, 1] = np.mean(res_ams)
    mean_var_biased[m, 2] = np.var(res_ams)
    np.savetxt("biasing_results.dat", mean_var_biased)
