import numpy as np
from ase import Atoms
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units
import ase.geometry

from aseams import AMS
from aseams import CollectiveVariables
from aseams import SingleWalkerSampler

from ase.parallel import parprint

n_ams = 2
n_rep = 25
n_bias = 6

# # Initial state.
atoms = Atoms("N2", positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])  # Start from contact pair COM at 0,0,0

calc = DoubleWell(a=0.1, rc=100.0)  # At 300k, a=0.05 is a nice value to observe transitions
atoms.calc = calc

atoms.set_cell((8.0, 8.0, 8.0))

temperature_K = 100.0

# Set initial seeds
rng_ams, rng_ini, rng_dyn_ini, rng_dyn_ams, rng_bias = [np.random.default_rng(s) for s in [0, 0, 0, 0, 0]]
# Setup dynamics
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng_dyn_ini)
atoms.set_constraint(FixCom())  # Fix the COM
dyn = Langevin(atoms,
               fixcm=True,
               timestep=1.0 * units.fs,
               temperature_K=temperature_K,
               friction=0.001 / units.fs,
               logfile=None,
               trajectory=None,
               rng=rng_dyn_ini)   # temperature in K


def distance(atoms):
    return atoms.get_distance(0, 1, mic=True)


def grad_distance(atoms):
    """
    Compute gradient of the cv with respect to atomic positions
    """
    r = atoms.get_positions()[[1], :] - atoms.get_positions()[[0], :]
    grad_r = ase.geometry.get_distances_derivatives(r, cell=atoms.cell, pbc=atoms.pbc)
    indices = [0, 1]
    return grad_r.squeeze()

cv = CollectiveVariables(distance, distance, distance, rc_grad=grad_distance)

cv.set_r_crit("below")
cv.set_in_r_boundary(1.03)
cv.set_sigma_r_level(1.05)
cv.set_out_of_r_zone(1.5)
cv.set_p_crit("above")
cv.set_in_p_boundary(1.95)


inicondsampler = SingleWalkerSampler(dyn, cv, rng=rng_ini, fixcm=True)

inicondsampler.set_run_dir("ini_conds_no_bias/")
inicondsampler.set_ini_cond_dir("ini_conds_no_bias/")
inicondsampler.sample(n_rep * n_ams)
dyn.observers.pop(-1)
"""
list_atoms = read(inicondsampler.run_dir + "/md_traj_0.traj", index=":")
cv_traj = []
for i in range(len(list_atoms)):
    cv_traj.append(distance(list_atoms[i]))

plt.plot(cv_traj)
plt.show()
"""
del(dyn)


# Bias initial conditions
bias_temps = np.linspace(temperature_K, 5 * temperature_K, n_bias)
mean_var_biased = np.zeros((len(bias_temps), 3))

seeds_ams = rng_ams.choice(10**6, size=n_ams)
seeds_dyn = rng_dyn_ams.choice(10**6, size=n_ams)
seeds_bias = rng_bias.choice(10**6, size=n_bias)
for m, temp_bias in enumerate(bias_temps):
    res_ams = np.zeros(n_ams)
    rng_bias = np.random.default_rng(seeds_bias[m])
    _ = inicondsampler.bias_initial_conditions(input_dir="ini_conds_no_bias/",
                                               output_dir=f"ini_conds_biased_{m}/",
                                               temp=temperature_K,
                                               bias_temp=temp_bias,
                                               resample_ortho=False,
                                               rng=rng_bias)
    for n in range(len(res_ams)):
        parprint(f"AMS {n}")
        rng_ams = np.random.default_rng(seeds_ams[n])
        rng_dyn_ams = np.random.default_rng(seeds_dyn[n])
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
        ams.set_ini_cond_dir(f"ini_conds_biased_{m}/")
        ams.set_ams_dir(f"AMS_bias_{n}/", clean=True)

        ams.run()
        res_ams[n] = ams.p_ams()
    mean_var_biased[m, 0] = temp_bias
    mean_var_biased[m, 1] = np.mean(res_ams)
    mean_var_biased[m, 2] = np.std(res_ams, ddof=1) / np.sqrt(len(res_ams))
    np.savetxt("biasing_results.dat", mean_var_biased)