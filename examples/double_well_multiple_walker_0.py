import numpy as np
from ase import Atoms
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units


from src.aseams import CollectiveVariables
from src.aseams import MultiWalkerSampler

# # Initial state.
atoms = Atoms("N2", positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])  # Start from contact pair COM at 0,0,0

calc = DoubleWell(a=0.1, rc=100.0)  # At 300k, a=0.05 is a nice value to observe transitions
atoms.calc = calc

atoms.set_cell((8.0, 8.0, 8.0))

temperature_K = 300.0

# Set initial seeds
rng_ini, rng_dyn_ini = [np.random.default_rng(s) for s in [0, 0]]
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

inicondsampler = MultiWalkerSampler(dyn, cv, n_walkers=2, walker_index=0, rng=rng_ini, fixcm=True)

inicondsampler.set_run_dir("ini_conds_walker_", append_traj=False)
inicondsampler.set_ini_cond_dir("ini_conds_multi_walkers")
inicondsampler.sample(100)


"""This example will crash after doing 5026 as in this case, both walker will be out of the reactant bassin at this time
In a reallistic scenario, the solution to such a problem is to run more walkers in parallel."""
