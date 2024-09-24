import numpy as np
from ase import Atoms
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units

from ase.io import Trajectory
import matplotlib.pyplot as plt

# # Initial state.
atoms = Atoms("N2", positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])  # Start from contact pair COM at 0,0,0
atoms.set_constraint(FixCom())  # Fix the COM

atoms.calc = DoubleWell(a=0.05, rc=4.0)  # At 300k, a=0.05 is a nice value to observe transitions

atoms.set_cell((8.0, 8.0, 8.0))

temperature_K = 300.0

# Checking calculators energy and forces
new_pos = atoms.positions
x_pos = np.linspace(0.75, 4.0, 500)
e = np.zeros(len(x_pos))
f = np.zeros(len(x_pos))

for n, x in enumerate(x_pos):
    new_pos[1, 0] = x
    atoms.set_positions(new_pos)
    e[n] = atoms.get_potential_energy()
    f[n] = atoms.get_forces()[1, 0]
plt.plot(x_pos, e)
plt.plot(x_pos, f)
plt.grid()


# Run MD
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=temperature_K, friction=0.01 / units.fs, logfile=None, trajectory=None)  # temperature in K

traj_obj = Trajectory("pair.traj", "w", atoms)
dyn.attach(traj_obj.write, interval=50)
dyn.run(50000)
plt.show()
