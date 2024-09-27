import numpy as np
from ase import Atoms
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units
import ase.geometry
import sys

sys.path.insert(0, "../")

from ams import AMS
from cvs import CollectiveVariables
from inicondsamplers import InitialConditionsSampler
from ase.parallel import parprint
from ase.io import read
import matplotlib.pyplot as plt

from importance_sampling import committor_estimation


def distance(atoms):
    return atoms.get_distance(0, 1, mic=True)


def grad_distance(atoms):
    """
    Compute gradient of the cv with respect to atomic positions
    """
    r = atoms.get_positions()[[1], :] - atoms.get_positions()[[0], :]
    grad_r = ase.geometry.get_distances_derivatives(r, cell=atoms.cell, pbc=atoms.pbc)
    indices = [0, 1]
    return grad_r.squeeze(), (np.repeat(indices, 3), np.tile([0, 1, 2], len(indices)))


vels, comm = committor_estimation(["AMS/"], grad_distance)
print(vels.shape, comm.shape)
plt.plot(vels, comm, "o")
plt.show()
