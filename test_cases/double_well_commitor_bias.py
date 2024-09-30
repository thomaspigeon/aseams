import numpy as np
from ase import Atoms
from double_well_calculator import DoubleWell
from ase.constraints import FixCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import Langevin
import ase.units as units
import ase.geometry
import sys
import scipy.integrate

sys.path.insert(0, "../")

from ams import AMS
from cvs import CollectiveVariables
from inicondsamplers import InitialConditionsSampler
from ase.parallel import parprint
from ase.io import read
import matplotlib.pyplot as plt

from importance_sampling import committor_estimation, rayleigh, build_commitor_bias, committor_bias


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
print(np.arctanh(2 * comm[comm > 0] - 1))
poly = np.polynomial.Polynomial.fit(vels[comm > 0], np.arctanh(2 * comm[comm > 0] - 1), 1)
print(poly.coef)
print(poly)
plt.plot(vels, comm, "o")


bias_param = build_commitor_bias(["AMS/"], grad_distance, 300.0, committor_type="kernel", n_points_eval=750)
# # print(bias_param)
v_space = bias_param["cdf_vels"]
# plt.plot(v_space, bias_param["committor_approx"](v_space), "-")
# # vmax = 5 * np.max(vels)
# # cdf = scipy.integrate.solve_ivp(lambda v, y: np.array([rayleigh(v, units.kB * 300) * 0.5 * (1 + np.tanh(poly(v)))]), [0, vmax], y0=[0.0], t_eval=np.linspace(0.0, vmax, 750))
# #
# # # cdf = scipy.integrate.solve_ivp(lambda v, y: np.array([rayleigh(v, units.kB * 300) * 0.5 * (1 + np.tanh(poly(v)))]), [0, vmax], y0=[0.0])
# # print(cdf)
# # plt.plot(cdf.t, cdf.y[0])
# #
# #
# # plt.plot(v_space, 0.5 * (1 + np.tanh(poly(v_space))))
plt.plot(v_space, bias_param["committor_approx"](v_space))
plt.plot(v_space, rayleigh(v_space, units.kB * 300) * bias_param["committor_approx"](v_space))
# plt.plot(v_space, rayleigh(v_space, units.kB * 300))
# #
# # res = scipy.integrate.cumulative_trapezoid(rayleigh(v_space, units.kB * 300) * 0.5 * (1 + np.tanh(poly(v_space))), v_space, initial=0.0)
# # plt.plot(v_space, res)
plt.show()

print(committor_bias(300, bias_param))
