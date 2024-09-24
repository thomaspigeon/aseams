import numpy as np
import glob
import sys
from ase import units
from ase.constraints import FixCom

from ase.io import read
import ase.geometry

sys.path.insert(0, "../")
from importance_sampling import rayleigh_bias_init_cond_velocity, set_velocities

import matplotlib.pyplot as plt


def get_velocity(atoms, rc_grad):
    grad, inclued_coords = rc_grad(atoms)
    velocities = atoms.get_momenta() / atoms.get_masses()[:, np.newaxis]
    return np.dot(grad.ravel(), velocities.ravel()[np.ravel_multi_index(inclued_coords, velocities.shape)])  #


def velocity_values(ini_conds_folder, rc_grad):
    vels = []
    # Once obtained data, we can bias the initial velocity
    for file in glob.glob(ini_conds_folder + "*.extxyz"):
        atoms = read(file)
        vels.append(get_velocity(atoms, rc_grad))
        # velocities = atoms.get_momenta() / atoms.get_masses()[:, np.newaxis]
        # vels.append(velocities[])

    return vels


def grad_distance(atoms):
    """
    Compute gradient of the cv with respect to atomic positions
    """
    r = atoms.get_positions()[[1], :] - atoms.get_positions()[[0], :]
    grad_r = ase.geometry.get_distances_derivatives(r, cell=atoms.cell, pbc=atoms.pbc)
    indices = [0, 1]
    return grad_r.squeeze(), (np.repeat(indices, 3), np.tile([0, 1, 2], len(indices)))


def rayleight(x, sig2):
    return x / sig2 * np.exp(-0.5 * x**2 / sig2)


vels_ini = velocity_values("ini_conds/", grad_distance)

rayleigh_bias_init_cond_velocity(grad_distance, "ini_conds/", "ini_conds_biased_dist/", 300, 300, constraints=[FixCom()])


vels_biased = velocity_values("ini_conds_biased_dist/", grad_distance)


plt.hist(vels_ini, bins=20, density=True)
plt.hist(vels_biased, bins=20, alpha=0.5, density=True)

sig_ini = 0.5 * np.mean(np.power(vels_ini, 2))
sig_biased = 0.5 * np.mean(np.power(vels_biased, 2))
print(((300 * units.kB) / sig_ini), (300 * units.kB) / sig_biased, sig_biased / sig_ini)

v_range = np.linspace(0, np.max(vels_biased), 250)
plt.plot(v_range, rayleight(v_range, sig_ini))
plt.plot(v_range, rayleight(v_range, sig_biased))

plt.show()


# def get_normal_components(rc_grad, atoms, bias_temp):
#     """
#     Set velocities of atoms, given the normal component and orthogonal componants at a given temperatures
#     If scale_normal_component is True multiply by effective masse
#     """
#     new_momenta = np.zeros_like(atoms.get_momenta())
#     normal = np.zeros_like(atoms.get_momenta()).ravel()

#     masses = atoms.get_masses()[:, np.newaxis]

#     grad, inclued_coords = rc_grad(atoms)
#     normal[np.ravel_multi_index(inclued_coords, new_momenta.shape)] = grad.ravel()
#     norm = np.sqrt(np.dot(normal, (normal.reshape(-1, 3) / masses).ravel()))
#     normal /= norm  # Normalize the normal

#     normal_components = np.dot(normal, (atoms.get_momenta() / masses).ravel())
#     return normal_components / np.sqrt(units.kB * bias_temp)


# def compare_normal_components(rc_grad, ini_conds, temp, bias_temp, constraints=[]):
#     rng = np.random.default_rng()
#     # Once obtained data, we can bias the initial velocity
#     for file in glob.glob(ini_conds + "*.extxyz"):
#         atoms = read(file)
#         for cons in constraints:  # If there is constraints
#             atoms.set_constraint(cons)
#         new_normal_components = rng.rayleigh(scale=1.0)
#         eff_mass = set_velocities(rc_grad, atoms, new_normal_components * np.sqrt(units.kB * bias_temp), float(temp), scale_normal_component=False)
#         print(new_normal_components - get_normal_components(rc_grad, atoms, bias_temp))


# compare_normal_components(grad_distance, "ini_conds/", 300, 300, constraints=[FixCom()])
