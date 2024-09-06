import numpy as np
import glob
import os
from ase import units
from ase.io import read, write
from ase.parallel import world


def set_velocities(rc_grad, atoms, normal_component, temp_orthogonal, scale_normal_component=True):
    """
    Set velocities of atoms, given the normal component and orthogonal componants at a given temperatures
    If scale_normal_component is True multiply by effective masse
    """
    new_momenta = np.zeros_like(atoms.get_momenta())
    normal = np.zeros_like(atoms.get_momenta()).ravel()

    masses = atoms.get_masses()[:, np.newaxis]

    grad, inclued_coords = rc_grad(atoms)
    normal[np.ravel_multi_index(inclued_coords, new_momenta.shape)] = grad.ravel()
    norm = np.sqrt(np.dot(normal, (normal.reshape(-1, 3) / masses).ravel()))
    normal /= norm  # Normalize the normal
    if scale_normal_component:
        normal_component /= norm
    new_momenta += normal_component * normal.reshape(-1, 3)
    # Generate orthogonal components
    m_kT = masses * units.kB * temp_orthogonal
    ortho_momenta = np.random.normal(size=(len(m_kT), 3)) * np.sqrt(m_kT)

    new_momenta += ortho_momenta - normal.reshape(-1, 3) * np.dot(normal, (ortho_momenta / masses).ravel())
    atoms.set_momenta(new_momenta)
    return 1.0 / norm


def rayleigh_ratio(x, temp_new, temp):
    """
    Ratio of 2 Rayleigh PDF for importance sampling
    """
    return temp_new / temp * np.exp(-0.5 * (x**2) * (temp_new / temp - 1.0))


def rayleigh_bias_init_cond_velocity(rc_grad, ini_conds, ini_conds_biased, temp, bias_temp, constraints=[]):

    if not os.path.exists(ini_conds_biased) and world.rank == 0:
        os.makedirs(ini_conds_biased, exist_ok=True)

    rng = np.random.default_rng()
    # Once obtained data, we can bias the initial velocity
    for file in glob.glob(ini_conds + "*.extxyz"):
        atoms = read(file)
        for cons in constraints:  # If there is constraints
            atoms.set_constraint(cons)
        new_normal_components = rng.rayleigh(scale=1.0)
        eff_mass = set_velocities(rc_grad, atoms, new_normal_components * np.sqrt(units.kB * bias_temp), float(temp), scale_normal_component=False)
        atoms.info["weight"] = rayleigh_ratio(new_normal_components, bias_temp, float(temp))  # Ici poids pour la fonction d'importance
        write(ini_conds_biased + os.path.basename(file), atoms)
