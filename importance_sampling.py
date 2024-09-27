import numpy as np
import glob
import os
from ase import units
from ase.io import read, write
from ase.parallel import world
import json
from statsmodels.nonparametric.kernel_regression import KernelReg


def rc_vel(atoms, rc_grad):
    grad, inclued_coords = rc_grad(atoms)
    velocities = atoms.get_momenta() / atoms.get_masses()[:, np.newaxis]
    return np.dot(grad.ravel(), velocities.ravel()[np.ravel_multi_index(inclued_coords, velocities.shape)])


def bias_init_cond_velocity(rc_grad, ini_conds, ini_conds_biased, temp, bias_params, constraints=[]):
    """
    Return a set of new initial conditions with biased velocity
    """

    if not os.path.exists(ini_conds_biased) and world.rank == 0:
        os.makedirs(ini_conds_biased, exist_ok=True)

    rng = np.random.default_rng()
    # Once obtained data, we can bias the initial velocity
    for file in glob.glob(ini_conds + "*.extxyz"):
        atoms = read(file)
        for cons in constraints:  # If there is constraints
            atoms.set_constraint(cons)
        new_velocity, weight, scale_normal_component = rayleigh_bias(temp, bias_params, rng)
        _ = set_velocities(rc_grad, atoms, new_velocity, temp, scale_normal_component=scale_normal_component)
        atoms.info["weight"] = weight  # Ici poids pour la fonction d'importance
        write(ini_conds_biased + os.path.basename(file), atoms)


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


def rayleigh_bias(temp, bias_param, rng):
    """
    Get a new velocity and
    """
    bias_temp = bias_param["bias_temp"]
    x = rng.rayleigh(scale=1.0)
    weight = bias_temp / temp * np.exp(-0.5 * (x**2) * (bias_temp / temp - 1.0))  # Ratio of 2 Rayleigh PDF for importance sampling
    return x * np.sqrt(units.kB * bias_temp), weight, False


def committor_estimation(ams_runs_paths, rc_grad):
    """
    Estimate committor from previous AMS runs
    Get a list of folder that contains
    """

    # Load data

    for data_ams in ams_runs_paths:
        json_file = open(data_ams + "ams_checkpoint.txt", "r")
        checkpoint_data = json.load(json_file)
        json_file.close()
        z_maxs = checkpoint_data["z_maxs"]
        vels = []
        weights = []
        is_reactive = []
        w_r = [w[-1] for w in checkpoint_data["rep_weights"]]
        n_reactives_files = len(z_maxs)
        for n in range(n_reactives_files):
            vel = rc_vel(read(data_ams + f"rep_{n}.traj", index="0"), rc_grad)
            vels.append(vel)
            weights.append(w_r[n])
            if z_maxs[n] >= np.infty:
                is_reactive.append(1)
            else:
                is_reactive.append(0)

        # Then load non reaction trajectories
        nr_files = glob.glob(data_ams + "nr_rep_*.traj")
        for file in nr_files:
            vel = rc_vel(read(file, index="0"), rc_grad)
            vels.append(vel)
            json_weightfile = open(file[:-5] + "_weights.txt", "r")
            weights_data = json.load(json_weightfile)
            json_weightfile.close()
            weights.append(weights_data["weights"][-1])
            is_reactive.append(0)

        vels = np.asarray(vels)
        weights = np.asarray(weights)
        is_reactive = np.asarray(is_reactive)

    print(vels.shape, weights.shape, is_reactive.shape)

    inds_for_sort = np.argsort(vels)
    uniques_vels, index_to_split = np.unique(vels[inds_for_sort], return_index=True)

    all_weights = np.split(weights[inds_for_sort], index_to_split[1:])
    reactives_weights = np.split(weights[inds_for_sort] * is_reactive[inds_for_sort], index_to_split[1:])

    committor_values = np.array([np.sum(arr) for arr in reactives_weights]) / np.array([np.sum(arr) for arr in all_weights])

    return uniques_vels, committor_values


# TODO: pour les 3 fonctions suivante, on doit retourner une fonction et il faut écrire une fonction qui sait sampler depuis cette fonction et calculer le poids
def non_parametric_committor_estimation(v, comm):
    """
    Obtain a non-parametric estimation of the committor from data points
    """

    model = KernelReg(comm, v, "c", "lc", bw=[bandwidth])


def parametric_committor_estimation_tanh(v, comm):
    """
    Obtain a parametric estimation of the committor from data points
    """


def parametric_committor_estimation_log(v, comm):
    """
    Obtain a parametric estimation of the committor from data points
    """
