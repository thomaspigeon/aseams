import numpy as np
import glob
import os
import warnings
from ase import units
from ase.io import read, write
from ase.parallel import world
import json
from statsmodels.nonparametric.kernel_regression import KernelReg

import scipy.integrate
from scipy.special import roots_laguerre
from scipy.integrate import quad

# def gauss_laguerre_quad(f,sigma, n):
#     """Quadrature de Gauss-Laguerre avec n points."""
#     nodes, weights = roots_laguerre(n)
#     return np.sum(weights *  f(np.sqrt(2*nodes)*sigma))


def infinite_integrale(f,sig2):
    """
    Compute Integrale of a function against a Rayleigh of parameter sigma, with error estimation
    """
    return quad(lambda x : f(np.sqrt(2*x*sig2))*np.exp(-x), 0,np.infty)[0]
    # I_n = np.zeros(100)
    # for n in range(5,50): 
    #     if n < 9 or n%2==0: # If we have already computed the integrale
    #         I_n[n] = gauss_laguerre_quad(f, sigma, n)
    #     I_n[2*n-1] = gauss_laguerre_quad(f, sigma, 2 * n - 1)
    #     if abs(I_n[2*n-1] - I_n[n]) <= eps:
    #         return I_n[2*n-1] ,abs(I_n[2*n-1] - I_n[n])
    # warnings.warn("Maximum order reached for integration")
    # return I_n[2*n-1],abs(I_n[2*n-1] - I_n[n])


def find_x_newton(f, df, target, x0=1.0, tol=1e-8, max_iter=50):
    """
    Trouve x tel que f(x) ≈ target en utilisant la méthode de Newton-Raphson.

    Paramètres :
        f        : fonction croissante
        df       : dérivée de f
        target   : valeur cible f(x) = target
        x0       : point de départ
        tol      : tolérance sur |f(x) - target|
        max_iter : nombre maximal d'itérations

    Retour :
        x tel que f(x) ≈ target
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Dérivée nulle — Newton échoue")
        dx = (fx - target) / dfx
        x -= dx
        if abs((fx - target)) < tol:
            return x
    raise ValueError("Méthode de Newton n'a pas convergé")





# -------------  Generating new velocities -------------


def bias_init_cond_velocity(rc_grad, ini_conds, ini_conds_biased, temp, bias_params, constraints=[]):
    """
    Write a set of new initial conditions with biased velocity

    Parameters
    -----------
    rc_grad: callable
        Gradient of the collective variable with respect to coordinates. Called as rc_grad(atoms)
    

    ini_conds: path
        Path towards initials conditions to be biased

    ini_conds_biased: path
        Path where biased initials conditions should be stored

    temp: float
        Temperature (in K) of the system

    bias_params: dict
        Parameters of the biasing for the importance function

    constraints:
        Set of contraints to be applied on the system

    """

    if not os.path.exists(ini_conds_biased) and world.rank == 0:
        os.makedirs(ini_conds_biased, exist_ok=True)

    rng = np.random.default_rng()
    # Once obtained data, we can bias the initial velocity
    for file in glob.glob(ini_conds + "*.extxyz"):
        atoms = read(file)
        for cons in constraints:  # If there is constraints
            atoms.set_constraint(cons)
        if bias_params["type"] == "rayleigh":
            new_velocity, weight, scale_normal_component = rayleigh_bias(temp, bias_params, rng)
        else:
            new_velocity, weight, scale_normal_component = committor_bias(temp, bias_params, rng)
        _ = set_velocities(rc_grad, atoms, new_velocity, temp, scale_normal_component=scale_normal_component)
        atoms.info["weight"] = weight  # Ici poids pour la fonction d'importance
        write(ini_conds_biased + os.path.basename(file), atoms)


def set_velocities(rc_grad, atoms, normal_component, temp_orthogonal, scale_normal_component=False):
    """
    Set velocities of atoms, given the normal component and orthogonal componants at a given temperatures
    If scale_normal_component is True divide the normal component by the norm of the gradient of the rc
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


#  -----------     Importance sampling of the normal components

def rayleigh_bias(temp, bias_param, rng=np.random.default_rng()):
    """
    Get a new velocity and associated weight

    Temp is initial temperature of the system

    bias_param contain all biais information

    rng is a random number generator
    """
    bias_temp = bias_param["bias_temp"]
    x = rng.rayleigh(scale=1.0)
    weight = bias_temp / temp * np.exp(-0.5 * (x ** 2) * (bias_temp / temp - 1.0))  # Ratio of 2 Rayleigh PDF for importance sampling
    return x * np.sqrt(units.kB * bias_temp), weight, False


def rayleigh(x, sig2):
    return x / sig2 * np.exp(-0.5 * x ** 2 / sig2)


def committor_bias(temp, bias_param, rng=np.random.default_rng()):
    """
    Get a new velocity and associated weight

    Temp is initial temperature of the system

    bias_param contain all biais information

    rng is a random number generator
    """
    x = np.interp(rng.random(), bias_param["cdf"], bias_param["cdf_vels"])  # Sample velocity from CDF
    weight = bias_param["norm"] / bias_param["committor_approx"](x)
    return x, float(weight), False



# ----------------------  Estimating bias function ---------------

def rc_vel(atoms, rc_grad, normalize=True):
    """
    Compute RC velocity for atoms given gradient of the RC with respect to atomic coordinates
    If normalize is True, this compute the projection on the normal of the RC
    """
    masses = atoms.get_masses()[:, np.newaxis]
    grad, inclued_coords = rc_grad(atoms)
    velocities = atoms.get_momenta() / masses
    if normalize:
        normal = np.zeros_like(atoms.get_momenta()).ravel()
        normal[np.ravel_multi_index(inclued_coords, velocities.shape)] = grad.ravel()
        norm = np.sqrt(np.dot(normal, (normal.reshape(-1, 3) / masses).ravel()))
    else:
        norm = 1.0
    return np.dot(grad.ravel(), velocities.ravel()[np.ravel_multi_index(inclued_coords, velocities.shape)]) / norm





def build_commitor_bias(ams_runs_paths, rc_grad, temp, committor_type="tanh", n_points_eval=1000, bandwidth=25, ceilling=1e-4):
    """
    Build estimation of the committor from previous AMS run for next iteration

    
    Parameters
    -----------

        ams_runs_paths: list
            List of paths of the previous AMS run to be used to estimate the committor in initial velocity

        rc_grad: callable
            Gradient of the collective variable with respect to coordinates. Called as rc_grad(atoms)
    
        temp: float
            Temperature (in K) of the system

        committor_type: ("tanh", "log" or "kernel")
            Which type of interpolation of the committor to be used

        n_points_eval: int
            Number of evaluation point to be taken for the CDF values of the velocity. Affect the quality of the sampling of the velocity

        
        bandwidth, ceilling: float
            parameters for the kernel estimation of the committor

    Return
    ----------

        bias_param: dict
        A set of parameters for applying committor biasing

    """
    v, comm = committor_estimation(ams_runs_paths, rc_grad)

    if np.max(comm) < 1e-12:
        warnings.warn("This is no reactive trajectories in your dataset, results wil be unreliable")

    bias_param = {"type": committor_type}
    # Committor value outside of the 0 and 1 values
    comm_valid_inds = np.logical_and(comm>0,comm<1)
    if committor_type == "tanh":
        
        poly = np.polynomial.Polynomial.fit(v[comm_valid_inds], np.arctanh(2 * comm[comm_valid_inds] - 1), 1)
        bias_param["committor_approx"] = lambda v: (0.5 * (1 + np.tanh(poly(v))))

        vmax = (-poly.coef[0]) / poly.coef[-1]

    elif committor_type == "log":
        poly = np.polynomial.Polynomial.fit(v[comm_valid_inds], np.log(comm[comm_valid_inds]), 1)
        bias_param["committor_approx"] = lambda v: np.minimum(np.exp(poly(v)), 1)

        vmax = (-poly.coef[0]) / poly.coef[-1]

    elif committor_type == "kernel":
        bandwidth = (np.max(v) - np.min(v)) / bandwidth
        model = KernelReg(comm, v, "c", "lc", bw=[bandwidth])
        bias_param["committor_approx"] = lambda v: ceilling + (1.0 - ceilling) * model.fit([v])[0]
        vmax = np.max(v)


    bias_param["norm"] =infinite_integrale(bias_param["committor_approx"], units.kB * temp)

    def cdf_val(vmax):
        return quad(lambda v : rayleigh(v, units.kB * temp) * bias_param["committor_approx"](v)/bias_param["norm"], 0,vmax)[0]
    
    def pdf_val(v):
        return rayleigh(v, units.kB * temp) * bias_param["committor_approx"](v)/bias_param["norm"]
    
    try:
        vmax=find_x_newton(cdf_val, pdf_val, 1-0.1/n_points_eval, vmax)
    except ValueError:
        warnings.warn("Unable to find a good max range for the cdf. Use heuristic")
        vmax=2*vmax


    # # The key parameter for discretizing the v range is the maximum values that have to be chosen such that biasing function is close to zero at max value
    # vmax = np.maximum(vmax, 4 * np.sqrt(units.kB * temp))
    # # Evaluate grossly where is the max
    # v_range = np.linspace(0.0, vmax, n_points_eval)
    # bias_vals = rayleigh(v_range, units.kB * temp) * bias_param["committor_approx"](v_range)
    # loc_max_bias = v_range[np.argmax(bias_vals)]
    # # we look for first value that is at 10 times lower than the max
    # ind = np.argmax(bias_vals[v_range > loc_max_bias] / np.max(bias_vals) < 0.01)
    # if ind == 0:  # In case no values are below the thresold
    #     ind = -1
    # alternative_vmax = v_range[v_range > loc_max_bias][ind]

    # vmax = np.minimum(vmax, alternative_vmax)  # Don't look to far

    
    bias_param["cdf_vels"] = np.linspace(0.0, vmax, n_points_eval)
    res_ivp = scipy.integrate.solve_ivp(lambda v, y: np.asarray([rayleigh(v, units.kB * temp) * bias_param["committor_approx"](v)/bias_param["norm"]]), [0, vmax], y0=[0.0], t_eval=bias_param["cdf_vels"])
    bias_param["cdf"] =res_ivp.y.squeeze()

    return bias_param


def committor_estimation(ams_runs_paths, rc_grad):
    """
    Estimate committor from previous AMS runs
    Parameters
    -----------

        ams_runs_paths: list
            List of paths of the previous AMS run to be used to estimate the committor in initial velocity

        rc_grad: callable
            Gradient of the collective variable with respect to coordinates. Called as rc_grad(atoms)
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
            if z_maxs[n] >= np.inf:
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

    inds_for_sort = np.argsort(vels)
    uniques_vels, index_to_split = np.unique(vels[inds_for_sort], return_index=True)

    all_weights = np.split(weights[inds_for_sort], index_to_split[1:])
    reactives_weights = np.split(weights[inds_for_sort] * is_reactive[inds_for_sort], index_to_split[1:])

    committor_values = np.array([np.sum(arr) for arr in reactives_weights]) / np.array([np.sum(arr) for arr in all_weights])

    return uniques_vels, committor_values
