import os
import ase
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import ase.units as units
from scipy.stats import rayleigh


from src.aseams.cvs import CollectiveVariables

def distance(atoms):
    return atoms.get_distance(0, 1, mic=True)


def dist_grad(atoms):
    """
    Compute gradient of the cv with respect to atomic positions
    """
    r = atoms.get_positions()[[1], :] - atoms.get_positions()[[0], :]
    grad_r = ase.geometry.get_distances_derivatives(r, cell=atoms.cell, pbc=atoms.pbc)
    return grad_r.squeeze()



cv = CollectiveVariables(cv_r=distance,
                         cv_p=distance,
                         reaction_coordinate=distance,
                         rc_grad=dist_grad,
                         cv_r_grad=[dist_grad])
cv.set_r_crit("below")
cv.set_in_r_boundary(1.03)
cv.set_sigma_r_level(1.05)
cv.set_out_of_r_zone(1.5)
cv.set_p_crit("above")
cv.set_in_p_boundary(1.9)

# --- CONFIGURATION ---
# --- CONFIGURATION ---
ref_dir = "./ini_mc_unbiased"
biased_dir = "./ini_mc_flux_6.0"  # À adapter
n_bins = 20
temp_physique = 300.0
ref_atoms = read(os.path.join(ref_dir, os.listdir(ref_dir)[0]))
grad = cv.rc_grad(ref_atoms)  # Doit être de forme (N, 3)
grad_c = grad.copy()
for c in ref_atoms.constraints:
    if hasattr(c, 'adjust_forces'):
        c.adjust_forces(ref_atoms, grad_c)
masses = ref_atoms.get_masses()
G = np.sum(np.linalg.norm(grad_c, axis=1)**2 / masses)

# sigma = sqrt(kT / mu)
sigma_flux = np.sqrt(units.kB * temp_physique * G )
norm_grad_c = np.sqrt(np.sum(grad_c**2))
sigma_vn = sigma_flux / norm_grad_c

v_flux_list_ref = []
v_normal_list_ref = []
weights_ref = []

# --- LECTURE DES DONNÉES ---
files = [f for f in os.listdir(ref_dir) if f.endswith('.extxyz')]
print(f"Analyse de {len(files)} fichiers...")

for f in files:
    atoms = read(os.path.join(ref_dir, f))
    v = atoms.get_velocities()
    #atoms.calc = calc
    grad = cv.cv_r_grad[0](atoms).flatten()
    norm_g = np.linalg.norm(grad)
    v_flat = v.flatten()

    # --- LES DEUX MESURES ---
    # A. Le Flux : z_point = v . grad
    z_dot = np.dot(v_flat, grad)

    # B. La Vitesse Normale : vn = (v . grad) / ||grad||
    v_n = z_dot / norm_g

    v_flux_list_ref.append(z_dot)
    v_normal_list_ref.append(v_n)
    weights_ref.append(atoms.info.get("weight_ini_cond", 1.0))

v_flux_list_bias = []
v_normal_list_bias = []
weights_bias = []

# --- LECTURE DES DONNÉES ---
files = [f for f in os.listdir(biased_dir) if f.endswith('.extxyz')]
print(f"Analyse de {len(files)} fichiers...")

for f in files:
    atoms = read(os.path.join(biased_dir, f))
    v = atoms.get_velocities()
    #atoms.calc = calc
    grad = cv.cv_r_grad[0](atoms).flatten()
    norm_g = np.linalg.norm(grad)
    v_flat = v.flatten()

    # --- LES DEUX MESURES ---
    # A. Le Flux : z_point = v . grad
    z_dot = np.dot(v_flat, grad)

    # B. La Vitesse Normale : vn = (v . grad) / ||grad||
    v_n = z_dot / norm_g

    v_flux_list_bias.append(z_dot)
    v_normal_list_bias.append(v_n)
    weights_bias.append(atoms.info.get("weight_ini_cond", 1.0))


# --- VISUALISATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

x = np.linspace(0, max(max(v_flux_list_bias), max(v_flux_list_bias)), 200)

# --- PLOT 1 : LE FLUX (z_dot) ---
ax1.hist(v_flux_list_ref, bins=n_bins, density=True, weights=weights_ref, alpha=0.3, color='green',
         label="Flux échantillonné par MD")
ax1.hist(v_flux_list_bias, bins=n_bins, density=True, weights=weights_bias, alpha=0.3, color='blue',
         label="Flux IS reweighted")
ax1.hist(v_flux_list_bias, bins=n_bins, density=True, weights=np.ones_like(weights_bias), alpha=0.3, color='red', label="Flux IS not reweighted")
ax1.plot(x, rayleigh.pdf(x, scale=sigma_flux), 'r-', lw=2, label=f"Théorie Rayleigh ($\sigma=\sqrt{{kT/\mu}}$)")
ax1.set_title("Distribution du FLUX (Variation de distance)")
ax1.set_xlabel("$\dot{\zeta}$ (Å/fs)")
ax1.legend()

# --- PLOT 2 : LA VITESSE NORMALE (v_n) ---
# Le sigma théorique de la vitesse projetée est réduit par la norme du gradient
ax2.hist(v_normal_list_ref, bins=n_bins, density=True, weights=weights_ref, alpha=0.3, color='green',
         label="Vitesse normale échantillonné par MD")
ax2.hist(v_normal_list_bias, bins=n_bins, density=True, weights=weights_bias, alpha=0.3, color='blue', label="Vitesse normale IS reweighted($v_n$)")
ax2.hist(v_normal_list_bias, bins=n_bins, density=True, weights=np.ones_like(weights_bias), alpha=0.3, color='red', label="Vitesse normale IS not weighted($v_n$)")
ax2.plot(x, rayleigh.pdf(x, scale=sigma_vn), 'r-', lw=2,
         label=f"Théorie Rayleigh ($\sigma_{{flux}} / \|\\nabla \zeta\|$)")
ax2.set_title("Distribution de la VITESSE NORMALE")
ax2.set_xlabel("$v_n$ (Å/fs)")
ax2.legend()


plt.tight_layout()
plt.show()

print(f"Sigma Flux Théorique : {sigma_flux:.5f}")
print(f"Sigma V_n Théorique : {sigma_vn:.5f}")
print("Empirical mean MD", np.average(v_flux_list_ref, weights=weights_ref))
print("Empirical mean IS", np.average(v_flux_list_bias, weights=weights_bias))
print("Theoretical mean", sigma_flux * np.sqrt(np.pi / 2))
