import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import ase
import ase.units as units
from scipy.stats import rayleigh

# --- CONFIGURATION ---
# Chemin vers le dossier contenant vos .extxyz
input_dir = "./ini_mc_rayleigh_400.0"

temp_physique = 300.0       # Température du système
# Importer votre CV (exemple distance pour N2)
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

# --- CALCUL DE LA MASSE RÉDUITE ET SIGMA THÉORIQUE ---
# Masse de l'azote en unités ASE
m_N = 14.0067  # Environ 14.0067
mu = (m_N * m_N) / (m_N + m_N)  # Masse réduite pour N2

# sigma = sqrt(kT / mu)
sigma_th = np.sqrt((temp_physique * units.kB) / mu)
print("th", sigma_th)
# --- RÉCUPÉRATION DES DONNÉES ---
v_projections = []
weights = []
files = [f for f in os.listdir(input_dir) if f.endswith('.extxyz')]

print(f"Analyse de {len(files)} fichiers...")
for fname in files:
    atoms = read(os.path.join(input_dir, fname))
    velocities = atoms.get_velocities()
    grad_xi = cv.rc_grad(atoms)

    # Projection vn = (v . grad) / |grad|
    v_flat = velocities.flatten()
    g_flat = grad_xi.flatten()
    norm_g = np.linalg.norm(g_flat)

    if norm_g > 1e-10:
        v_n = np.dot(v_flat, g_flat) / norm_g
        v_projections.append(v_n)
        weights.append(atoms.info["weight"])
# --- VISUALISATION ---
plt.figure(figsize=(10, 6))

# Histogramme des données
count, bins, _ = plt.hist(v_projections, bins=20, density=True,
                          alpha=0.6, color='skyblue', edgecolor='black',
                          label=f"Sampled (Projetée sur $e_R$)")
count, bins, _ = plt.hist(v_projections, bins=20, density=True, weights=weights,
                          alpha=0.6, color='green', edgecolor='black',
                          label=f"Sampled reweighted (Projetée sur $e_R$)")

# Courbe théorique de Rayleigh
# La PDF de Rayleigh dans scipy est : f(x) = (x/s^2) * exp(-x^2 / (2s^2))
x = np.linspace(0, max(bins), 200)
pdf_rayleigh = rayleigh.pdf(x, scale=sigma_th)
plt.plot(x, pdf_rayleigh, 'r-', lw=2.5, label=f"Rayleigh Théorique (T={temp_physique}K)")

# Habillage
plt.xlabel(r"Vitesse normale $v_n$ (Å/fs)", fontsize=12)
plt.ylabel("Densité de probabilité", fontsize=12)
plt.title(f"Vérification de l'équilibre du flux à travers la surface\n(Distribution des vitesses initiales)",
          fontsize=14)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Affichage des stats
plt.text(0.6 * max(bins), 0.6 * max(pdf_rayleigh),
         f"$\sigma_{{th}} = {sigma_th:.4f}$\n$\langle v_n \\rangle_{{data}} = {np.mean(v_projections):.4f}$\n$\langle v_n \\rangle_{{th}} = {sigma_th * np.sqrt(np.pi / 2):.4f}$",
         bbox=dict(facecolor='white', alpha=0.8))

plt.show()