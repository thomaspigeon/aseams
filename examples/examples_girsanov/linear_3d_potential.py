import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class LinearDoubleWell2D:
    def __init__(self, theta_vec=None):
        if theta_vec is None:
            self.theta_vec = np.array([ -2.0, 1., 4.0])
        else:
            self.theta_vec = np.array(theta_vec)

    def get_descriptors(self, x_2d):
        x = x_2d[0, 0]
        y = x_2d[0, 1]

        D = np.array(  [x**2, x**4, y**2])

        J_D = np.array([
            
            [2.0 * x, 0.0],
            [4.0 * x**3, 0.0],
            [0.0, 2.0 * y]
        ])

        return D, J_D


class LinearTriatomicPotentialCalculator(Calculator):
    """
    Potentiel pour atomes A B C.
    Calculateur qui prend en entrée un potentiel 2D linéaire
    et y concatène un potentiel 1D harmonique angulaire.
    Une transformation affine est présente :

    1. Transformation des distances :
    x = (d_AB - center) / scale
    y = d_BC - 1.0
    avec center = (d1 + d2) / 2 et scale = (d2 - d1) / 2.
    Ici, d1 = 1.0 et d2 = 2.0.

    2. Expression de l'énergie :
    L'énergie totale est le produit scalaire entre le vecteur des 
    paramètres theta_tot et le vecteur des descripteurs D :
    E = dot(theta_tot, D)

    3. Composition du vecteur D :
    D = [ D_2D(x, y), 0.5 * (theta - theta_0)**2 ]
    où theta est l'angle formé par les vecteurs BA et BC.

    4. Paramètres :
    theta_tot = [ theta_vec_2D, k_theta ]
    """

    implemented_properties = [
        'energy',
        'forces',
        'stress',
        'descriptors',
        'grad_descriptors'
    ]

    default_parameters = {
        'k_theta': 1.0,
        'theta_0': np.pi / 2
    }

    def __init__(self, potential_2d, **kwargs):
        super().__init__(**kwargs)
        self.potential_2d = potential_2d

    def calculate(
        self,
        atoms=None,
        properties=['energy', 'forces', 'stress', 'descriptors', 'grad_descriptors'],
        system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)

        d1, d2 = 1.0, 2.0
        center = (d1 + d2) / 2.0
        scale = (d2 - d1) / 2.0

        pos = self.atoms.positions
        rA, rB, rC = pos[0], pos[1], pos[2]

        vBA = rA - rB
        vBC = rC - rB

        dAB = np.linalg.norm(vBA)
        dBC = np.linalg.norm(vBC)

        eps_dist = 1e-12

        uBA = vBA / max(dAB, eps_dist)
        uBC = vBC / max(dBC, eps_dist)

        x_val = (dAB - center) / scale
        y_val = dBC - 1.0

        x_2d = np.array([[x_val, y_val]])

        D_2d, J_D_2d = self.potential_2d.get_descriptors(x_2d)

        theta_2d = self.potential_2d.theta_vec
        theta_0 = self.parameters['theta_0']
        k_theta = self.parameters['k_theta']

        cos_theta = np.clip(np.dot(uBA, uBC), -1.0, 1.0)
        theta = np.arccos(cos_theta)

        d_theta = theta - theta_0

        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))

        D_theta = 0.5 * d_theta**2

        D = np.concatenate((D_2d, [D_theta]))
        theta_tot = np.concatenate((theta_2d, [k_theta]))

        grad_q_rA = np.zeros((3, 3))
        grad_q_rB = np.zeros((3, 3))
        grad_q_rC = np.zeros((3, 3))

        grad_q_rA[0] = uBA / scale
        grad_q_rB[0] = -uBA / scale

        grad_q_rB[1] = -uBC
        grad_q_rC[1] = uBC

        if sin_theta > 1e-12:
            gA = (uBC - cos_theta * uBA) / dAB
            gC = (uBA - cos_theta * uBC) / dBC

            grad_q_rA[2] = -gA / sin_theta
            grad_q_rC[2] = -gC / sin_theta
            grad_q_rB[2] = -(grad_q_rA[2] + grad_q_rC[2])

        M_2d = len(D_2d)

        grad_D_rA = np.zeros((M_2d + 1, 3))
        grad_D_rB = np.zeros((M_2d + 1, 3))
        grad_D_rC = np.zeros((M_2d + 1, 3))

        for k in range(M_2d):
            dx = J_D_2d[k, 0]
            dy = J_D_2d[k, 1]

            grad_D_rA[k] = dx * grad_q_rA[0] + dy * grad_q_rA[1]
            grad_D_rB[k] = dx * grad_q_rB[0] + dy * grad_q_rB[1]
            grad_D_rC[k] = dx * grad_q_rC[0] + dy * grad_q_rC[1]

        grad_D_rA[-1] = d_theta * grad_q_rA[2]
        grad_D_rB[-1] = d_theta * grad_q_rB[2]
        grad_D_rC[-1] = d_theta * grad_q_rC[2]

        energy = float(np.dot(theta_tot, D))

        F_A = -np.dot(theta_tot, grad_D_rA)
        F_B = -np.dot(theta_tot, grad_D_rB)
        F_C = -np.dot(theta_tot, grad_D_rC)

        forces = np.array([F_A, F_B, F_C])

        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['descriptors'] = D
        self.results['grad_descriptors'] = np.array([
            grad_D_rA,
            grad_D_rB,
            grad_D_rC
        ])

        if 'stress' in properties:
            volume = self.atoms.get_volume()

            stress_tensor = np.zeros((3, 3))

            for i in range(3):
                stress_tensor += np.outer(pos[i], forces[i])

            stress_tensor = 0.5 * (stress_tensor + stress_tensor.T)

            res = stress_tensor / volume

            self.results['stress'] = np.array([
                res[0, 0],
                res[1, 1],
                res[2, 2],
                res[1, 2],
                res[0, 2],
                res[0, 1]
            ])

    def plot_2D(
        self,
        r1_range=(0.5, 2.5),
        r2_range=(0.5, 2.5),
        n_points=100,
        theta=None
    ):
        """
        Génère une carte de chaleur de l'énergie potentielle
        en fonction de r1 (AB) et r2 (BC).
        """

        import matplotlib.pyplot as plt
        from ase import Atoms

        r1_vec = np.linspace(*r1_range, n_points)
        r2_vec = np.linspace(*r2_range, n_points)

        R1, R2 = np.meshgrid(r1_vec, r2_vec)

        energy_grid = np.zeros((n_points, n_points))

        if theta is None:
            theta = self.parameters['theta_0']

        for i in range(n_points):
            for j in range(n_points):
                r_ab = R1[i, j]
                r_bc = R2[i, j]

                pos = np.array([
                    [r_ab, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [
                        r_bc * np.cos(theta),
                        r_bc * np.sin(theta),
                        0.0
                    ]
                ])

                atoms = Atoms('HOH', positions=pos)
                atoms.calc = self

                energy_grid[i, j] = atoms.get_potential_energy()

        plt.figure(figsize=(8, 6))

        contour = plt.contourf(
            R1,
            R2,
            energy_grid,
            levels=50,
            cmap='viridis'
        )

        plt.colorbar(contour, label='Énergie Potentielle')

        plt.xlabel('$r_1$ (AB) [Å]')
        plt.ylabel('$r_2$ (BC) [Å]')

        plt.title(
            f"Surface d'Énergie Potentielle ($\\theta = {np.degrees(theta):.1f}^o$)"
        )