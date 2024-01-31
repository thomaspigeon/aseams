#!/usr/bin/env python
# coding: utf-8

import numpy as np


class CollectiveVariables:
    """Class to gather collective variables used to sample initial conditions and run AMS"""

    def __init__(self, model, cv_r, cv_p, reaction_coordinate=None, domain_r=0.0):
        """
        Parameters:

        model: dimensionnality reduction from R^3N to R^(1,2 or 3)

        cv_r: function or list of fct taking as argument an atoms object and returning a float or a list of such functions

        cv_p: function taking as argument an atoms object and returning a float or a list of such functions

        reaction_coordinate: function taking as argument an atoms object and returning a float
        """
        self.model = model
        if (isinstance(cv_r, list) and np.prod([callable(_) for _ in cv_r])) or callable(cv_r):
            self.cv_r = cv_r
        else:
            raise ValueError("""cv_r must be either a function or a list of functions""")
        if (isinstance(cv_p, list) and np.prod([callable(_) for _ in cv_p])) or callable(cv_p):
            self.cv_p = cv_p
        else:
            raise ValueError("""cv_p must be either a function or a list of functions""")

        if reaction_coordinate is None:
            reaction_coordinate = LinearCommittor(model, cv_r, cv_p)
        if callable(reaction_coordinate):
            self.rc = reaction_coordinate
        else:
            raise ValueError("""reaction_coordinate must be a function""")

        self.domain_r = domain_r

    def test_the_collective_variables(self, atoms):
        """Test whether the collective variable was properly set and whether it "works" with a given atoms
        (molecular structure)

        Parameters:

        atoms: Atoms object
            A molecular structure for which at least the atomic positions are set for which the values of cv_r, cv_p and
            reaction coordinate can be computed
        """
        for fct in [self.in_which_r, self.above_which_sigma, self.in_which_p]:
            res = fct(atoms)
            try:
                [bool(v) for v in res]
            except:
                raise ValueError("The fct {} does not return appropriate value. Get {}".format(fct.__name__, res))

        if not isinstance(self.rc(atoms), float):
            raise ValueError(
                """The function reaction_coordinate does not returns a float, cannot run AMS or initial conditions
                            sampler with this type of structures. If the structure atoms is the one desired, the CollectiveVariables
                            object is not properly set and you should modify reaction_coordinate"""
            )
        self.is_out_of_r_zone(atoms)

    def in_which_r(self, atoms):
        """Evaluate whether the structure atoms is in the R substate

        Parameters:

        atoms: Atoms object

        Return

        list of booleans
        """
        s = self.model(atoms)
        if isinstance(self.cv_r, list):
            return np.array([r(s) for r in self.cv_r]).ravel()
        else:
            return self.cv_r(s)

    def in_r(self, atoms):
        """Evaluate whether the structure atoms is in the R state

        Parameters:

        atoms: Atoms object

        Return

        booleans
        """
        return np.max(self.in_which_r(atoms))

    def above_which_sigma(self, atoms):
        """Evaluate whether the structure atoms is beyond the Sigma_R of each sub-state

        Parameters:

        atoms: Atoms object

        Return

        list of booleans
        """
        s = self.model(atoms)
        if isinstance(self.cv_r, list):
            return np.array([np.logical_not(cv.sigma(s)) for r in self.cv_r]).ravel()
        else:
            return np.logical_not(self.cv_r.sigma(s))

    def above_sigma(self, atoms):
        return np.max(self.above_which_sigma(atoms))

    def is_out_of_r_zone(self, atoms):
        """Evaluate whether the structure atoms is out of the metastable bassin of R

        Parameters:

        atoms: Atoms object

        Return

        boolean
        """
        return self.rc(atoms) >= self.domain_r

    def in_which_p(self, atoms):
        """Evaluate whether the structure atoms is in the P substate

        Parameters:

        atoms: Atoms object

        Return

        list of booleans
        """
        s = self.model(atoms)
        if isinstance(self.cv_p, list):
            return np.array([r(s) for r in self.cv_p]).ravel()
        else:
            return self.cv_p(s)

    def in_p(self, atoms):
        """Evaluate whether the structure atoms is in the P state

        Parameters:

        atoms: Atoms object

        Return

        booleans
        """
        return np.max(self.in_which_p(atoms))


class LinearCommittor:
    """
    Given 2 states do linear interpolation between both
    """

    def __init__(self, model, cv_r, cv_p):
        self.model = model
        if (isinstance(cv_r, list) and np.prod([callable(_) for _ in cv_r])) or callable(cv_r):
            self.cv_r = cv_r
        else:
            raise ValueError("""cv_r must be either a function or a list of functions""")
        if (isinstance(cv_p, list) and np.prod([callable(_) for _ in cv_p])) or callable(cv_p):
            self.cv_p = cv_p
        else:
            raise ValueError("""cv_p must be either a function or a list of functions""")

    def __call__(self, atoms):
        """
        Return the linear committor try function
        """

        s = self.model(atoms)
        return self.in_reduced_space(s)[0]

    def in_reduced_space(self, s):
        if isinstance(self.cv_r, list):
            d_r = np.concatenate([r.dist_to_state(s) for r in self.cv_r], axis=-1)
        else:
            d_r = self.cv_r.dist_to_state(s)
        if isinstance(self.cv_p, list):
            d_p = np.concatenate([r.dist_to_state(s) for r in self.cv_p], axis=-1)
        else:
            d_p = self.cv_p.dist_to_state(s)
        d_r = np.min(d_r, axis=-1)
        d_p = np.min(d_p, axis=-1)
        return d_r / (d_r + d_p)


if __name__ == "__main__":
    from state_from_points import EnclosingCircleState, DensityState
    from ase.io import vasp, read
    import matplotlib.pyplot as plt

    def simple_descriptors(atoms):
        d1, d2 = atoms.get_distances(2, [1, 3], mic=True)
        dOC = atoms.get_distances(15, [0, 1, 2, 3], mic=True)
        beta = 5.0
        smooth_mindOC = -np.log(np.exp(-beta * dOC).sum()) / beta
        return d1 - d2, smooth_mindOC

    data_folder = "../../isobutanol/data_gpaw/"
    # Create dataset ==============================
    atoms_R = read(data_folder + "R/300K/md_1.traj", index="::10")

    X_R = np.empty((len(atoms_R), 2))
    for n, at in enumerate(atoms_R):
        X_R[n, :] = simple_descriptors(at)

    sR = EnclosingCircleState(X_R)

    atoms_P1 = read(data_folder + "P1/300K/md_1.traj", index="::10")

    X_P1 = np.empty((len(atoms_P1), 2))
    for n, at in enumerate(atoms_P1):
        X_P1[n, :] = simple_descriptors(at)
    sP1 = DensityState(X_P1, state_level=0.8, k_max=15, sigma_factor=1.2)

    atoms_P2 = read(data_folder + "P2/300K/md_2.traj", index="::10")

    X_P2 = np.empty((len(atoms_P2), 2))
    for n, at in enumerate(atoms_P2):
        X_P2[n, :] = simple_descriptors(at)
    sP2 = DensityState(X_P2, state_level=0.8, k_max=15, sigma_factor=1.2)

    cv = CollectiveVariables(simple_descriptors, sR, [sP1, sP2])

    for ref in ["R", "P1", "P2", "TS"]:
        atoms = vasp.read_vasp("../../isobutanol/vasp_files/POSCAR.{}".format(ref))

        print(ref, "CV:", cv.rc(atoms), "which R:", cv.in_which_r(atoms), "which P:", cv.in_which_p(atoms))

    cv.test_the_collective_variables(atoms)

    # Analyse ==============================

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    x = np.linspace(-1.5, 1.5, 250)
    y = np.linspace(0, 5.0, 250)
    xx, yy = np.meshgrid(x, y)
    X_eval = np.vstack([xx.ravel(), yy.ravel()]).T

    res = cv.rc.in_reduced_space(X_eval)
    h = ax.contourf(x, y, res.reshape(x.shape[0], y.shape[0]))

    res = sR(X_eval)
    h = ax.contour(x, y, res.reshape(x.shape[0], y.shape[0]), colors="red")
    res = sR.sigma(X_eval)
    h = ax.contour(x, y, res.reshape(x.shape[0], y.shape[0]))

    res = sP1(X_eval)
    h = ax.contour(x, y, res.reshape(x.shape[0], y.shape[0]), colors="red")
    res = sP1.sigma(X_eval)
    h = ax.contour(x, y, res.reshape(x.shape[0], y.shape[0]))

    res = sP2(X_eval)
    h = ax.contour(x, y, res.reshape(x.shape[0], y.shape[0]), colors="red")
    res = sP2.sigma(X_eval)
    h = ax.contour(x, y, res.reshape(x.shape[0], y.shape[0]))

    ax.scatter(X_R[:, 0], X_R[:, 1])
    ax.scatter(X_P1[:, 0], X_P1[:, 1])
    ax.scatter(X_P2[:, 0], X_P2[:, 1])

    for ref in ["R", "P1", "P2", "TS"]:
        atoms = vasp.read_vasp("../../isobutanol/vasp_files/POSCAR.{}".format(ref))
        pos = simple_descriptors(atoms)
        ax.scatter(pos[0], pos[1], marker="x", color="red")
        # for trj in trajs:
        #     s = m(trj).numpy()
        #     ax.plot(s[:, 0], s[:, 1], "-")

    ax.set_xlabel(f"CV {0}")
    ax.set_ylabel(f"CV {1}")
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([0, 5.0])
    plt.show()
