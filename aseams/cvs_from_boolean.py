#!/usr/bin/env python
# coding: utf-8

import numpy as np

class CollectiveVariables:
    """Class to gather collective variables used to sample initial conditions and run AMS"""

    def __init__(self, cv_r, cv_p, reaction_coordinate, domain_r=0.0, sigma_level=0.0, rc_grad=None):
        """
        Parameters:

        model: dimensionnality reduction from R^3N to R^(1,2 or 3)

        cv_r: function or list of fct taking as argument an atoms object and returning a float or a list of such functions

        cv_p: function taking as argument an atoms object and returning a float or a list of such functions

        reaction_coordinate: function taking as argument an atoms object and returning a float
        """
        if (isinstance(cv_r, list) and np.prod([callable(_) for _ in cv_r])) or callable(cv_r):
            self.cv_r = cv_r
        else:
            raise ValueError("""cv_r must be either a function or a list of functions""")
        if (isinstance(cv_p, list) and np.prod([callable(_) for _ in cv_p])) or callable(cv_p):
            self.cv_p = cv_p
        else:
            raise ValueError("""cv_p must be either a function or a list of functions""")

        if callable(reaction_coordinate):
            self.rc = reaction_coordinate
        else:
            raise ValueError("""reaction_coordinate must be a function""")

        self.domain_r = domain_r
        self.sigma_level = sigma_level
        if rc_grad is None:
            if hasattr(self.rc, "grad"):
                self.rc_grad = self.rc.grad
            else:
                print("No gradient available for RC")
        else:
            self.rc_grad = rc_grad

    def test_the_collective_variables(self, atoms):
        """Test whether the collective variable was properly set and whether it "works" with a given atoms
        (molecular structure)

        Parameters:

        atoms: Atoms object
            A molecular structure for which at least the atomic positions are set for which the values of cv_r, cv_p and
            reaction coordinate can be computed
        """
        for fct in [self.in_which_r, self.in_which_p]:
            res = fct(atoms)
            try:
                [bool(v) for v in res]
            except:
                raise ValueError("The fct {} does not return appropriate value. Get {}".format(fct.__name__, res))

        if not isinstance(self.rc(atoms), float):
            rc = self.rc(atoms)
            if not (isinstance(rc, np.ndarray) and (rc.shape == (1,) or rc.shape == ())):
                raise ValueError(
                    """The function reaction_coordinate does not returns a float, cannot run AMS or initial conditions
                                sampler with this type of structures. If the structure atoms is the one desired, the CollectiveVariables
                                object is not properly set and you should modify reaction_coordinate"""
                )
        self.above_sigma(atoms)
        self.is_out_of_r_zone(atoms)

    def rc_vel(self, atoms):
        grad, inclued_coords = self.rc_grad(atoms)
        velocities = atoms.get_momenta() / atoms.get_masses()[:, np.newaxis]
        return np.dot(grad.ravel(), velocities.ravel()[np.ravel_multi_index(inclued_coords, velocities.shape)])

    def in_which_r(self, atoms):
        """Evaluate whether the structure atoms is in the R substate

        Parameters:

        atoms: Atoms object

        Return

        list of booleans
        """
        if isinstance(self.cv_r, list):
            return np.array([r(atoms) for r in self.cv_r]).ravel()
        else:
            return self.cv_r(atoms)

    def in_r(self, atoms):
        """Evaluate whether the structure atoms is in the R state

        Parameters:

        atoms: Atoms object

        Return

        booleans
        """
        return np.max(self.in_which_r(atoms))

    def above_sigma(self, atoms):
        return self.rc(atoms) >= self.sigma_level

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
        if isinstance(self.cv_p, list):
            return np.array([r(atoms) for r in self.cv_p]).ravel()
        else:
            return self.cv_p(atoms)

    def in_p(self, atoms):
        """Evaluate whether the structure atoms is in the P state

        Parameters:

        atoms: Atoms object

        Return

        booleans
        """
        return np.max(self.in_which_p(atoms))


