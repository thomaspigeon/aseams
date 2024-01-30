import numpy as np
import inspect
import os
import shutil
import json
from ase.io import Trajectory, read, write
from ase.parallel import parprint, paropen, world, barrier


class CollectiveVariables:
    """Class to gather collective variables used to sample initial conditions and run AMS"""

    def __init__(self, cv_r, cv_p, reaction_coordinate):
        """
        Parameters:

        cv_r: function taking as argument an atoms object and returning a float or a list of such functions

        cv_p: function taking as argument an atoms object and returning a float or a list of such functions

        reaction_coordinate: function taking as argument an atoms object and returning a float
        """

        if (isinstance(cv_r, list) and np.prod([inspect.isfunction(_) for _ in cv_r])) or inspect.isfunction(cv_r):
            self.cv_r = cv_r
        else:
            raise ValueError("""cv_r must be either a function or a list of functions""")
        if (isinstance(cv_p, list) and np.prod([inspect.isfunction(_) for _ in cv_p])) or inspect.isfunction(cv_p):
            self.cv_p = cv_p
        else:
            raise ValueError("""cv_p must be either a function or a list of functions""")
        if inspect.isfunction(reaction_coordinate):
            self.rc = reaction_coordinate
        else:
            raise ValueError("""reaction_coordinate must be a function""")
        self.r_crit = None
        self.p_crit = None
        self.in_r_boundary = None
        self.sigma_r_level = None
        self.out_of_r_zone = None
        self.in_p_boundary = None

    def set_r_crit(self, r_crit):
        """Set the criterion to identify where a structure is in R state

        Parameters

        r_crit: string or list of string, either "above", "between" or "below"
            If cv_r is a function, r_crit is a single string, if cv_r is a list, r_crit should be a list of strings
            matching the length of cv_r
        """
        if inspect.isfunction(self.cv_r):
            if r_crit != "above" and r_crit != "between" and r_crit != "below":
                raise ValueError("""When cv_r is a function, r_crit should either be "above", "between" or "below".""")
            else:
                self.r_crit = r_crit
        if isinstance(self.cv_r, list):
            if len(r_crit) != len(self.cv_r):
                raise ValueError(
                    """When cv_r is a list of functions,
                                    r_crit should be a list of string of the same length"""
                )
            else:
                self.r_crit = []
                for i in range(len(self.cv_r)):
                    if r_crit[i] != "above" and r_crit[i] != "between" and r_crit[i] != "below":
                        raise ValueError(
                            """When cv_r is a list of functions,
                                            r_crit should be a list of string being either
                                            "above", "between" or "below"."""
                        )
                    else:
                        self.r_crit.append(r_crit[i])

    def set_in_r_boundary(self, in_r_boundary):
        """Set the numerical value of the boundary to indentify whether a structure is in R state.

        Parameters

        in_r_boundary: float, or list of two floats or list of (floats or list two floats)
            in_r_boundary must match the definition of cv_r and r_crit.
            If cv_r is a function and r_crit is either "above" or "below", in_r_boundary must be a float
            If cv_r is a function and r_crit is "between", in_r_boundary must be a list of two floats in increasing
            order
            If cv_r is a list of functions, in_r_boundary must be a list of matching length. Each of its elements must
            be either a float or a list of two floats depending on the value of the corresponding element in r_crit
        """
        if inspect.isfunction(self.cv_r):
            if self.r_crit == "above" or self.r_crit == "below":
                if not isinstance(in_r_boundary, float):
                    raise ValueError(
                        """When cv_r is a function and r_crit is either "above" or "below", in_r_boundary
                                        must be a float"""
                    )
                else:
                    self.in_r_boundary = in_r_boundary
            elif self.r_crit == "between":
                if not (isinstance(in_r_boundary, list) and isinstance(in_r_boundary[0], float) and isinstance(in_r_boundary[1], float) and in_r_boundary[0] <= in_r_boundary[1]):
                    raise ValueError(
                        """When cv_r is a function and r_crit is "between", in_r_boundary
                                        must be list of two floats in increasing order"""
                    )
                else:
                    self.in_r_boundary = in_r_boundary
        if isinstance(self.cv_r, list):
            if len(in_r_boundary) != len(self.cv_r):
                raise ValueError(
                    """When cv_r is a list of functions,
                                    in_r_boundary should be a list of the same length"""
                )
            else:
                self.in_r_boundary = []
                for i in range(len(self.cv_r)):
                    if self.r_crit[i] == "above" or self.r_crit[i] == "below":
                        if not isinstance(in_r_boundary[i], float):
                            raise ValueError(
                                """When cv_r is a list of functions and r_crit[i] is either "above" or
                            "below", in_r_boundary[i] must be a float"""
                            )
                        else:
                            self.in_r_boundary.append(in_r_boundary[i])
                    elif self.r_crit[i] == "between":
                        if not (isinstance(in_r_boundary[i], list) and isinstance(in_r_boundary[i][0], float) and isinstance(in_r_boundary[i][1], float) and in_r_boundary[i][0] >= in_r_boundary[i][1]):
                            raise ValueError(
                                """When cv_r is a list of functions and r_crit[i] is "between",
                                                in_r_boundary[i] must be list of two floats in increasing order"""
                            )
                        else:
                            self.in_r_boundary.append(in_r_boundary[i])

    def set_sigma_r_level(self, sigma_r_level):
        """Set the numerical value of the boundary to indentify whether a structure an initial condition leaving R state

        Parameters

        sigma_r_level: float, or list of two floats or list of (floats or list two floats)
            sigma_r_level must match the definition of cv_r, r_crit and in_r_boundary.
            If cv_r is a function and r_crit is "below", sigma_r_level must be a float bigger than in_r_boundary
            If cv_r is a function and r_crit is "above", sigma_r_level must be a float smaller than in_r_boundary
            If cv_r is a function and r_crit is "between", sigma_r_level must be a list of two floats in increasing
            order such that: sigma_r_level[0] <= in_r_boundary[0] < in_r_boundary[1] <= sigma_r_level[1]
            If cv_r is a list of functions, sigma_r_level must be a list of matching length. Each of its elements must
            be either a float or a list of two floats depending on the value of the corresponding element in r_crit and
            consistent with the definition of in_r_boundary
        """
        if inspect.isfunction(self.cv_r):
            if self.r_crit == "above":
                if not (isinstance(sigma_r_level, float) and sigma_r_level <= self.in_r_boundary):
                    raise ValueError(
                        """When cv_r is a function and r_crit is "above", sigma_r_level must be a float
                                        smaller or equal to in_r_boundary"""
                    )
                else:
                    self.sigma_r_level = sigma_r_level
            elif self.r_crit == "below":
                if not (isinstance(sigma_r_level, float) and sigma_r_level >= self.in_r_boundary):
                    raise ValueError(
                        """When cv_r is a function and r_crit is "below", sigma_r_level must be a float
                                        greater or equal to in_r_boundary"""
                    )
                else:
                    self.sigma_r_level = sigma_r_level
            elif self.r_crit == "between":
                if not (isinstance(sigma_r_level, list) and isinstance(sigma_r_level[0], float) and isinstance(sigma_r_level[1], float) and sigma_r_level[0] <= self.in_r_boundary[0] and self.in_r_boundary[1] <= sigma_r_level[1]):
                    raise ValueError(
                        """When cv_r is a function and r_crit is "between", sigma_r_level must be a list of
                                        two floats in increasing order such that:
                                        sigma_r_level[0] <= in_r_boundary[0] < in_r_boundary[1] <= sigma_r_level[1]"""
                    )
                else:
                    self.sigma_r_level = sigma_r_level
        if isinstance(self.cv_r, list):
            if len(sigma_r_level) != len(self.cv_r):
                raise ValueError(
                    """When cv_r is a list of functions,
                                    sigma_r_level should be a list of the same length"""
                )
            else:
                self.sigma_r_level = []
                for i in range(len(self.cv_r)):
                    if self.r_crit[i] == "above":
                        if not (isinstance(sigma_r_level[i], float) and sigma_r_level[i] > self.in_r_boundary[i]):
                            raise ValueError(
                                """When cv_r is a list of functions and r_crit[i] is "above",
                                                sigma_r_level[i] must be a float smaller than in_r_boundary[i]"""
                            )
                        else:
                            self.sigma_r_level.append(sigma_r_level[i])
                    elif self.r_crit[i] == "below":
                        if not (isinstance(sigma_r_level[i], float) and sigma_r_level[i] < self.in_r_boundary[i]):
                            raise ValueError(
                                """When cv_r is a list of functions and r_crit[i] is "above",
                                                sigma_r_level[i] must be a float greater than in_r_boundary[i]"""
                            )
                        else:
                            self.sigma_r_level.append(sigma_r_level[i])
                    elif self.r_crit[i] == "between":
                        if not (isinstance(sigma_r_level[i], list) and isinstance(sigma_r_level[i][0], float) and isinstance(sigma_r_level[i][1], float) and sigma_r_level[i][0] <= self.in_r_boundary[i][0] and self.in_r_boundary[i][1] <= self.sigma_r_level[i][1]):
                            raise ValueError(
                                """When cv_r is a list of functions and r_crit[i] is "between",
                                                sigma_r_level must be a list of two floats in increasing order such that:
                                        sigma_r_level[i][0] <= in_r_boundary[i][0] < in_r_boundary[i][1] <= sigma_r_level[i][1]"""
                            )
                        else:
                            self.sigma_r_level.append(sigma_r_level[i])

    def set_out_of_r_zone(self, out_of_r_zone):
        """Set the numerical value of the boundary to indentify whether a structure is completely out of the R
        metastability bassin

        Parameters

        out_of_r_zone: float, or list of two floats or list of (floats or list two floats)
            out_of_r_zone must match the definition of cv_r, r_crit and sigma_r_level.
            If cv_r is a function and r_crit is "below", out_of_r_zone must be a float bigger than sigma_r_level
            If cv_r is a function and r_crit is "above", out_of_r_zone must be a float smaller than sigma_r_level
            If cv_r is a function and r_crit is "between", out_of_r_zone must be a list of two floats in increasing
            order such that: out_of_r_zone[0] <= sigma_r_level[0] < sigma_r_level[1] <= out_of_r_zone[1]
            If cv_r is a list of functions, out_of_r_zone must be a list of matching length. Each of its elements must
            be either a float or a list of two floats depending on the value of the corresponding element in r_crit and
            consistent with the definition of sigma_r_level
        """
        if inspect.isfunction(self.cv_r):
            if self.r_crit == "above":
                if not (isinstance(out_of_r_zone, float) and out_of_r_zone <= self.sigma_r_level):
                    raise ValueError(
                        """When cv_r is a function and r_crit is "above", out_of_r_zone must be a float
                                        smaller or equal to sigma_r_level"""
                    )
                else:
                    self.out_of_r_zone = out_of_r_zone
            elif self.r_crit == "below":
                if not (isinstance(out_of_r_zone, float) and out_of_r_zone >= self.sigma_r_level):
                    raise ValueError(
                        """When cv_r is a function and r_crit is "below", out_of_r_zone must be a float
                                        greater or equal to sigma_r_level"""
                    )
                else:
                    self.out_of_r_zone = out_of_r_zone
            elif self.r_crit == "between":
                if not (isinstance(out_of_r_zone, list) and isinstance(out_of_r_zone[0], float) and isinstance(out_of_r_zone[1], float) and out_of_r_zone[0] <= self.sigma_r_level[0] and self.sigma_r_level[1] <= out_of_r_zone[1]):
                    raise ValueError(
                        """When cv_r is a function and r_crit is "between", out_of_r_zone must be a list of
                                        two floats in increasing order such that:
                                        out_of_r_zone[0] <= sigma_r_level[0] < sigma_r_level[1] <= out_of_r_zone[1]"""
                    )
                else:
                    self.out_of_r_zone = out_of_r_zone
        if isinstance(self.cv_r, list):
            if len(out_of_r_zone) != len(self.cv_r):
                raise ValueError(
                    """When cv_r is a list of functions,
                                    out_of_r_zone should be a list of the same length"""
                )
            else:
                self.out_of_r_zone = []
                for i in range(len(self.cv_r)):
                    if self.r_crit[i] == "above":
                        if not (isinstance(out_of_r_zone[i], float) and out_of_r_zone[i] > self.sigma_r_level[i]):
                            raise ValueError(
                                """When cv_r is a list of functions and r_crit[i] is "above",
                                                out_of_r_zone[i] must be a float smaller than sigma_r_level[i]"""
                            )
                        else:
                            self.out_of_r_zone.append(out_of_r_zone[i])
                    elif self.r_crit[i] == "below":
                        if not (isinstance(out_of_r_zone[i], float) and out_of_r_zone[i] < self.sigma_r_level[i]):
                            raise ValueError(
                                """When cv_r is a list of functions and r_crit[i] is "above",
                                                out_of_r_zone[i] must be a float greater than sigma_r_level[i]"""
                            )
                        else:
                            self.out_of_r_zone.append(out_of_r_zone[i])
                    elif self.r_crit[i] == "between":
                        if not (isinstance(out_of_r_zone[i], list) and isinstance(out_of_r_zone[i][0], float) and isinstance(out_of_r_zone[i][1], float) and out_of_r_zone[i][0] <= self.sigma_r_level[i][0] and self.sigma_r_level[i][1] <= out_of_r_zone[i][1]):
                            raise ValueError(
                                """When cv_r is a list of functions and r_crit[i] is "between",
                                                sigma_r_level must be a list of two floats in increasing order such that:
                                        out_of_r_zone[i][0] <= sigma_r_level[i][0] < sigma_r_level[i][1] <= out_of_r_zone[i][1]"""
                            )
                        else:
                            self.out_of_r_zone.append(out_of_r_zone[i])

    def in_which_r(self, atoms):
        """Evaluate whether the structure atoms is in the R substate

        Parameters:

        atoms: Atoms object

        Return

        list of booleans
        """
        if isinstance(self.cv_r, list):
            in_r = []
            for i in range(len(self.cv_r)):
                if self.r_crit == "above":
                    in_r.append(self.cv_r[i](atoms) >= self.in_r_boundary[i])
                if self.r_crit == "below":
                    in_r.append(self.cv_r[i](atoms) <= self.in_r_boundary[i])
                if self.r_crit == "between":
                    in_r.append(self.in_r_boundary[i][0] <= self.cv_r[i](atoms) <= self.in_r_boundary[i][1])
            return in_r
        else:
            if self.r_crit == "above":
                return [self.cv_r(atoms) >= self.in_r_boundary]
            if self.r_crit == "below":
                return [self.cv_r(atoms) <= self.in_r_boundary]
            if self.r_crit == "between":
                return [self.in_r_boundary[0] <= self.cv_r(atoms) <= self.in_r_boundary[1]]

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
        if isinstance(self.cv_r, list):
            above_sigma_r = []
            for i in range(len(self.cv_r)):
                if self.r_crit == "above":
                    above_sigma_r.append(self.cv_r[i](atoms) < self.sigma_r_level[i])
                if self.r_crit == "below":
                    above_sigma_r.append(self.cv_r[i](atoms) > self.sigma_r_level[i])
                if self.r_crit == "between":
                    above_sigma_r.append(self.cv_r[i](atoms) < self.sigma_r_level[i][0] or self.cv_r[i](atoms) > self.sigma_r_level[i][1])
            return above_sigma_r
        else:
            if self.r_crit == "above":
                return [self.cv_r(atoms) < self.sigma_r_level]
            if self.r_crit == "below":
                return [self.cv_r(atoms) > self.sigma_r_level]
            if self.r_crit == "between":
                return [self.cv_r(atoms) < self.sigma_r_level[0] or self.cv_r(atoms) > self.sigma_r_level[1]]

    def above_sigma(self, atoms):
        return np.max(self.above_which_sigma(atoms))

    def is_out_of_r_zone(self, atoms):
        """Evaluate whether the structure atoms is out of the metastable bassin of R

        Parameters:

        atoms: Atoms object

        Return

        list of booleans
        """
        if isinstance(self.cv_r, list):
            out = []
            for i in range(len(self.cv_r)):
                if self.r_crit == "above":
                    out.append(self.cv_r[i](atoms) < self.out_of_r_zone[i])
                if self.r_crit == "below":
                    out.append(self.cv_r[i](atoms) > self.out_of_r_zone[i])
                if self.r_crit == "between":
                    out.append(self.cv_r[i](atoms) < self.out_of_r_zone[i][0] or self.cv_r[i](atoms) > self.out_of_r_zone[i][1])
            return np.prod(out)
        else:
            if self.r_crit == "above":
                return self.cv_r(atoms) < self.out_of_r_zone
            if self.r_crit == "below":
                return self.cv_r(atoms) > self.out_of_r_zone
            if self.r_crit == "between":
                return self.cv_r(atoms) < self.out_of_r_zone[0] or self.cv_r(atoms) > self.out_of_r_zone[1]

    def set_p_crit(self, p_crit):
        """Set the criterion to identify where a structure is in P state

        Parameters

        p_crit: string or list of string, either "above", "between" or "below"
            If cv_p is a function, p_crit is a single string, if cv_p is a list, p_crit should be a list of strings
            matching the length of cv_p
        """
        if inspect.isfunction(self.cv_p):
            if p_crit != "above" and p_crit != "between" and p_crit != "below":
                raise ValueError("""When cv_p is a function, p_crit should either be "above", "between" or "below".""")
            else:
                self.p_crit = p_crit
        if isinstance(self.cv_p, list):
            if len(p_crit) != len(self.cv_p):
                raise ValueError(
                    """When cv_p is a list of functions,
                                    p_crit should be a list of string of the same length"""
                )
            else:
                self.p_crit = []
                for i in range(len(self.cv_p)):
                    if p_crit[i] != "above" and p_crit[i] != "between" and p_crit[i] != "below":
                        raise ValueError(
                            """When cv_p is a list of functions,
                                            p_crit should be a list of string being either
                                            "above", "between" or "below"."""
                        )
                    else:
                        self.r_crit.append(p_crit[i])

    def set_in_p_boundary(self, in_p_boundary):
        """Set the numerical value of the boundary to indentify whether a structure is in P state.

        Parameters

        in_p_boundary: float, or list of two floats or list of (floats or list two floats)
            in_p_boundary must match the definition of cv_p and p_crit.
            If cv_p is a function and p_crit is either "above" or "below", in_p_boundary must be a float
            If cv_p is a function and p_crit is "between", in_p_boundary must be a list of two floats in increasing
            order
            If cv_p is a list of functions, in_p_boundary must be a list of matching length. Each of its elements must
            be either a float or a list of two floats depending on the value of the corresponding element in p_crit
        """
        if inspect.isfunction(self.cv_p):
            if self.p_crit == "above" or self.p_crit == "below":
                if not isinstance(in_p_boundary, float):
                    raise ValueError(
                        """When cv_p is a function and p_crit is either "above" or "below", in_p_boundary
                                        must be a float"""
                    )
                else:
                    self.in_p_boundary = in_p_boundary
            elif self.p_crit == "between":
                if not (isinstance(in_p_boundary, list) and isinstance(in_p_boundary[0], float) and isinstance(in_p_boundary[1], float) and in_p_boundary[0] < in_p_boundary[1]):
                    raise ValueError(
                        """When cv_p is a function and p_crit is "between", in_p_boundary
                                        must be list of two floats in increasing order"""
                    )
                else:
                    self.in_p_boundary = in_p_boundary
        if isinstance(self.cv_p, list):
            if len(in_p_boundary) != len(self.cv_p):
                raise ValueError(
                    """When cv_p is a list of functions,
                                    in_p_boundary should be a list of the same length"""
                )
            else:
                self.in_p_boundary = []
                for i in range(len(self.cv_p)):
                    if self.p_crit[i] == "above" or self.p_crit[i] == "below":
                        if not isinstance(in_p_boundary[i], float):
                            raise ValueError(
                                """When cv_p is a list of functions and p_crit[i] is either "above" or
                            "below", in_p_boundary[i] must be a float"""
                            )
                        else:
                            self.in_p_boundary.append(in_p_boundary[i])
                    elif self.p_crit[i] == "between":
                        if not (isinstance(in_p_boundary[i], list) and isinstance(in_p_boundary[i][0], float) and isinstance(in_p_boundary[i][1], float) and in_p_boundary[i][0] < in_p_boundary[i][1]):
                            raise ValueError(
                                """When cv_p is a list of functions and p_crit[i] is "between",
                                                in_p_boundary[i] must be list of two floats in increasing order"""
                            )
                        else:
                            self.in_p_boundary.append(in_p_boundary[i])

    def in_which_p(self, atoms):
        """Evaluate whether the structure atoms is in the P substate

        Parameters:

        atoms: Atoms object

        Return

        list of booleans
        """
        if isinstance(self.cv_p, list):
            in_p = []
            for i in range(len(self.cv_p)):
                if self.p_crit == "above":
                    in_p.append(self.cv_p[i](atoms) >= self.in_p_boundary[i])
                if self.p_crit == "below":
                    in_p.append(self.cv_p[i](atoms) <= self.in_p_boundary[i])
                if self.r_crit == "between":
                    in_p.append(self.in_p_boundary[i][0] <= self.cv_p[i](atoms) <= self.in_p_boundary[i][1])
            return in_p
        else:
            if self.p_crit == "above":
                return [self.cv_r(atoms) >= self.in_p_boundary]
            if self.p_crit == "below":
                return [self.cv_p(atoms) <= self.in_p_boundary]
            if self.p_crit == "between":
                return [self.in_p_boundary[0] <= self.cv_p(atoms) <= self.in_p_boundary[1]]

    def in_p(self, atoms):
        """Evaluate whether the structure atoms is in the P state

        Parameters:

        atoms: Atoms object

        Return

        booleans
        """
        return np.max(self.in_which_p(atoms))

    def test_the_collective_variables(self, atoms):
        """Test whether the collective variable was properly set and whether it "works" with a given atoms
        (molecular structure)

        Parameters:

        atoms: Atoms object
            A molecular structure for which at least the atomic positions are set for which the values of cv_r, cv_p and
            reaction coordinate can be computed
        """
        if inspect.isfunction(self.cv_r):
            if not isinstance(self.cv_r(atoms), float):
                raise ValueError(
                    """The function cv_r does not returns a float, cannot run AMS or initial conditions
                sampler with this type of structures. If the structure atoms is the one desired, the CollectiveVariables
                object is not properly set and you should modify cv_r"""
                )
        else:
            for i in range(len(self.cv_r)):
                if not isinstance(self.cv_r[i](atoms), float):
                    raise ValueError(
                        """The function cv_r["""
                        + str(i)
                        + """] does not returns a float, cannot run AMS or initial conditions
                                    sampler with this type of structures. If the structure atoms is the one desired, the CollectiveVariables
                                    object is not properly set and you should modify cv_r"""
                    )
        if inspect.isfunction(self.cv_p):
            if not isinstance(self.cv_p(atoms), float):
                raise ValueError(
                    """The function cv_p does not returns a float, cannot run AMS or initial conditions
                                sampler with this type of structures. If the structure atoms is the one desired, the CollectiveVariables
                                object is not properly set and you should modify cv_p"""
                )
        else:
            for i in range(len(self.cv_p)):
                if not isinstance(self.cv_p[i](atoms), float):
                    raise ValueError(
                        """The function cv_p["""
                        + str(i)
                        + """] does not returns a float, cannot run AMS or initial conditions
                                        sampler with this type of structures. If the structure atoms is the one desired, the CollectiveVariables
                                        object is not properly set and you should modify cv_p"""
                    )
        if not isinstance(self.rc(atoms), float):
            raise ValueError(
                """The function reaction_coordinate does not returns a float, cannot run AMS or initial conditions
                            sampler with this type of structures. If the structure atoms is the one desired, the CollectiveVariables
                            object is not properly set and you should modify reaction_coordinate"""
            )

    def evaluate_cv_r(self, atoms):
        """Evaluate all the functions cv_r

        Parameters

        atoms: Atoms object

        Returns:

        list of floats
        """
        if isinstance(self.cv_r, list):
            return [self.cv_r[i](atoms) for i in range(len(self.cv_r))]
        else:
            return [self.cv_r(atoms)]

    def evaluate_cv_p(self, atoms):
        """Evaluate all the functions cv_p

        Parameters

        atoms: Atoms object

        Returns:

        list of floats
        """
        if isinstance(self.cv_p, list):
            return [self.cv_p[i](atoms) for i in range(len(self.cv_p))]
        else:
            return [self.cv_p(atoms)]


class InitialConditionsSampler:
    """Class to sample initial conditions to run AMS later"""

    def __init__(self, dyn, xi, cv_interval=1):
        """An initial conditions sampler with a single replica

        Parameters:

        dyn: MolecularDynamics object
            Should be a stochastic dynamics

        xi: CollectiveVariable object
            Object that allows to measure whether the dynamics is in reactant (R) state, in product (P) state and the
            progress on the transitions

        cv_interval: int
            The CV is evaluated every cv_interval time steps
        """
        if type(xi).__name__ != "CollectiveVariables":
            raise ValueError("""xi must be a CollectiveVariables object""")
        if type(dyn).__name__ != "Langevin":
            raise ValueError("""dyn must be a Langevin object""")
        if isinstance(cv_interval, int) and cv_interval >= 0:
            self.cv_interval = cv_interval
        else:
            raise ValueError("""cv_interval must be an int >= 0""")
        xi.test_the_collective_variables(dyn.atoms)
        self.xi = xi
        self.dyn = dyn
        self.dyn.nsteps = 1
        self.calc = dyn.atoms.calc
        self.run_dir = None
        self.ini_cond_dir = None

    def set_run_dir(self, run_dir="./ini_conds_md_logs"):
        """Where the md logs will be written, if the directory does not exist, it will create it."""
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir) and world.rank == 0:
            os.mkdir(self.run_dir)
        traj = Trajectory(filename=self.run_dir + "/md_traj.traj", mode="a", atoms=self.dyn.atoms)
        self.dyn.attach(traj.write, interval=self.cv_interval)

    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds"):
        """Where the initial conditions for AMS will be written, if the directory does not exist, it will create it."""
        self.ini_cond_dir = ini_cond_dir
        if not os.path.exists(self.ini_cond_dir) and world.rank == 0:
            os.mkdir(self.ini_cond_dir)

    def sample(self, n_conditions=100, n_steps=None):
        """Sampling initial conditions

        Parameters:

        n_conditions: int
            Number of initial conditions to sample, default is 100. If this argument is given, n_steps should be kept to
            None
        n_steps: int
            Number of integration time steps that should be performed. If this argument is given, n_conditions should be
            set to None
        """
        if self.run_dir is None:
            raise ValueError("""The running directory is not set ! Call initialconditionssampler.set_run_dir""")
        if self.ini_cond_dir is None:
            raise ValueError("""The directory to store initial conditions is not defined ! Call initialconditionssampler.set_ini_cond_dir""")
        if n_steps is None:
            n_steps = -1
            if not isinstance(n_conditions, int) or n_conditions <= 0:
                raise ValueError("""n_conditions should be an int > 0 if n_steps is left to None""")
        elif n_conditions is None:
            n_conditions = -1
            if not isinstance(n_steps, int) or n_steps <= 0:
                raise ValueError("""n_steps should be an int > 0 if n_conditions is set to None""")
        else:
            raise ValueError(
                """n_conditions should be an int > 0 if n_steps is left to None or n_steps should be an
                                int > 0 if n_conditions is set to None"""
            )
        n_cdt, n_stp = 0, 0
        n_ini_conds_already = len(os.listdir(self.ini_cond_dir))
        if isinstance(self.xi.cv_r, list):
            t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
            t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
        else:
            t_r_sigma = [[]]
            t_sigma_r = [[]]
        while not self.xi.in_r(self.dyn.atoms):
            self.dyn.run(self.cv_interval)
            n_stp += self.cv_interval
        while n_cdt < n_conditions or n_stp < n_steps:
            which_r = np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
            t_r_sigma[which_r].append(0)
            t_sigma_r[which_r].append(0)
            while not self.xi.above_which_sigma(self.dyn.atoms)[which_r]:
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                t_r_sigma[which_r][-1] += self.cv_interval
            fname = self.ini_cond_dir + "/" + str(n_ini_conds_already + n_cdt + 1) + ".extxyz"
            write(fname, self.dyn.atoms)
            while not self.xi.in_r(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                t_sigma_r[which_r][-1] += self.cv_interval
                if self.xi.is_out_of_r_zone(self.dyn.atoms):
                    list_atoms = read(self.run_dir + "/md_traj.traj", index=":")
                    at = list_atoms[np.random.randint(len(list_atoms))]
                    self.dyn.atoms.set_scaled_positions(at.get_scaled_positions())
                    self.dyn.atoms.set_momenta(at.get_momenta())
            n_cdt += 1


class AMS:
    """Running AMS"""

    def __init__(
        self,
        n_rep,
        k_min,
        dyn,
        xi,
        cv_interval=1,
        rc_threshold=1.0e-3,
        save_all=False,
    ):
        """
        Parameters:

        n_rep: int
            Number of replicas, must be strictly greater than 1

        k_min: int
            minimum number of replicas to kill at each iteration, must be superior or equal to 1 and strictly smaller
            than n_rep

        dyn: MolecularDynamics object
            Should be a stochastic dynamics
            TODO: Take a list of n_rep dynamics for parallel runs, check NEB calculations

        xi: CollectiveVariable object
            Object that allows to measure whether the dynamics is in reactant (R) state, in product (P) state and the
            progress on the transitions

        cv_interval: int
            The CV is evaluated every cv_interval time steps

        rc_threshold: float
            The biggest difference between two rc values so that they are considered identical.

        save_all: boolean
            whether all the trajectories of the replicas should be saved. If false, only the current state of the
            replicas is written in the AMS.current_replicas_dir
        """
        if isinstance(n_rep, int) and n_rep > 1:
            self.n_rep = n_rep
        else:
            raise ValueError("""n_rep must be an int > 1""")
        if isinstance(k_min, int) and 1 <= k_min < n_rep:
            self.k_min = k_min
        else:
            raise ValueError("""k_min must be an int such that 1 =< k_min < n_rep""")
        if type(xi).__name__ != "CollectiveVariables":
            raise ValueError("""xi must be a CollectiveVariables object""")
        if type(dyn).__name__ != "Langevin":
            raise ValueError("""dyn must be a Langevin object""")
        if isinstance(cv_interval, int) and cv_interval >= 1:
            self.cv_interval = cv_interval
        else:
            raise ValueError("""cv_interval must be an int >= 0""")
        if isinstance(save_all, bool):
            self.save_all = save_all
        else:
            raise ValueError("""save_all must be a boolean""")
        xi.test_the_collective_variables(dyn.atoms)
        self.success = False
        self.initialized = False
        self.finished = False
        self.xi = xi
        self.dyn = dyn
        self.dyn.nsteps = 1
        self.ini_cond_dir = None
        self.alive_traj_dir = None
        self.z_maxs = None
        self.ams_it = 0
        self.current_p = 1
        self.rep_weights = [[] for i in range(n_rep)]
        self.z_kill = []
        self.killed = []
        self.calc = dyn.atoms.calc
        self.rc_threshold = rc_threshold
        if save_all:
            self.non_reac_traj_dir = None

    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds"):
        """Where the initial conditions for AMS will be written, raise error if the directory does not exist or is empty"""
        if not os.path.exists(ini_cond_dir):
            raise ValueError("""ini_cond_dir should exist, if not create it using initialconditionssampler.""")
        if len(os.listdir(ini_cond_dir)) == 0:
            raise ValueError("""ini_cond_dir should not be empty, use initialconditionssampler to write in it.""")
        self.ini_cond_dir = ini_cond_dir

    def set_progress_folder(self, progress_dir="./AMS_progress", clean=False):
        """Where the non reactive trajectories will be stored, if the directory does not exist, it will create it."""
        self.progress_dir = progress_dir
        if world.rank == 0:
            if clean and os.path.exists(self.progress_dir):
                shutil.rmtree(self.progress_dir)
            if not os.path.exists(self.progress_dir):
                os.mkdir(self.progress_dir)

    def set_non_reac_traj_dir(self, non_reac_traj_dir="./AMS_non_reactive_trajectories", clean=False):
        """Where the non reactive trajectories will be stored, if the directory does not exist, it will create it."""
        self.non_reac_traj_dir = non_reac_traj_dir
        if world.rank == 0:
            if clean and os.path.exists(self.non_reac_traj_dir):
                shutil.rmtree(self.non_reac_traj_dir)
            if not os.path.exists(self.non_reac_traj_dir):
                os.mkdir(self.non_reac_traj_dir)

    def set_alive_traj_dir(self, alive_traj_dir="./AMS_alive_trajectories", clean=False):
        """Where the alive trajectories will be stored, if the directory does not exist, it will create it.
        At the end of the run, if the estimated probability is not 0, these are reactive trajectories."""
        self.alive_traj_dir = alive_traj_dir
        if world.rank == 0:
            if clean and os.path.exists(self.alive_traj_dir):
                shutil.rmtree(self.alive_traj_dir)
            if not os.path.exists(self.alive_traj_dir):
                os.mkdir(self.alive_traj_dir)

    def _until_r_or_p(self, i):
        while not self.xi.in_r(self.dyn.atoms) and not self.xi.in_p(self.dyn.atoms):
            self.dyn.run(self.cv_interval)
            z = self.xi.rc(self.dyn.atoms)
            f = paropen(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt", "a")
            if self.xi.in_r(self.dyn.atoms):
                z = -1.0e8
            elif self.xi.in_p(self.dyn.atoms):
                z = 1.0e8
            f.write(str(z) + "\n")
            if z >= self.z_maxs[i]:
                self.z_maxs[i] = z
            f.close()

    def _initialize(self):
        """Run the N_rep replicas from the initial condition until it enters either R or P"""
        if self.ini_cond_dir is None:
            raise ValueError("""The directory of initial conditions is not defined ! Call ams.set_ini_cond_dir""")
        if self.alive_traj_dir is None:
            raise ValueError("""The directory of alive trajectories is not defined ! Call ams.set_alive_traj_dir""")
        if self.save_all is True and self.non_reac_traj_dir is None:
            raise ValueError("""The directory of non reactive trajectories is not defined ! Call ams.set_non_reac_traj_dir""")
        self.z_maxs = (np.ones(self.n_rep) * (-1.0e8)).tolist()
        i = 0
        if len(os.listdir(self.alive_traj_dir)) == 0:
            rep = -1
        else:
            rep = (len(os.listdir(self.alive_traj_dir)) // 2) - 1
            i = rep
            for j in range(i + 1):
                self.z_maxs[j] = np.max(np.loadtxt(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt"))
        while i < self.n_rep:
            if i > rep:
                ini_cond = np.random.choice([ini for ini in os.listdir(self.ini_cond_dir) if "_used" not in ini])
                atoms = read(self.ini_cond_dir + "/" + ini_cond, index=0)
                if world.rank == 0:
                    filename, file_extension = os.path.splitext(ini_cond)
                    os.rename(self.ini_cond_dir + "/" + ini_cond, self.ini_cond_dir + "/" + filename + "_used" + file_extension)
                self.dyn.atoms.set_scaled_positions(atoms.get_scaled_positions())
                self.dyn.atoms.set_momenta(atoms.get_momenta())
                self.dyn.atoms.set_calculator(self.calc)
                traj = self.dyn.closelater(Trajectory(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", mode="w", atoms=self.dyn.atoms))
            else:
                read_traj = read(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", format="traj", index=":")
                if world.rank == 0:
                    os.remove(self.alive_traj_dir + "/rep_" + str(i) + ".traj")
                self.dyn.atoms.set_scaled_positions(read_traj[-1].get_scaled_positions())
                self.dyn.atoms.set_momenta(read_traj[-1].get_momenta())
                self.dyn.atoms.set_calculator(self.calc)
                traj = self.dyn.closelater(Trajectory(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", mode="a", atoms=self.dyn.atoms))
                for at in read_traj[:-1]:
                    traj.write(at)
            z = self.xi.rc(self.dyn.atoms)
            f = paropen(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt", "a")
            if self.xi.in_r(self.dyn.atoms):
                z = -1.0e8
            elif self.xi.in_p(self.dyn.atoms):
                z = 1.0e8
            f.write(str(z) + "\n")
            f.close()
            self.dyn.attach(traj.write, interval=self.cv_interval)
            self.dyn.call_observers()
            self._until_r_or_p(i)
            self.dyn.close()
            self.dyn.observers.pop(-1)
            i += 1
            self._write_checkpoint()
        for i in range(self.n_rep):
            self.rep_weights[i].append(1 / self.n_rep)
        self.initialized = True
        self._write_checkpoint()

    def _branch_replica(self, i, j, z_kill):
        """Branch replica i by copying replica j until z_kill and run the dynamics until it reaches either R or P"""
        if self.save_all and world.rank == 0:
            os.system("cp " + self.alive_traj_dir + "/rep_" + str(i) + ".traj " + self.non_reac_traj_dir + "/rep_" + str(i) + "_killed_at_" + str(self.ams_it) + ".traj")
            json_file = paropen(self.non_reac_traj_dir + "/rep_" + str(i) + "_killed_at_" + str(self.ams_it) + "_weights.txt", "w")
            weights = {"weights": self.rep_weights[i]}
            json.dump(weights, json_file, indent=4)
            json_file.close()
        self.rep_weights[i] = []
        if world.rank == 0:
            os.remove(self.alive_traj_dir + "/rep_" + str(i) + ".traj")
            os.remove(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt")
        read_traj = read(filename=self.alive_traj_dir + "/rep_" + str(j) + ".traj", format="traj", index=":")
        k = 0
        traj = self.dyn.closelater(Trajectory(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", mode="w", atoms=read_traj[k]))
        f = paropen(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt", "a")
        while np.abs(self.xi.rc(read_traj[k]) - z_kill) <= self.rc_threshold:
            traj.write(read_traj[k])
            if self.xi.rc(read_traj[k]) >= self.z_maxs[i]:
                self.z_maxs[i] = self.xi.rc(read_traj[k])
            f.write(str(self.xi.rc(read_traj[k])) + "\n")
            k += 1
        f.write(str(self.xi.rc(read_traj[k])) + "\n")
        f.close()
        self.dyn.close()
        self.dyn.atoms.set_scaled_positions(read_traj[k].get_scaled_positions())
        self.dyn.atoms.set_momenta(read_traj[k].get_momenta())
        self.dyn.atoms.set_calculator(self.calc)
        traj = self.dyn.closelater(Trajectory(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", mode="a", atoms=self.dyn.atoms))
        self.dyn.attach(traj.write, interval=self.cv_interval)
        self.dyn.call_observers()

    def _iteration(self):
        """Perform one iteration of the AMS algorithm"""
        if np.min(self.z_maxs) >= 1.0e8:
            self.finished = True
            self.success = True
            self._write_checkpoint()
            return False
        z_kill = np.sort(self.z_maxs)[self.k_min - 1]
        self.z_kill.append(z_kill)
        killed = np.where(np.abs(self.z_maxs - z_kill) <= self.rc_threshold)[0].tolist()
        self.killed.append(killed.copy())
        alive = np.setdiff1d(np.arange(self.n_rep), killed)
        self.current_p = self.current_p * ((self.n_rep - len(self.killed[-1])) / self.n_rep)
        self._write_checkpoint()
        if len(killed) == self.n_rep:
            self.finished = True
            self.success = False
            return False
        while len(killed) > 0:
            i = killed.pop()
            j = np.random.choice(alive)
            self._branch_replica(i, j, z_kill)
            self._until_r_or_p(i)
            self.dyn.close()
            self.dyn.observers.pop(-1)
            self._write_checkpoint()
        for i in range(self.n_rep):
            self.rep_weights[i].append((self.n_rep - len(self.killed[-1])) / self.n_rep)
        return True

    def _finish_iteration(self):
        z_kill = self.z_kill[-1]
        killed = self.killed[-1].copy()
        alive = np.setdiff1d(np.arange(self.n_rep), killed)
        while len(killed) > 0:
            i = killed.pop()
            rc_traj = np.loadtxt(self.alive_traj_dir + "/rc_rep_" + str(i) + ".txt")
            if len(rc_traj.shape) == 0:
                rc_traj.reshape([1])
            if np.max(rc_traj) <= z_kill and (rc_traj[-1] <= -1.0e8 or rc_traj[-1] >= 1.0e8):
                j = np.random.choice(alive)
                self._branch_replica(i, j, z_kill)
                self._until_r_or_p(i)
            elif np.max(rc_traj) > z_kill and not (rc_traj[-1] <= -1.0e8 or rc_traj[-1] >= 1.0e8):
                read_traj = read(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", format="traj", index=":")
                self.dyn.atoms.set_scaled_positions(read_traj[-1].get_scaled_positions())
                self.dyn.atoms.set_momenta(read_traj[-1].get_momenta())
                traj = self.dyn.closelater(Trajectory(filename=self.alive_traj_dir + "/rep_" + str(i) + ".traj", mode="a", atoms=self.dyn.atoms))
                self.dyn.attach(traj.write, interval=self.cv_interval)
                self._until_r_or_p(i)
                self.dyn.close()
                self.dyn.observers.pop(-1)
            self._write_checkpoint()
        for i in range(self.n_rep):
            self.rep_weights[i].append((self.n_rep - len(self.killed[-1])) / self.n_rep)

    def _read_checkpoint(self):
        """Read the necessary information to restart an AMS run from the checkpoint file"""
        json_file = paropen(self.progress_dir + "/ams_checkpoint.txt", "r")
        checkpoint_data = json.load(json_file)
        json_file.close()
        self.z_maxs = checkpoint_data["z_maxs"]
        self.ams_it = checkpoint_data["iteration_number"]
        self.initialized = checkpoint_data["initialized"]
        self.finished = checkpoint_data["finished"]
        self.success = checkpoint_data["success"]
        self.rep_weights = checkpoint_data["rep_weights"]
        self.z_kill = checkpoint_data["z_kill"]
        self.killed = checkpoint_data["killed"]
        self.current_p = checkpoint_data["current_p"]

    def _write_checkpoint(self):
        """Write information on the current state of AMS run to be able to restart"""
        checkpoint_data = {"z_maxs": self.z_maxs, "iteration_number": self.ams_it, "initialized": self.initialized, "finished": self.finished, "success": self.success, "rep_weights": self.rep_weights, "z_kill": self.z_kill, "killed": self.killed, "current_p": self.current_p}
        json_file = paropen(self.progress_dir + "/ams_checkpoint.txt", "w")
        json.dump(checkpoint_data, json_file, indent=4)
        json_file.close()

    def run(self):
        """Run AMS, should handle the restarts correctly"""
        if os.path.exists(self.progress_dir + "/ams_checkpoint.txt"):
            parprint("Read checkpoint")
            self._read_checkpoint()
        barrier()  # Wait for all threads to read checkpoint
        if not self.initialized:
            self.dyn.observers = []
            self._initialize()
        if os.path.exists(self.progress_dir + "/ams_checkpoint.txt") and self.initialized and not self.finished and self.dyn.nsteps == 1:
            self.dyn.observers = []
            self._finish_iteration()
        continue_running = True
        with paropen(self.progress_dir + "/ams_progress.txt", "a") as progress_file:
            while continue_running:
                self.dyn.observers = []
                continue_running = self._iteration()
                np.savetxt(progress_file, np.hstack(([self.ams_it, self.current_p], self.z_maxs))[None, :])
                self.ams_it += 1
