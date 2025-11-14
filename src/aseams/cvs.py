import inspect
import numpy as np

class CollectiveVariables:
    """Class to gather collective variables used to sample initial conditions and run AMS"""

    def __init__(self, cv_r, cv_p, reaction_coordinate, rc_grad=None):
        """
        Parameters:

        cv_r: function taking as argument an atoms object and returning a float or a list of such functions

        cv_p: function taking as argument an atoms object and returning a float or a list of such functions

        reaction_coordinate: function taking as argument an atoms object and returning a float

        rc_grad: function that return the gradient of the reaction coordinate with respect to the positions of atoms
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
        if rc_grad is not None:
            if inspect.isfunction(rc_grad):
                self.rc_grad = rc_grad
            else:
                raise ValueError("""reaction_coordinate gradient must be a function""")
        else:
            self.rc_grad = None
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
                        if not (isinstance(sigma_r_level[i], float) and sigma_r_level[i] <= self.in_r_boundary[i]):
                            raise ValueError(
                                """When cv_r is a list of functions and r_crit[i] is "above",
                                                sigma_r_level[i] must be a float smaller than in_r_boundary[i]"""
                            )
                        else:
                            self.sigma_r_level.append(sigma_r_level[i])
                    elif self.r_crit[i] == "below":
                        if not (isinstance(sigma_r_level[i], float) and sigma_r_level[i] >= self.in_r_boundary[i]):
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
                        if not (isinstance(out_of_r_zone[i], float) and out_of_r_zone[i] <= self.sigma_r_level[i]):
                            raise ValueError(
                                """When cv_r is a list of functions and r_crit[i] is "above",
                                                out_of_r_zone[i] must be a float smaller than sigma_r_level[i]"""
                            )
                        else:
                            self.out_of_r_zone.append(out_of_r_zone[i])
                    elif self.r_crit[i] == "below":
                        if not (isinstance(out_of_r_zone[i], float) and out_of_r_zone[i] >= self.sigma_r_level[i]):
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
                if self.r_crit[i] == "above":
                    in_r.append(self.cv_r[i](atoms) >= self.in_r_boundary[i])
                if self.r_crit[i] == "below":
                    in_r.append(self.cv_r[i](atoms) <= self.in_r_boundary[i])
                if self.r_crit[i] == "between":
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
                if self.r_crit[i] == "above":
                    above_sigma_r.append(self.cv_r[i](atoms) < self.sigma_r_level[i])
                if self.r_crit[i] == "below":
                    above_sigma_r.append(self.cv_r[i](atoms) > self.sigma_r_level[i])
                if self.r_crit[i] == "between":
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
                if self.r_crit[i] == "above":
                    out.append(self.cv_r[i](atoms) < self.out_of_r_zone[i])
                if self.r_crit[i] == "below":
                    out.append(self.cv_r[i](atoms) > self.out_of_r_zone[i])
                if self.r_crit[i] == "between":
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
                        self.p_crit.append(p_crit[i])

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
                if self.p_crit[i] == "above":
                    in_p.append(self.cv_p[i](atoms) >= self.in_p_boundary[i])
                if self.p_crit[i] == "below":
                    in_p.append(self.cv_p[i](atoms) <= self.in_p_boundary[i])
                if self.p_crit[i] == "between":
                    in_p.append(self.in_p_boundary[i][0] <= self.cv_p[i](atoms) <= self.in_p_boundary[i][1])
            return in_p
        else:
            if self.p_crit == "above":
                return [self.cv_p(atoms) >= self.in_p_boundary]
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
        if self.rc_grad is not None:
            if not self.rc_grad(atoms).shape == (len(atoms), 3):
                raise ValueError(
                """The function reaction_coordinate gradient does not returns a  tensor or a float of shape 
                (len(atoms), 3)  cannot bias the initial conditions, the CollectiveVariables object is not properly set 
                and you should modify rc_grad"""
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
