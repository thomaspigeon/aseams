import numpy as np
import os, shutil, time
from ase.io import Trajectory, read, write
from ase.parallel import world, paropen


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
        self.run_dir = None
        self.ini_cond_dir = None

    def set_run_dir(self, run_dir="./ini_conds_md_logs"):
        """Where the md logs will be written, if the directory does not exist, it will create it."""
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir) and world.rank == 0:
            os.mkdir(self.run_dir)
        n_traj_already = len([fi for fi in os.listdir(self.run_dir) if fi.endswith(".traj")])
        traj = Trajectory(filename=self.run_dir + "/md_traj_{}.traj".format(n_traj_already), mode="a", atoms=self.dyn.atoms)
        self.trajfile = self.run_dir + "/md_traj_{}.traj".format(n_traj_already)
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
        n_ini_conds_already = len([ini for ini in os.listdir(self.ini_cond_dir) if ini.endswith(".extyxz")])
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
            while not self.xi.above_sigma(self.dyn.atoms):
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
                    list_atoms = read(self.trajfile, index=":")
                    at = list_atoms[np.random.randint(len(list_atoms))]
                    self.dyn.atoms.set_scaled_positions(at.get_scaled_positions())
                    self.dyn.atoms.set_momenta(at.get_momenta())
            n_cdt += 1
        return t_r_sigma, t_sigma_r


class InitialConditionsSamplerFromFile:
    """Class to sample initial conditions to run AMS later"""

    def __init__(self, xi, cv_interval=1):
        """An initial conditions sampler with a single replica

        Parameters:

        xi: CollectiveVariable object
            Object that allows to measure whether the dynamics is in reactant (R) state, in product (P) state and the
            progress on the transitions

        cv_interval: int
            The CV is evaluated every cv_interval time steps
        """
        if type(xi).__name__ != "CollectiveVariables":
            raise ValueError("""xi must be a CollectiveVariables object""")
        if isinstance(cv_interval, int) and cv_interval >= 0:
            self.cv_interval = cv_interval
        else:
            raise ValueError("""cv_interval must be an int >= 0""")

        self.xi = xi
        self.run_dir = None
        self.ini_cond_dir = None

    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds", clean=False):
        """Where the initial conditions for AMS will be written, if the directory does not exist, it will create it."""
        self.ini_cond_dir = ini_cond_dir
        if world.rank == 0:
            if clean and os.path.exists(self.ini_cond_dir):
                shutil.rmtree(self.ini_cond_dir)
            if not os.path.exists(self.ini_cond_dir):
                os.mkdir(self.ini_cond_dir)

    def sample(self, file):
        """Sampling initial conditions

        Parameters:

        file: str
            path to the trajectory file
        """
        if self.ini_cond_dir is None:
            raise ValueError("""The directory to store initial conditions is not defined ! Call initialconditionssampler.set_ini_cond_dir""")

        if isinstance(self.xi.cv_r, list):
            t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
            t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
        else:
            t_r_sigma = [[]]
            t_sigma_r = [[]]

        n_cdt, n_stp = 0, 0
        n_ini_conds_already = len([ini for ini in os.listdir(self.ini_cond_dir) if ini.endswith(".extyxz")])
        traj = read(file, index=":")
        n_steps = len(traj)
        while not self.xi.in_r(traj[n_stp]):
            n_stp += self.cv_interval
            if n_stp >= n_steps:
                break
        while n_stp < n_steps:
            which_r = np.where(self.xi.in_which_r(traj[n_stp]) == np.max(self.xi.in_which_r(traj[n_stp])))[0][0]
            t_r_sigma[which_r].append(0)
            while not self.xi.above_sigma(traj[n_stp]):
                n_stp += self.cv_interval
                if n_stp >= n_steps:
                    t_r_sigma[which_r].pop()
                    break
                t_r_sigma[which_r][-1] += self.cv_interval
            if n_stp >= n_steps:  # If we have quit previous loop, we should also quit the main loop
                break
            fname = self.ini_cond_dir + "/" + str(n_ini_conds_already + n_cdt + 1) + ".extxyz"
            write(fname, traj[n_stp])
            t_sigma_r[which_r].append(0)
            while not self.xi.in_r(traj[n_stp]):
                n_stp += self.cv_interval
                if n_stp >= n_steps:
                    t_sigma_r[which_r].pop()
                    break
                t_sigma_r[which_r][-1] += self.cv_interval
            n_cdt += 1
        return t_r_sigma, t_sigma_r


class FlemmingViotInitialConditionsSampler:
    """Class to sample initial conditions to run AMS later"""

    def __init__(self, dyn, xi, n_walkers=4, walker_index=0, cv_interval=1):
        """An initial conditions sampler with multiple replicas

        Parameters:

        dyn: MolecularDynamics object
            Should be a stochastic dynamics

        xi: CollectiveVariable object
            Object that allows to measure whether the dynamics is in reactant (R) state, in product (P) state and the
            progress on the transitions

        n_walkers: int
            number of walkers, positive

        waler_index: int
            index of this walker such that 0 <= waler_index < n_walkers

        cv_interval: int
            The CV is evaluated every cv_interval time steps
        """
        if type(xi).__name__ != "CollectiveVariables":
            raise ValueError("""xi must be a CollectiveVariables object""")
        if type(dyn).__name__ != "Langevin":
            raise ValueError("""dyn must be a Langevin object""")
        if isinstance(n_walkers, int) and n_walkers >= 0:
            self.n_walkers = n_walkers
        else:
            raise ValueError("""n_walkers must be an int >= 0""")
        if isinstance(walker_index, int) and (0 <= walker_index < n_walkers):
            self.w_i = walker_index
        else:
            raise ValueError("""walker_index must be an int such that 0 <= walker_index < n_walkers""")
        if isinstance(cv_interval, int) and cv_interval >= 0:
            self.cv_interval = cv_interval
        else:
            raise ValueError("""cv_interval must be an int >= 0""")
        xi.test_the_collective_variables(dyn.atoms)
        self.xi = xi
        self.dyn = dyn
        self.dyn.nsteps = 1
        self.run_dir = None
        self.ini_cond_dir = None

    def _set_initialcond_dyn(self, atoms):
        """
        Set atomic position and momenta of a dynamic
        """
        self.dyn.atoms.set_scaled_positions(atoms.get_scaled_positions())
        self.dyn.atoms.set_momenta(atoms.get_momenta())

    def set_run_dir(self, run_dir="./ini_conds_walker_"):
        """Where the md logs will be written, if the directory does not exist, it will create it."""
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir + str(self.w_i)) and world.rank == 0:
            os.mkdir(self.run_dir + str(self.w_i))
        n_traj_already = len([fi for fi in os.listdir(self.run_dir + str(self.w_i)) if fi.endswith(".traj")])
        t = []
        for i in range(n_traj_already):
            trajfile = self.run_dir + str(self.w_i) + "/md_traj_{}.traj".format(i)
            list_atoms = read(trajfile, index=":")
            if i == 0:
                t.append(len(list_atoms))
            else:
                t.append(len(list_atoms) + t[-1])
            if i == n_traj_already - 1:
                self._set_initialcond_dyn(list_atoms[-1])
        if len(t) > 0:
            self.dyn.nsteps = t[-1]
        trajfile = self.run_dir + str(self.w_i) + "/md_traj_{}.traj".format(n_traj_already)
        traj = Trajectory(filename=trajfile, mode="a", atoms=self.dyn.atoms)
        self.dyn.attach(traj.write, interval=self.cv_interval)

    def set_ini_cond_dir(self, ini_cond_dir="./ini_conds"):
        """Where the initial conditions for AMS will be written, if the directory does not exist, it will create it."""
        self.ini_cond_dir = ini_cond_dir
        if not os.path.exists(self.ini_cond_dir) and world.rank == 0 and self.w_i == 0:
            os.mkdir(self.ini_cond_dir)

    def _branch_FV_paricle(self):
        traj_idx = None
        branch_rep_number = np.random.choice(np.setdiff1d(np.arange(self.n_walkers), self.w_i))
        while traj_idx is None:
            n_traj_already = len([fi for fi in os.listdir(self.run_dir + str(branch_rep_number)) if fi.endswith(".traj")])
            t = []
            for i in range(n_traj_already):
                trajfile = self.run_dir + str(branch_rep_number) + "/md_traj_{}.traj".format(i)
                list_atoms = read(trajfile, index=":")
                if i == 0:
                    t.append(len(list_atoms))
                    if self.dyn.nsteps < t[-1]:
                        traj_idx = i
                else:
                    t.append(len(list_atoms) + t[-1])
                    if t[-2] <= self.dyn.nsteps < t[-1]:
                        traj_idx = i
                f = paropen(self.run_dir + str(self.w_i) + "/FV_history.txt", "a")
                f.write("Attempt branching replica " + str(self.w_i) + " from replica index: " + str(branch_rep_number) + " at time " + str(self.dyn.nsteps) + " \n")
                if traj_idx is not None:
                    f.write("Succesful branching\n \n")
                    f.close()
                else:
                    f.write("Cannot branch as replica " + str(branch_rep_number) + "  did not reach time " + str(self.dyn.nsteps) + "\n")
                    f.write("Going to sleep for 1 minute\n")
                    f.close()
                    time.sleep(60)
        trajfile = self.run_dir + str(branch_rep_number) + "/md_traj_{}.traj".format(traj_idx)
        list_atoms = read(trajfile, index=":")
        at = list_atoms[self.dyn.nsteps - t[traj_idx - 1]]
        self.dyn.atoms.set_scaled_positions(at.get_scaled_positions())
        self.dyn.atoms.set_momenta(at.get_momenta())

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
        while not os.path.exists(self.ini_cond_dir):
            time.sleep(1)
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
        n_ini_conds_already = len([fi for fi in os.listdir(self.ini_cond_dir) if fi.startswith("walker_" + str(self.w_i) + "_")])
        if isinstance(self.xi.cv_r, list):
            t_r_sigma = [[] for i in range(len(self.xi.cv_r))]
            t_sigma_r = [[] for i in range(len(self.xi.cv_r))]
        else:
            t_r_sigma = [[]]
            t_sigma_r = [[]]
        while not self.xi.in_r(self.dyn.atoms):
            self.dyn.run(self.cv_interval)
            n_stp += self.cv_interval
            if self.xi.is_out_of_r_zone(self.dyn.atoms):
                self._branch_FV_paricle()
        while n_cdt < n_conditions or n_stp < n_steps:
            which_r = np.where(self.xi.in_which_r(self.dyn.atoms) == np.max(self.xi.in_which_r(self.dyn.atoms)))[0][0]
            t_r_sigma[which_r].append(0)
            t_sigma_r[which_r].append(0)
            while not self.xi.above_sigma(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                t_r_sigma[which_r][-1] += self.cv_interval
            fname = self.ini_cond_dir + "/walker_" + str(self.w_i) + "_ini_cond_" + str(n_ini_conds_already + n_cdt + 1) + ".extxyz"
            write(fname, self.dyn.atoms)
            while not self.xi.in_r(self.dyn.atoms):
                self.dyn.run(self.cv_interval)
                n_stp += self.cv_interval
                t_sigma_r[which_r][-1] += self.cv_interval
                if self.xi.is_out_of_r_zone(self.dyn.atoms):
                    self._branch_FV_paricle()
            n_cdt += 1
        return t_r_sigma, t_sigma_r
